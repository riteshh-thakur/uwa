"""Inference-only probe of the 12 ViT-block MLPs of a trained ViT-UWA-B (no training).

H1 redundancy   per-neuron activation rate / contribution, importance concentration,
                output error when keeping the top-k% neurons: static (one global set,
                chosen on image split A, measured on split B) vs per-token top-k.
H2 degradation  per-token degradation descriptors (local noise, contrast, dark channel,
                red attenuation) and foreground: linear-probe R^2 from the MLP hidden layer,
                neuron-usage shift between the cleanest and most degraded token quartiles.
H3 frequency    spatial spectrum on the token grid of the block input, the attention update
                and the MLP update (high-frequency energy fraction per block).

    python analysis/mlp_study/mlp_probe.py --ckpt best.pth --cfg <dumped config.py> \
        --n-images 100 --device mps --out analysis/results/mlp
"""
import argparse
import json
import math
import os
import os.path as osp
import random
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # bicubic pos-embed resize has no MPS kernel
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__))
from build_backbone import IMAGENET_NORM, ROOT, USIS_NORM, build_vituwa  # noqa: E402

DESC = ['noise', 'contrast', 'dark', 'red_ratio', 'fg']
KEEP = [0.05, 0.1, 0.25, 0.5, 0.75]

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True, help="detector checkpoint, or 'random' for a smoke test")
ap.add_argument('--cfg', required=True)
ap.add_argument('--data', default=osp.join(ROOT, 'datasets', 'USIS10K'))
ap.add_argument('--split', default='val')
ap.add_argument('--n-images', type=int, default=100)
ap.add_argument('--tokens-per-image', type=int, default=64, help='stored tokens per image for probes / CKA')
ap.add_argument('--norm', choices=['imagenet', 'usis'], default='imagenet')
ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
ap.add_argument('--out', default=osp.join(ROOT, 'analysis', 'results', 'mlp'))
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed), random.seed(args.seed), np.random.seed(args.seed)
dev = torch.device(args.device)
norm = IMAGENET_NORM if args.norm == 'imagenet' else USIS_NORM


# ----------------------------------------------------------------------------- data
def load_split():
    ann = json.load(open(osp.join(args.data, 'multi_class_annotations', f'multi_class_{args.split}_annotations.json')))
    by_img = {}
    for a in ann['annotations']:
        by_img.setdefault(a['image_id'], []).append(a)
    imgs = sorted(ann['images'], key=lambda d: d['id'])
    random.Random(args.seed).shuffle(imgs)
    return imgs[:args.n_images], by_img


def preprocess(info, anns):
    bgr = cv2.imread(osp.join(args.data, args.split, info['file_name']))
    h, w = bgr.shape[:2]
    s = min(1333 / max(h, w), 800 / min(h, w))  # mmcv imrescale, keep_ratio (1333, 800)
    nh, nw = int(h * s + 0.5), int(w * s + 0.5)
    rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB).astype(np.float32)
    fg = np.zeros((nh, nw), np.uint8)
    for a in anns:
        seg = a['segmentation']
        if isinstance(seg, list):
            for poly in seg:
                p = (np.asarray(poly, np.float32).reshape(-1, 2) * [nw / w, nh / h]).round().astype(np.int32)
                cv2.fillPoly(fg, [p], 1)
    x = (rgb - np.array(norm['mean'], np.float32)) / np.array(norm['std'], np.float32)
    H, W = math.ceil(nh / 32) * 32, math.ceil(nw / 32) * 32
    xp = np.zeros((H, W, 3), np.float32)
    xp[:nh, :nw] = x
    return torch.from_numpy(xp).permute(2, 0, 1)[None], rgb, fg, (nh, nw)


def token_descriptors(rgb, fg, hw, grid):
    """Per 16x16 patch descriptors on the valid (unpadded) region; returns (Ht*Wt, 5) and valid mask."""
    nh, nw = hw
    Hg, Wg = grid
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    out = np.full((Hg, Wg, len(DESC)), np.nan, np.float32)
    for i in range(Hg):
        for j in range(Wg):
            y0, x0 = i * 16, j * 16
            if y0 + 16 > nh or x0 + 16 > nw:
                continue  # patch touches padding
            L = lum[y0:y0 + 16, x0:x0 + 16]
            P = rgb[y0:y0 + 16, x0:x0 + 16]
            hh = (L[0::2, 0::2] - L[0::2, 1::2] - L[1::2, 0::2] + L[1::2, 1::2]) / 2
            out[i, j] = [np.median(np.abs(hh)) / 0.6745, L.std(), P.min() / 255.,
                         P[..., 0].mean() / (P[..., 1:].mean() + 1e-3), fg[y0:y0 + 16, x0:x0 + 16].mean()]
    d = out.reshape(-1, len(DESC))
    return d, ~np.isnan(d[:, 0])


# ----------------------------------------------------------------------------- model + hooks
model, bcfg = build_vituwa(args.cfg, None if args.ckpt == 'random' else args.ckpt, dev)
blocks = model.blocks
nb = len(blocks)
hid = blocks[0].mlp.fc1.out_features
cap = {}


def hook(name):
    def fn(mod, inp, out):
        cap[name] = (inp[0], out)
    return fn


for i, b in enumerate(blocks):
    b.register_forward_hook(hook(f'blk{i}'))
    b.attn.register_forward_hook(hook(f'attn{i}'))
    b.mlp.act.register_forward_hook(hook(f'act{i}'))
    b.mlp.register_forward_hook(hook(f'mlp{i}'))

W2 = [b.mlp.fc2.weight.detach() for b in blocks]                        # (768, hid)
g1 = [b.gamma1.detach() if b.layer_scale else torch.ones(b.mlp.fc2.out_features, device=dev) for b in blocks]
g2 = [b.gamma2.detach() if b.layer_scale else torch.ones(b.mlp.fc2.out_features, device=dev) for b in blocks]
wnorm = [(g2[i][:, None] * W2[i]).norm(dim=0) for i in range(nb)]       # contribution scale per neuron

z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)  # noqa: E731  (MPS has no float64; counts < 2^24)
st = {k: [z(hid) for _ in range(nb)] for k in ('act_rate', 'abs', 'imp', 'q1_rate', 'q4_rate', 'fg_rate', 'bg_rate')}
sc = {k: [0.0] * nb for k in ('tok_active', 'rel_attn', 'rel_mlp', 'n_tok', 'n_q1', 'n_q4', 'n_fg', 'n_bg')}
recon = {(m, k): [[0.0, 0.0] for _ in range(nb)] for m in ('static', 'token') for k in KEEP}  # [sum rel err, n]
spec = {k: [[0.0, 0] for _ in range(nb)] for k in ('x_in', 'attn', 'mlp')}
store_h = [[] for _ in range(nb)]
store_m = [[] for _ in range(nb)]
store_d, store_img = [], []
static_sets = None


def hf_fraction(t, Hv, Wv, Hg, Wg):
    """t: (N, C) tokens on the (Hg, Wg) grid -> fraction of non-DC energy above half Nyquist (valid crop)."""
    g = t.float().cpu().reshape(Hg, Wg, -1)[:Hv, :Wv].permute(2, 0, 1)  # CPU: MPS has no complex FFT
    g = g - g.mean(dim=(1, 2), keepdim=True)
    win = torch.hann_window(Hv, periodic=False)[:, None] * torch.hann_window(Wv, periodic=False)[None]
    p = torch.fft.fft2(g * win).abs().pow(2).sum(0)
    fy = torch.fft.fftfreq(Hv)[:, None].abs() / 0.5
    fx = torch.fft.fftfreq(Wv)[None].abs() / 0.5
    r = torch.sqrt(fy ** 2 + fx ** 2)
    tot = p.sum() - p[0, 0]
    return float(p[r > 0.5].sum() / tot) if tot > 0 else 0.0


# ----------------------------------------------------------------------------- pass 0: descriptor quartiles
imgs, by_img = load_split()
split_a = set(i['id'] for i in imgs[:len(imgs) // 2])                    # importance from A, error measured on B
print(f'{len(imgs)} {args.split} images (A={len(split_a)}, B={len(imgs) - len(split_a)}), norm={args.norm}, device={dev}')
pre, all_d = [], []
for info in imgs:
    x, rgb, fg, hw = preprocess(info, by_img.get(info['id'], []))
    grid = (x.shape[2] // 16, x.shape[3] // 16)
    d, valid = token_descriptors(rgb, fg, hw, grid)
    pre.append((info, x, d, valid, grid, hw))
    all_d.append(d[valid])
all_d = np.concatenate(all_d)
q_noise = np.quantile(all_d[:, 0], [0.25, 0.75])
deg_score = lambda d: d[:, 0]  # noqa: E731  quartiles on local noise sigma (primary degradation axis)

# ----------------------------------------------------------------------------- main pass
t0 = time.time()
for n_img, (info, x, d, valid, grid, hw) in enumerate(pre):
    in_a = info['id'] in split_a
    if not in_a and static_sets is None:     # A finished: freeze static importance ranking
        static_sets = [torch.argsort(st['imp'][i], descending=True) for i in range(nb)]
    with torch.no_grad():
        model(x.to(dev))
    Hg, Wg = grid
    Hv, Wv = hw[0] // 16, hw[1] // 16
    vmask = torch.from_numpy(valid).to(dev)
    s = deg_score(d)
    q1 = torch.from_numpy(valid & (s <= q_noise[0])).to(dev)
    q4 = torch.from_numpy(valid & (s >= q_noise[1])).to(dev)
    fgm = torch.from_numpy(valid & (d[:, 4] > 0.5)).to(dev)
    bgm = torch.from_numpy(valid & (np.nan_to_num(d[:, 4], nan=1) < 0.05)).to(dev)
    vidx = np.flatnonzero(valid)
    pick = np.random.choice(vidx, min(args.tokens_per_image, len(vidx)), replace=False)
    store_d.append(d[pick]), store_img.append(np.full(len(pick), int(in_a)))
    for i in range(nb):
        x_in = cap[f'blk{i}'][0][0]                  # (N, 768)
        h = cap[f'act{i}'][1][0]                     # (N, hid) GELU output
        mo = cap[f'mlp{i}'][1][0]                    # (N, 768) before gamma2
        dm, da = g2[i] * mo, g1[i] * cap[f'attn{i}'][1][0]
        hv, xv = h[vmask], x_in[vmask]
        pos = (hv > 0).float()
        st['act_rate'][i] += pos.sum(0)
        st['abs'][i] += hv.abs().float().sum(0)
        st['imp'][i] += (hv.abs() * wnorm[i]).float().sum(0)
        st['q1_rate'][i] += (h[q1] > 0).float().sum(0)
        st['q4_rate'][i] += (h[q4] > 0).float().sum(0)
        st['fg_rate'][i] += (h[fgm] > 0).float().sum(0)
        st['bg_rate'][i] += (h[bgm] > 0).float().sum(0)
        sc['tok_active'][i] += float(pos.mean(1).sum())
        xn = xv.norm(dim=1).clamp_min(1e-6)
        sc['rel_mlp'][i] += float((dm[vmask].norm(dim=1) / xn).sum())
        sc['rel_attn'][i] += float((da[vmask].norm(dim=1) / xn).sum())
        sc['n_tok'][i] += int(vmask.sum())
        for k, m in (('n_q1', q1), ('n_q4', q4), ('n_fg', fgm), ('n_bg', bgm)):
            sc[k][i] += int(m.sum())
        for k, t in (('x_in', x_in), ('attn', da), ('mlp', dm)):
            if Hv >= 8 and Wv >= 8:
                spec[k][i][0] += hf_fraction(t, Hv, Wv, Hg, Wg)
                spec[k][i][1] += 1
        if not in_a:                                  # reconstruction error on split B
            contrib = hv.abs() * wnorm[i]            # (Nv, hid)
            full = (hv @ W2[i].T) * g2[i]
            fnorm = full.norm(dim=1).clamp_min(1e-6)
            for k in KEEP:
                kk = int(hid * k)
                ms = torch.zeros(hid, device=dev)
                ms[static_sets[i][:kk]] = 1
                es = (((hv * ms) @ W2[i].T) * g2[i] - full).norm(dim=1) / fnorm
                top = contrib.topk(kk, dim=1).indices
                mt = torch.zeros_like(hv).scatter_(1, top, 1.)
                et = (((hv * mt) @ W2[i].T) * g2[i] - full).norm(dim=1) / fnorm
                recon[('static', k)][i][0] += float(es.sum()); recon[('static', k)][i][1] += len(es)
                recon[('token', k)][i][0] += float(et.sum()); recon[('token', k)][i][1] += len(et)
        pk = torch.from_numpy(pick).to(dev)
        store_h[i].append(h[pk].half().cpu())
        store_m[i].append(dm[pk].half().cpu())
    if n_img == 0 or (n_img + 1) % 10 == 0:
        print(f'  {n_img + 1}/{len(pre)} images, {(time.time() - t0) / (n_img + 1):.1f} s/img', flush=True)


# ----------------------------------------------------------------------------- summaries
def ridge_r2(X, y, tr, te):
    X = torch.nan_to_num(X.float(), nan=0., posinf=0., neginf=0.)
    mu = X[tr].mean(0)
    X = (X - mu) / X[tr].std().clamp_min(1e-6)  # one global scale: per-feature scaling blows up near-constant sparse neurons
    Xt = X[tr].double()
    evals, evecs = torch.linalg.eigh(Xt.T @ Xt)                # PCA on train tokens only
    X = X @ evecs[:, -min(256, X.shape[1]):].float()           # 256 components: far fewer dims than tokens
    y = torch.as_tensor(y, dtype=torch.float32)
    ym = y[tr].mean()
    itr = np.flatnonzero(tr)
    fit, val = itr[: int(len(itr) * 0.8)], itr[int(len(itr) * 0.8):]
    best = None
    for lam in (1e0, 1e1, 1e2, 1e3, 1e4, 1e5):
        for a, b in ((fit, val), (itr, np.flatnonzero(te))):
            A = X[a]
            w = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ (y[a] - ym))
            pred = (X[b] @ w + ym).clamp(float(y[a].min()), float(y[a].max()))  # stay inside the training target range
            r2 = 1 - float(((pred - y[b]) ** 2).sum() / ((y[b] - y[b].mean()) ** 2).sum())
            if b is val:
                score = r2
            else:
                test_r2 = r2
        if best is None or score > best[0]:
            best = (score, test_r2, lam)
    return best[1]


def linear_cka(a, b):
    a = a.float() - a.float().mean(0)
    b = b.float() - b.float().mean(0)
    return float((a.T @ b).norm() ** 2 / ((a.T @ a).norm() * (b.T @ b).norm()))


D = np.concatenate(store_d)
is_a = np.concatenate(store_img).astype(bool)
targets = {'noise': np.log(D[:, 0] + 1e-3), 'contrast': np.log(D[:, 1] + 1e-3), 'dark': D[:, 2],
           'red_ratio': np.log(D[:, 3] + 1e-3), 'fg': D[:, 4]}
res = dict(meta=dict(ckpt=args.ckpt, n_images=len(pre), split=args.split, norm=args.norm, hidden=hid,
                     tokens_per_image=args.tokens_per_image, noise_quartiles=q_noise.tolist(),
                     sec_per_image=(time.time() - t0) / len(pre)), blocks=[])
Hst = [torch.cat(store_h[i]) for i in range(nb)]
Mst = [torch.cat(store_m[i]) for i in range(nb)]
for i in range(nb):
    n = sc['n_tok'][i]
    rate = (st['act_rate'][i] / n).cpu().numpy()
    imp = st['imp'][i].cpu().numpy() / n
    srt = np.sort(imp)[::-1]
    cum = np.cumsum(srt) / srt.sum()
    gini = 1 - 2 * np.sum(np.cumsum(np.sort(imp)) / imp.sum()) / len(imp) + 1 / len(imp)
    r1 = (st['q1_rate'][i] / max(sc['n_q1'][i], 1)).cpu().numpy()
    r4 = (st['q4_rate'][i] / max(sc['n_q4'][i], 1)).cpu().numpy()
    rf = (st['fg_rate'][i] / max(sc['n_fg'][i], 1)).cpu().numpy()
    rb = (st['bg_rate'][i] / max(sc['n_bg'][i], 1)).cpu().numpy()
    top = lambda r: set(np.argsort(r)[::-1][: hid // 10])  # noqa: E731
    jac = lambda a, b: len(top(a) & top(b)) / len(top(a) | top(b))  # noqa: E731
    blk = dict(
        block=i, windowed=bool(bcfg['window_attn'][i]),
        act_rate_mean=float(rate.mean()), token_active_frac=sc['tok_active'][i] / n,
        rare_neurons_lt1pct=int((rate < 0.01).sum()), frequent_neurons_gt50pct=int((rate > 0.5).sum()),
        neurons_for_50_90_99pct_importance=[int(np.searchsorted(cum, q) + 1) for q in (0.5, 0.9, 0.99)],
        importance_gini=float(gini),
        rel_update_norm=dict(attn=sc['rel_attn'][i] / n, mlp=sc['rel_mlp'][i] / n),
        recon_err={f'{m}_{int(k * 100)}': recon[(m, k)][i][0] / max(recon[(m, k)][i][1], 1)
                   for m in ('static', 'token') for k in KEEP},
        degradation=dict(
            top10_jaccard_clean_vs_degraded=jac(r1, r4), mean_abs_rate_shift=float(np.abs(r4 - r1).mean()),
            neurons_rate_ratio_gt2=int(((r4 + 1e-3) / (r1 + 1e-3) > 2).sum() + ((r1 + 1e-3) / (r4 + 1e-3) > 2).sum())),
        foreground=dict(top10_jaccard_fg_vs_bg=jac(rf, rb), mean_abs_rate_shift=float(np.abs(rf - rb).mean())),
        probe_r2={k: ridge_r2(Hst[i], v, is_a, ~is_a) for k, v in targets.items()},
        hf_fraction={k: spec[k][i][0] / max(spec[k][i][1], 1) for k in spec},
        cka_mlp_out_next=linear_cka(Mst[i], Mst[i + 1]) if i + 1 < nb else None,
    )
    res['blocks'].append(blk)
    print(f"blk{i:2d} act={blk['act_rate_mean']:.3f} n90={blk['neurons_for_50_90_99pct_importance'][1]:4d} "
          f"err50 static={blk['recon_err']['static_50']:.3f} token={blk['recon_err']['token_50']:.3f} "
          f"R2noise={blk['probe_r2']['noise']:.2f} jacQ1Q4={blk['degradation']['top10_jaccard_clean_vs_degraded']:.2f} "
          f"HF x/attn/mlp={blk['hf_fraction']['x_in']:.3f}/{blk['hf_fraction']['attn']:.3f}/{blk['hf_fraction']['mlp']:.3f}")
    np.save(osp.join(args.out, f'neuron_stats_blk{i}.npy'), np.stack([rate, imp, r1, r4, rf, rb]))

json.dump(res, open(osp.join(args.out, 'mlp_probe.json'), 'w'), indent=1)
print('saved', osp.join(args.out, 'mlp_probe.json'))
