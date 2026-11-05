"""Does token-adaptive MLP computation survive at hardware-friendly (group) granularity?

Each block's 3072 hidden neurons are split into G equal groups (balanced co-activation k-means,
or a random partition as control). A token then runs only g of the G groups (fc1 rows + fc2
columns of those groups), which a GPU can execute as dense slices. Relative error of the MLP
update vs. the full MLP is measured for three ways of choosing the groups:
    static  - one fixed set for all tokens (top mean contribution on split A)
    router  - per token, top-g of a linear predictor from the MLP input (768 -> G); deployable
    oracle  - per token, top-g by the true group outputs; upper bound, needs the full MLP
Clustering / router / static sets are fitted on image split A; errors are measured on split B.

    python analysis/mlp_study/mlp_groups.py --ckpt best.pth --cfg <dumped config.py> --device cpu
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, osp.dirname(__file__))
from build_backbone import IMAGENET_NORM, ROOT, build_vituwa  # noqa: E402
from data_utils import load_split, preprocess, valid_token_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--cfg', required=True)
ap.add_argument('--data', default=osp.join(ROOT, 'datasets', 'USIS10K'))
ap.add_argument('--split', default='val')
ap.add_argument('--n-images', type=int, default=80, help='first half = fit (A), second half = evaluate (B)')
ap.add_argument('--tokens-per-image', type=int, default=192, help='stored split-A tokens per image')
ap.add_argument('--eval-tokens', type=int, default=1024, help='random valid tokens per split-B image')
ap.add_argument('--groups', type=int, nargs='+', default=[8, 16])
ap.add_argument('--keep', type=float, nargs='+', default=[0.125, 0.25, 0.5])
ap.add_argument('--device', default='cpu')
ap.add_argument('--out', default=osp.join(ROOT, 'analysis', 'results', 'mlp'))
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)

model, _ = build_vituwa(args.cfg, args.ckpt, dev)
blocks = model.blocks
nb, hid = len(blocks), blocks[0].mlp.fc1.out_features
cap = {}
for i, b in enumerate(blocks):
    b.mlp.register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(f'u{i}', inp[0]))
    b.mlp.act.register_forward_hook(lambda m, inp, out, i=i: cap.__setitem__(f'h{i}', out))
W2g = [(b.gamma2.detach()[:, None] if b.layer_scale else 1.) * b.mlp.fc2.weight.detach() for b in blocks]  # (768, hid)
wnorm = [w.norm(dim=0) for w in W2g]

imgs, _ = load_split(args.data, args.split, args.n_images, args.seed)
A, Bset = imgs[:len(imgs) // 2], imgs[len(imgs) // 2:]
print(f'{len(A)} fit images, {len(Bset)} eval images, device={dev}', flush=True)


def run(info):
    x, _, hw = preprocess(osp.join(args.data, args.split, info['file_name']), IMAGENET_NORM)
    with torch.no_grad():
        model(x.to(dev))
    return torch.from_numpy(valid_token_mask(hw, (x.shape[2] // 16, x.shape[3] // 16))).to(dev)


# ---------------------------------------------------------------- phase A: collect tokens
t0 = time.time()
U = [[] for _ in range(nb)]
Hs = [[] for _ in range(nb)]
for n, info in enumerate(A):
    v = run(info)
    idx = torch.nonzero(v).squeeze(1)
    pick = idx[torch.randperm(len(idx), device=dev)[:args.tokens_per_image]]
    for i in range(nb):
        U[i].append(cap[f'u{i}'][0][pick].half().cpu())
        Hs[i].append(cap[f'h{i}'][0][pick].half().cpu())
    if (n + 1) % 10 == 0:
        print(f'  A {n + 1}/{len(A)}  {(time.time() - t0) / (n + 1):.1f} s/img', flush=True)
U = [torch.cat(u).float() for u in U]
Hs = [torch.cat(h).float() for h in Hs]


def balanced_kmeans(feat, G, iters=15):
    """feat: (hid, d) neuron profiles (L2-normalised). Equal-size clusters via greedy capacity assignment."""
    n = feat.shape[0]
    capg = n // G
    cent = feat[torch.from_numpy(rng.choice(n, G, replace=False))]
    for _ in range(iters):
        sim = feat @ cent.T                                      # (n, G)
        order = np.argsort(-sim.flatten().numpy(), kind='stable')
        a = np.full(n, -1)
        fill = np.zeros(G, int)
        for o in order:
            j, g = divmod(int(o), G)
            if a[j] < 0 and fill[g] < capg:
                a[j] = g
                fill[g] += 1
        assign = torch.from_numpy(a).long()
        cent = torch.stack([feat[assign == g].mean(0) for g in range(G)])
        cent = cent / cent.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return assign


def group_outputs(h, W, assign, G):
    """h (N, hid), W (768, hid) -> per-group MLP updates (N, G, 768)."""
    return torch.stack([h[:, assign == g] @ W[:, assign == g].T for g in range(G)], 1)


def fit_router(u, s):
    """ridge u (N,768) -> s (N,G) with bias; returns (w, b)."""
    mu = u.mean(0)
    X = u - mu
    lam = 1e-2 * X.pow(2).sum(1).mean()
    w = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ (s - s.mean(0)))
    return w, s.mean(0) - mu @ w


# ---------------------------------------------------------------- fit per block / G / partition
fits = {}
for i in range(nb):
    Wc = W2g[i].cpu().float()
    contrib = Hs[i].abs() * wnorm[i].cpu()                      # (N, hid)
    prof = contrib.T / contrib.T.norm(dim=1, keepdim=True).clamp_min(1e-8)
    for G in args.groups:
        parts = {'coact': balanced_kmeans(prof, G),
                 'random': torch.from_numpy(rng.permutation(np.arange(hid) % G)).long()}
        for pname, assign in parts.items():
            s = group_outputs(Hs[i], Wc, assign, G).norm(dim=2)  # (N, G) true group-output norms
            w, b = fit_router(U[i], s)
            fits[(i, G, pname)] = dict(assign=assign.to(dev), static=torch.argsort(s.mean(0), descending=True).to(dev),
                                       w=w.to(dev), b=b.to(dev))
    print(f'  fitted block {i}', flush=True)
del U, Hs

# ---------------------------------------------------------------- phase B: evaluate
err = {}  # (i, G, part, method, keep) -> [sum, n]
for n, info in enumerate(Bset):
    v = run(info)
    idx = torch.nonzero(v).squeeze(1)
    v = idx[torch.randperm(len(idx), device=dev)[:args.eval_tokens]]
    for i in range(nb):
        u = cap[f'u{i}'][0][v].float()
        h = cap[f'h{i}'][0][v].float()
        full = h @ W2g[i].T
        fnorm = full.norm(dim=1).clamp_min(1e-6)
        for G in args.groups:
            for pname in ('coact', 'random'):
                f = fits[(i, G, pname)]
                Y = group_outputs(h, W2g[i], f['assign'], G)    # (N, G, 768)
                true = Y.norm(dim=2)
                pred = u @ f['w'] + f['b']
                for keep in args.keep:
                    g = max(1, int(round(G * keep)))
                    sel = {'oracle': true.topk(g, dim=1).indices, 'router': pred.topk(g, dim=1).indices,
                           'static': f['static'][:g].expand(len(h), g)}
                    for m, ix in sel.items():
                        out = torch.gather(Y, 1, ix[:, :, None].expand(-1, -1, Y.shape[2])).sum(1)
                        e = ((out - full).norm(dim=1) / fnorm).sum().item()
                        acc = err.setdefault((i, G, pname, m, keep), [0.0, 0])
                        acc[0] += e
                        acc[1] += len(h)
    if (n + 1) % 10 == 0:
        print(f'  B {n + 1}/{len(Bset)}  {(time.time() - t0) / (len(A) + n + 1):.1f} s/img', flush=True)

res = dict(meta=dict(ckpt=args.ckpt, n_fit=len(A), n_eval=len(Bset), groups=args.groups, keep=args.keep,
                     router_macs_per_token={G: 768 * G for G in args.groups}, mlp_macs_per_token=2 * 768 * hid),
           errors={f'blk{i}_G{G}_{p}_{m}_keep{k}': v[0] / v[1] for (i, G, p, m, k), v in err.items()})
json.dump(res, open(osp.join(args.out, 'mlp_groups.json'), 'w'), indent=1)

for G in args.groups:
    for keep in args.keep:
        print(f'\nG={G} keep={keep:.3f} ({int(round(G * keep))}/{G} groups): relative MLP-update error  '
              'static | router | oracle   (random partition: router | oracle)')
        for i in range(nb):
            e = lambda p, m: err[(i, G, p, m, keep)][0] / err[(i, G, p, m, keep)][1]  # noqa: E731
            print(f'  blk{i:2d}  {e("coact", "static"):.3f} | {e("coact", "router"):.3f} | {e("coact", "oracle"):.3f}'
                  f'      {e("random", "router"):.3f} | {e("random", "oracle"):.3f}')
print('saved', osp.join(args.out, 'mlp_groups.json'))
