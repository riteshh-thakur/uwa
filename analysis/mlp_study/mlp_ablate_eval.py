"""Block-wise MLP interventions with full detection / segmentation evaluation (GPU, mmdet env).

Runs on a fixed random val subset; a disjoint calibration subset provides the dataset-mean MLP output
(for mean ablation) and the static neuron ranking (for top-k pruning). Interventions:
    base                 unmodified model
    zero:i               block i's MLP skipped (update = 0; also skips its compute -> latency)
    mean:i               block i's MLP replaced by its dataset-mean output (removes token-specific part)
    prune:i[,j..]@k      blocks keep their top-k fraction of hidden neurons (fc1 rows / fc2 columns
                         physically sliced, so latency is real), ranking from the calibration subset
    zero:i,j,..          several blocks skipped together
Per run: bbox / segm AP, AP50, AP75, APs, APm, APl (pycocotools), per-image latency, and 50 seeded
80% image subsamples of AP for paired confidence intervals vs. base. Results append to a JSON file,
so an interrupted job resumes where it stopped.

The A40 training code is NOT needed: the config of ViTUWAX / FreqFusionFPN is mapped onto the reconstruction in
analysis/uwa_ext (ViTUWAv2 with a ConvNeXt-V2 Pico branch, FPNFreqFusion with feature_resample=True), and the
checkpoint is loaded through ckpt_compat.remap_checkpoint with strict key/shape checking. The base run is
the gate: its AP must match the checkpoint's recorded val segm mAP (about 46) before any other number is trusted.

    python analysis/mlp_study/mlp_ablate_eval.py --config configs/b6_best_3x_config.py \
        --checkpoint ckpt_b6/best_segm_mAP_epoch_29.pth --data-root datasets/USIS10K \
        --runs default --out analysis/results/mlp/ablate_val500.json
"""
import argparse
import contextlib
import copy
import io
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
for p in (osp.join(ROOT, 'analysis', 'pylib'), osp.join(ROOT, 'analysis'), osp.join(ROOT, 'detection')):
    if osp.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
ORIG_CWD = os.getcwd()
absp = lambda p: p if p is None or osp.isabs(p) else osp.abspath(osp.join(ORIG_CWD, p))  # noqa: E731

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--data-root', required=True, help='USIS10K folder (overrides the paths stored in the config)')
ap.add_argument('--split', default='val')
ap.add_argument('--subset', type=int, default=500, help='evaluation images (fixed random subset; 0 = all remaining)')
ap.add_argument('--calib', type=int, default=100, help='calibration images, disjoint from the evaluation subset')
ap.add_argument('--runs', nargs='+', default=['default'],
                help="run specs (see docstring) or 'default' = base, zero:0..11, prune 8-11, progressive late removal")
ap.add_argument('--with-mean', action='store_true', help='add mean:0..11 to the default runs')
ap.add_argument('--n-boot', type=int, default=50)
ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                help="config overrides after the ViTUWAX mapping, e.g. model.neck.ff_cfg.hr_residual=False "
                     "model.backbone.dam_cfg.proj_norm=ln2d fp16=None (values via ast.literal_eval)")
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()
for k in ('config', 'checkpoint', 'data_root', 'out'):
    setattr(args, k, absp(getattr(args, k)))
os.chdir(osp.join(ROOT, 'detection'))

from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import wrap_fp16_model  # noqa: E402
from mmcv.utils import import_modules_from_strings  # noqa: E402
from mmdet.core import encode_mask_results  # noqa: E402
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor  # noqa: E402
from mmdet.models import build_detector  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

import mmcv_custom  # noqa: E402,F401
import mmdet_custom  # noqa: E402,F401

sys.path.insert(0, osp.dirname(__file__))
from ckpt_compat import remap_checkpoint  # noqa: E402

# ------------------------------------------------------------------ runs
if args.runs == ['default']:
    runs = ['base'] + [f'zero:{i}' for i in range(12)]
    if args.with_mean:
        runs += [f'mean:{i}' for i in range(12)]
    runs += [f'prune:{i}@{k}' for i in (8, 9, 10, 11) for k in (0.5, 0.25)]
    runs += ['prune:8,9,10,11@0.5', 'prune:8,9,10,11@0.25', 'prune:9,10,11@0.25',
             'zero:11', 'zero:10,11', 'zero:9,10,11', 'zero:8,9,10,11']
    runs = list(dict.fromkeys(runs))
else:
    runs = args.runs

# ------------------------------------------------------------------ config / data / model
cfg = Config.fromfile(args.config)
bb, neck = cfg.model.backbone, cfg.model.neck
if bb.type == 'ViTUWAX':  # A40-only classes -> reconstruction in analysis/uwa_ext
    assert bb.get('hfcp') == 'off' and bb.get('dam') == 'pico', (bb.get('hfcp'), bb.get('dam'))
    for k in ('hfcp', 'dam', 'dam_lr_mult'):
        bb.pop(k)
    bb.type = 'ViTUWAv2'
    bb.dam_cfg = dict(model_name='convnextv2_pico.fcmae_ft_in1k', pretrained=False)
    assert neck.type == 'FreqFusionFPN', neck.type
    neck.type = 'FPNFreqFusion'  # parameter-exact match to the checkpoint needs feature_resample=True
    neck.ff_cfg = dict(compressed_channels=64, lowpass_kernel=5, highpass_kernel=3, feature_resample=True)
    neck.ff_fp32 = True
    cfg.custom_imports = dict(imports=['uwa_ext'], allow_failed_imports=False)
import ast  # noqa: E402
for kv in args.set:
    key, val = kv.split('=', 1)
    try:
        val = ast.literal_eval(val)
    except (ValueError, SyntaxError):
        pass  # keep as string
    node = cfg
    *path, last = key.split('.')
    for k in path:
        node = node[k]
    node[last] = val
    print(f'override {key} = {val!r}')
if cfg.get('custom_imports'):
    import_modules_from_strings(**cfg.custom_imports)
cfg.model.pretrained = None
bb.pretrained = None
if 'dam_cfg' in bb:
    bb.dam_cfg['pretrained'] = False
dcfg = copy.deepcopy(cfg.data.val)
dcfg.ann_file = osp.join(args.data_root, 'multi_class_annotations', f'multi_class_{args.split}_annotations.json')
dcfg.img_prefix = osp.join(args.data_root, args.split) + '/'
dcfg.test_mode = True
dcfg.pipeline = replace_ImageToTensor(dcfg.pipeline)
full = build_dataset(dcfg)


def subset(ds, idx):
    s = copy.copy(ds)
    s.data_infos = [ds.data_infos[i] for i in idx]
    s.img_ids = [ds.img_ids[i] for i in idx]
    if hasattr(ds, 'flag'):
        s.flag = ds.flag[idx]
    return s


perm = np.random.RandomState(args.seed).permutation(len(full))
calib_ds = subset(full, perm[:args.calib])
eval_idx = perm[args.calib:] if args.subset <= 0 else perm[args.calib:args.calib + args.subset]
eval_ds = subset(full, eval_idx)
loader = lambda ds: build_dataloader(ds, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)  # noqa: E731

model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
if cfg.get('fp16'):
    wrap_fp16_model(model)
ck = torch.load(args.checkpoint, map_location='cpu')
res = model.load_state_dict(remap_checkpoint(ck.get('state_dict', ck)), strict=False)
assert not res.missing_keys and not res.unexpected_keys, \
    f'checkpoint does not match the model: missing={res.missing_keys[:6]} unexpected={res.unexpected_keys[:6]}'
print('checkpoint loaded strictly (%d tensors); recorded best score: %s' % (len(ck.get('state_dict', ck)), ck.get('meta', {}).get('hook_msgs', {}).get('best_score')))
model.CLASSES = ck.get('meta', {}).get('CLASSES', full.CLASSES)
model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
blocks = model.module.backbone.blocks
print(f'eval {len(eval_ds)} / calib {len(calib_ds)} {args.split} images; {len(runs)} runs; fp16={bool(cfg.get("fp16"))}')


# ------------------------------------------------------------------ calibration: mean MLP output, neuron ranking
def calibrate():
    nb = len(blocks)
    mean_out = [0.] * nb
    imp = [0.] * nb
    cnt = [0] * nb
    hooks = []
    wn = [((b.gamma2.float()[:, None] if b.layer_scale else 1.) * b.mlp.fc2.weight.float()).norm(dim=0) for b in blocks]
    for i, b in enumerate(blocks):
        def h_out(m, inp, out, i=i):
            mean_out[i] = mean_out[i] + out[0].float().sum(0)
            cnt[i] += out.shape[1]
        def h_act(m, inp, out, i=i):  # noqa: E306
            imp[i] = imp[i] + (out[0].float().abs() * wn[i]).sum(0)
        hooks += [b.mlp.register_forward_hook(h_out), b.mlp.act.register_forward_hook(h_act)]
    with torch.no_grad():
        for data in loader(calib_ds):
            model(return_loss=False, rescale=True, **data)
    for h in hooks:
        h.remove()
    return [m / c for m, c in zip(mean_out, cnt)], [torch.argsort(v, descending=True) for v in imp]


class ConstMLP(torch.nn.Module):
    def __init__(self, vec):
        super().__init__()
        self.register_buffer('vec', vec)

    def forward(self, x):
        return self.vec.to(x.dtype).expand(*x.shape[:-1], -1)


class ZeroMLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.new_zeros(*x.shape[:-1], self.dim)


def sliced_mlp(mlp, keep_idx):
    m = copy.deepcopy(mlp)
    keep_idx = keep_idx.sort().values.to(mlp.fc1.weight.device)
    fc1, fc2 = mlp.fc1, mlp.fc2
    m.fc1 = torch.nn.Linear(fc1.in_features, len(keep_idx)).to(fc1.weight.device, fc1.weight.dtype)
    m.fc2 = torch.nn.Linear(len(keep_idx), fc2.out_features).to(fc2.weight.device, fc2.weight.dtype)
    with torch.no_grad():
        m.fc1.weight.copy_(fc1.weight[keep_idx]); m.fc1.bias.copy_(fc1.bias[keep_idx])
        m.fc2.weight.copy_(fc2.weight[:, keep_idx]); m.fc2.bias.copy_(fc2.bias)
    return m


def apply(spec, mean_out, rank):
    """Returns the list of (block, original mlp) to restore."""
    if spec == 'base':
        return []
    kind, rest = spec.split(':')
    keep = None
    if '@' in rest:
        rest, keep = rest.split('@')
        keep = float(keep)
    saved = []
    for i in map(int, rest.split(',')):
        b = blocks[i]
        saved.append((i, b.mlp))
        if kind == 'zero':
            b.mlp = ZeroMLP(b.mlp.fc2.out_features)
        elif kind == 'mean':
            b.mlp = ConstMLP(mean_out[i].to(b.mlp.fc2.weight.device))
        elif kind == 'prune':
            b.mlp = sliced_mlp(b.mlp, rank[i][:int(round(keep * b.mlp.fc1.out_features))])
        else:
            raise ValueError(spec)
    return saved


# ------------------------------------------------------------------ evaluation
def ap_from_eval(E):
    p = E.eval['precision']  # [T, R, K, A, M]
    f = lambda x: float(np.mean(x[x > -1])) if (x > -1).any() else float('nan')  # noqa: E731
    return dict(AP=f(p[:, :, :, 0, -1]), AP50=f(p[0, :, :, 0, -1]), AP75=f(p[5, :, :, 0, -1]),
                APs=f(p[:, :, :, 1, -1]), APm=f(p[:, :, :, 2, -1]), APl=f(p[:, :, :, 3, -1]))


boot_sets = [sorted(np.random.RandomState(1000 + s).choice(eval_ds.img_ids, int(0.8 * len(eval_ds)), replace=False).tolist())
             for s in range(args.n_boot)]


def evaluate(results, tag):
    files, tmp = eval_ds.format_results(results, jsonfile_prefix=osp.join(osp.dirname(args.out), f'_tmp_{tag}'))
    out = {}
    for kind in ('bbox', 'segm'):
        E = COCOeval(eval_ds.coco, eval_ds.coco.loadRes(files[kind]), kind)
        E.params.imgIds = eval_ds.img_ids
        with contextlib.redirect_stdout(io.StringIO()):
            E.evaluate(), E.accumulate()
        out[kind] = ap_from_eval(E)
        # paired resamples: keep p.imgIds, blank the per-image results of excluded images (a changed imgIds
        # after evaluate() would make accumulate() index the wrong images)
        K, A, I = len(E.params.catIds), len(E.params.areaRng), len(E.params.imgIds)
        full_eval = np.empty(K * A * I, dtype=object)
        full_eval[:] = E.evalImgs
        full_eval = full_eval.reshape(K, A, I)
        pos = {iid: n for n, iid in enumerate(E.params.imgIds)}
        boots = []
        with contextlib.redirect_stdout(io.StringIO()):
            for s in boot_sets:
                keep = np.zeros(I, bool)
                keep[[pos[i] for i in s]] = True
                masked = full_eval.copy()
                masked[:, :, ~keep] = None
                E.evalImgs = masked.reshape(-1).tolist()
                E.accumulate()
                boots.append(ap_from_eval(E)['AP'])
        out[kind]['boot_AP'] = boots
    for f in set(files.values()):  # 'proposal' points at the bbox file
        if osp.isfile(f):
            os.remove(f)
    return out


def run_once(ds):
    results, times = [], []
    with torch.no_grad():
        for n, data in enumerate(loader(ds)):
            torch.cuda.synchronize()
            t = time.time()
            r = model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            if n >= 10:
                times.append(time.time() - t)
            if isinstance(r[0], tuple):
                r = [(b, encode_mask_results(m)) for b, m in r]
            results.extend(r)
    return results, float(np.median(times))


done = json.load(open(args.out)) if osp.isfile(args.out) else {}
os.makedirs(osp.dirname(args.out), exist_ok=True)
mean_out, rank = calibrate()
for spec in runs:
    if spec in done:
        print(f'skip {spec} (done)')
        continue
    saved = apply(spec, mean_out, rank)
    t0 = time.time()
    results, lat = run_once(eval_ds)
    m = evaluate(results, spec.replace(':', '_').replace(',', '-').replace('@', '_k'))
    m['mlp_params_M'] = sum(p.numel() for b in blocks for p in b.mlp.parameters()) / 1e6  # 12 ViT MLPs as modified
    for i, mlp in saved:
        blocks[i].mlp = mlp
    m['latency_s'] = lat
    done[spec] = m
    json.dump(done, open(args.out, 'w'), indent=1)
    b = done.get('base')
    d = (lambda k: f"{100 * (m[k]['AP'] - b[k]['AP']):+.2f}") if b else (lambda k: '')
    print(f"{spec:24s} box {100 * m['bbox']['AP']:.2f} ({d('bbox')})  mask {100 * m['segm']['AP']:.2f} ({d('segm')})  "
          f"mask APs {100 * m['segm']['APs']:.2f}  {lat * 1000:.0f} ms/img  [{time.time() - t0:.0f}s]", flush=True)

# ------------------------------------------------------------------ summary with paired subsample intervals
base = done['base']
lines = ['| run | box AP | Δ box (90% CI) | mask AP | Δ mask (90% CI) | mask AP50 | mask AP75 | mask APs | ms/img |',
         '|---|---|---|---|---|---|---|---|---|']
for spec, m in done.items():
    cells = []
    for k in ('bbox', 'segm'):
        dlt = 100 * (np.array(m[k]['boot_AP']) - np.array(base[k]['boot_AP']))
        cells.append((100 * m[k]['AP'], f"{100 * (m[k]['AP'] - base[k]['AP']):+.2f} [{np.percentile(dlt, 5):+.2f}, {np.percentile(dlt, 95):+.2f}]"))
    s = m['segm']
    lines.append(f"| {spec} | {cells[0][0]:.2f} | {cells[0][1]} | {cells[1][0]:.2f} | {cells[1][1]} | "
                 f"{100 * s['AP50']:.2f} | {100 * s['AP75']:.2f} | {100 * s['APs']:.2f} | {1000 * m['latency_s']:.0f} |")
open(osp.splitext(args.out)[0] + '.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
