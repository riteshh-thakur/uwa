"""Find which reconstruction detail makes the base model reproduce the trained one.

Runs mlp_ablate_eval.py --runs base once per variant on the SAME fixed val subset and prints one table.
Variants change only details that leave every tensor name and shape unchanged (so the strict checkpoint
load cannot reveal them). The checkpoint recorded 45.98 val mask AP; the right variant should move
towards that. Screening uses a small subset (noise about +-2 AP), so only large jumps are meaningful.

    python analysis/mlp_study/variant_sweep.py --checkpoint ckpt.pth --data-root datasets/USIS10K --python /content/uwa/bin/python
"""
import argparse
import json
import os
import os.path as osp
import re
import subprocess
import sys

HERE = osp.dirname(osp.abspath(__file__))
ROOT = osp.abspath(osp.join(HERE, '..', '..'))

VARIANTS = {
    'baseline': [],
    'fp32_eval': ['fp16=None'],
    'neck_not_fp32': ['model.neck.ff_fp32=False'],
    'ff_hr_residual_off': ['model.neck.ff_cfg.hr_residual=False'],
    'ff_semi_conv_off': ['model.neck.ff_cfg.semi_conv=False'],
    'ff_comp_feat_upsample_off': ['model.neck.ff_cfg.comp_feat_upsample=False'],
    'ff_upsample_bilinear': ['model.neck.ff_cfg.upsample_mode=bilinear'],
    'proj_norm_gn1': ['model.backbone.dam_cfg.proj_norm=gn1'],
    'proj_norm_ln2d': ['model.backbone.dam_cfg.proj_norm=ln2d'],
}

ap = argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--data-root', required=True)
ap.add_argument('--config', default=osp.join(ROOT, 'configs', 'b6_best_3x_config.py'))
ap.add_argument('--subset', type=int, default=150)
ap.add_argument('--python', default=sys.executable)
ap.add_argument('--out-dir', default='/content/sweep')
ap.add_argument('--only', nargs='*', help='variant names to run (default: all)')
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

rows = []
for name, sets in VARIANTS.items():
    if args.only and name not in args.only:
        continue
    out = osp.join(args.out_dir, f'{name}.json')
    if not osp.isfile(out):
        cmd = [args.python, osp.join(HERE, 'mlp_ablate_eval.py'), '--config', args.config, '--checkpoint', args.checkpoint,
               '--data-root', args.data_root, '--runs', 'base', '--subset', str(args.subset), '--n-boot', '2', '--out', out]
        if sets:
            cmd += ['--set'] + sets
        print(f'>>> {name}: {" ".join(sets) or "(no override)"}', flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not osp.isfile(out):
            print('FAILED:\n' + '\n'.join(r.stdout.splitlines()[-3:] + r.stderr.splitlines()[-8:]), flush=True)
            rows.append((name, None))
            continue
    m = json.load(open(out))['base']
    rows.append((name, m))
    print(f"    box {100 * m['bbox']['AP']:.2f}  mask {100 * m['segm']['AP']:.2f}  mask APs {100 * m['segm']['APs']:.2f}", flush=True)

print(f'\nsubset={args.subset} val images; recorded full-val mask AP of the checkpoint: 45.98')
print(f"{'variant':28s} {'box AP':>7s} {'mask AP':>8s} {'mask AP50':>10s} {'mask AP75':>10s} {'ms/img':>7s}")
for name, m in rows:
    if m is None:
        print(f'{name:28s}  failed')
        continue
    print(f"{name:28s} {100 * m['bbox']['AP']:7.2f} {100 * m['segm']['AP']:8.2f} {100 * m['segm']['AP50']:10.2f} "
          f"{100 * m['segm']['AP75']:10.2f} {1000 * m['latency_s']:7.0f}")
