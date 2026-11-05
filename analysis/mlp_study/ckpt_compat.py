"""Key/shape mapping from the A40-trained checkpoint (official ConvNeXt-V2 module names, neck.ff.*) to the
reconstruction in analysis/uwa_ext (timm ConvNeXt-V2 features_only, neck.freq_fusions.*)."""
import re


def official_convnext_to_timm(sd, prefix='dam.cnn.'):
    """official facebookresearch ConvNeXt-V2 names/layouts (downsample_layers, stages.i.j.dwconv/pwconv1/2,
    grn.gamma/beta) -> timm features_only layout. Other keys pass through unchanged."""
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix) or ('downsample_layers' not in k and not re.match(prefix + r'stages\.\d+\.\d+\.', k)):
            out[k] = v
            continue
        r = k[len(prefix):]
        m = re.match(r'downsample_layers\.0\.(\d)\.(weight|bias)$', r)
        if m:                                              # stem: conv4x4 then LayerNorm
            nk = f'stem_{m.group(1)}.{m.group(2)}'
        else:
            m = re.match(r'downsample_layers\.([123])\.(\d)\.(weight|bias)$', r)
            if m:                                          # per-stage downsample: LayerNorm then conv2x2
                nk = f'stages_{m.group(1)}.downsample.{m.group(2)}.{m.group(3)}'
            else:
                i, j, rest = re.match(r'stages\.(\d)\.(\d+)\.(.+)$', r).groups()
                rest = (rest.replace('dwconv', 'conv_dw').replace('pwconv1', 'mlp.fc1').replace('pwconv2', 'mlp.fc2')
                        .replace('grn.gamma', 'mlp.grn.weight').replace('grn.beta', 'mlp.grn.bias'))
                nk = f'stages_{i}.blocks.{j}.{rest}'
                if '.mlp.fc' in nk and nk.endswith('weight'):
                    v = v[:, :, None, None]                # Linear (out, in) -> 1x1 conv (out, in, 1, 1)
                elif 'mlp.grn' in nk:
                    v = v.reshape(-1)                      # (1,1,1,C) -> (C,)
        out[prefix + nk] = v
    return out


def remap_checkpoint(sd):
    """Full detector state dict -> keys of the uwa_ext reconstruction (backbone DAM CNN, neck FreqFusion)."""
    sd = official_convnext_to_timm(sd, prefix='backbone.dam.cnn.')
    return {(k.replace('neck.ff.', 'neck.freq_fusions.', 1) if k.startswith('neck.ff.') else k): v
            for k, v in sd.items()}
