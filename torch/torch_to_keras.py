"""Convert a retrained PyTorch checkpoint back into the Keras .h5 layout.

The repository's notebook, predict.py and Prediction_code/ are all Keras, so
the released weights should be too. This inverts verify_port.load_from_keras()
and checks the round trip numerically before writing.
"""
import os, sys
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import numpy as np
import torch
from verify_port import CONV, DECONV, BN, LN, get


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch-weights", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from lfanet import LiteFusion_UNet
    from lfanet_torch import LFANet

    t = LFANet(); t.load_state_dict(torch.load(a.torch_weights, map_location="cpu")); t.eval()
    k = LiteFusion_UNet()

    for l in k.layers:
        if not l.weights:
            continue
        n = l.name
        if n in CONV:
            m = get(t, CONV[n])
            w = [np.transpose(m.weight.detach().numpy(), (2, 3, 1, 0))]
            if m.bias is not None:
                w.append(m.bias.detach().numpy())
            l.set_weights(w)
        elif n in DECONV:
            m = get(t, DECONV[n]).conv
            l.set_weights([np.transpose(m.weight.detach().numpy(), (2, 3, 1, 0)),
                           m.bias.detach().numpy()])
        elif n in BN:
            m = get(t, BN[n])
            l.set_weights([m.weight.detach().numpy(), m.bias.detach().numpy(),
                           m.running_mean.numpy(), m.running_var.numpy()])
        elif n in LN:
            m = get(t, LN[n])
            l.set_weights([m.g.detach().numpy(), m.b.detach().numpy()])
        elif type(l).__name__ == "VisionMambaBlock":
            m = t.att.mamba
            l.set_weights([
                m.norm1.g.detach().numpy(), m.norm1.b.detach().numpy(),
                np.transpose(m.dw.weight.detach().numpy(), (2, 3, 0, 1)),
                m.dw.bias.detach().numpy(),
                m.norm2.g.detach().numpy(), m.norm2.b.detach().numpy(),
                m.fc1.weight.detach().numpy()[:, :, 0, 0].T, m.fc1.bias.detach().numpy(),
                m.fc2.weight.detach().numpy()[:, :, 0, 0].T, m.fc2.bias.detach().numpy(),
            ])
        else:
            raise SystemExit("unmapped layer: %s" % n)

    rng = np.random.default_rng(0)
    x = rng.random((1, 512, 512, 3), dtype=np.float32)
    ko = k.predict(x, verbose=0)[0, ..., 0]
    with torch.no_grad():
        to = t(torch.tensor(x).permute(0, 3, 1, 2))[0, 0].numpy()
    d = float(np.abs(ko - to).max())
    print("max |torch - keras| = %.3e  %s" % (d, "OK" if d < 1e-4 else "FAIL"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    k.save_weights(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
