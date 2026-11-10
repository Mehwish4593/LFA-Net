"""Copy the Keras weights into the PyTorch port and check the outputs agree.

If this does not print OK, the port is not usable and must not be trained.
"""
import os, sys
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch

# Keras layer name -> torch module path.  Order of creation, read off the
# Keras layer listing; see README for the derivation.
CONV = {
    "conv2d": "cb1.a", "conv2d_1": "cb1.b", "conv2d_2": "cb1.c",
    "conv2d_3": "cb2.a", "conv2d_4": "cb2.b", "conv2d_5": "cb2.c",
    "conv2d_6": "cb3.a", "conv2d_7": "cb3.b", "conv2d_8": "cb3.c",
    "conv2d_9": "att.proj_in", "conv2d_10": "att.conv",
    "conv2d_11": "att.ctx.conv1", "conv2d_12": "att.ctx.conv2",
    "conv2d_13": "att.ctx.gate", "conv2d_14": "att.ctx.focal",
    "conv2d_15": "att.proj_out",
    "conv2d_16": "ra1.conv", "conv2d_17": "ra2.conv", "conv2d_19": "ra3.conv",
    "conv2d_18": "cv1", "conv2d_20": "cv2", "conv2d_21": "cv3",
    "final_output1": "out",
}
DECONV = {"conv2d_transpose": "up1", "conv2d_transpose_1": "up2",
          "conv2d_transpose_2": "up3"}
BN = {"batch_normalization": "bn1", "batch_normalization_1": "bn2",
      "batch_normalization_2": "bn3", "batch_normalization_3": "ra1.bn",
      "batch_normalization_4": "ra2.bn", "batch_normalization_5": "ra3.bn"}
LN = {"layer_normalization": "att.norm"}


def get(model, path):
    m = model
    for p in path.split("."):
        m = getattr(m, p)
    return m


def load_from_keras(torch_model, keras_model):
    t = torch_model
    for l in keras_model.layers:
        if not l.weights:
            continue
        w = [x.numpy() for x in l.weights]
        n = l.name
        if n in CONV:
            m = get(t, CONV[n])
            m.weight.data = torch.tensor(np.transpose(w[0], (3, 2, 0, 1)))
            if m.bias is not None:
                m.bias.data = torch.tensor(w[1])
        elif n in DECONV:                      # Keras (kh,kw,out,in) -> torch (in,out,kh,kw)
            m = get(t, DECONV[n]).conv
            m.weight.data = torch.tensor(np.transpose(w[0], (3, 2, 0, 1)))
            m.bias.data = torch.tensor(w[1])
        elif n in BN:
            m = get(t, BN[n])
            m.weight.data = torch.tensor(w[0]); m.bias.data = torch.tensor(w[1])
            m.running_mean.data = torch.tensor(w[2]); m.running_var.data = torch.tensor(w[3])
        elif n in LN:
            m = get(t, LN[n])
            m.g.data = torch.tensor(w[0]); m.b.data = torch.tensor(w[1])
        elif type(l).__name__ == "VisionMambaBlock":
            m = t.att.mamba
            m.norm1.g.data = torch.tensor(w[0]); m.norm1.b.data = torch.tensor(w[1])
            m.dw.weight.data = torch.tensor(np.transpose(w[2], (2, 3, 0, 1)))
            m.dw.bias.data = torch.tensor(w[3])
            m.norm2.g.data = torch.tensor(w[4]); m.norm2.b.data = torch.tensor(w[5])
            m.fc1.weight.data = torch.tensor(w[6].T[:, :, None, None].copy())
            m.fc1.bias.data = torch.tensor(w[7])
            m.fc2.weight.data = torch.tensor(w[8].T[:, :, None, None].copy())
            m.fc2.bias.data = torch.tensor(w[9])
        else:
            raise SystemExit("unmapped layer: %s (%s)" % (n, type(l).__name__))
    return t


def main():
    from lfanet import LiteFusion_UNet
    from lfanet_torch import LFANet

    weights = sys.argv[1] if len(sys.argv) > 1 else "models/Model_weights/LF-Net_DRIVE.h5"
    k = LiteFusion_UNet(); k.load_weights(weights)
    t = LFANet(); load_from_keras(t, k); t.eval()

    rng = np.random.default_rng(0)
    x = rng.random((1, 512, 512, 3), dtype=np.float32)
    ko = k.predict(x, verbose=0)[0, ..., 0]
    with torch.no_grad():
        to = t(torch.tensor(x).permute(0, 3, 1, 2))[0, 0].numpy()

    d = float(np.abs(ko - to).max())
    print("keras mean %.6f | torch mean %.6f" % (ko.mean(), to.mean()))
    print("max |keras - torch| = %.3e   %s" % (d, "OK" if d < 1e-4 else "FAIL"))
    if d >= 1e-4:
        sys.exit(1)
    torch.save(t.state_dict(), "models/lfanet_torch_from_keras.pt")
    print("saved models/lfanet_torch_from_keras.pt")


if __name__ == "__main__":
    main()
