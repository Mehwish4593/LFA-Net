"""Train LFA-Net on the GPU, same recipe as the notebook.

The TensorFlow build on this machine has CUDA compiled out, so train.py runs on
CPU at ~81 s/epoch. This is the same model (verified against the Keras version
at 1.5e-05 by verify_port.py) in PyTorch, which does have CUDA.

Recipe, unchanged from LFA-Net.ipynb:
    get_data   skimage resize to 512, /255, masks from GT_BV
    split      train_test_split(test_size=0.1, random_state=42)
    loss       DiceLoss, squared denominator, smooth 1e-6, over the whole batch
    optimiser  Adam(1e-3)
    batch 16, up to 300 epochs, keep the best validation loss
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfanet_torch import LFANet

SIZE = 512


def get_data(path, size=SIZE):
    """The notebook's get_data(), in numpy."""
    from skimage.transform import resize
    img_dir, gt_dir = os.path.join(path, "images"), os.path.join(path, "GT_BV")
    if not os.path.isdir(img_dir):                 # STARE/CHASE use Images/ + GT/
        img_dir, gt_dir = os.path.join(path, "Images"), os.path.join(path, "GT")
    ids, gids = sorted(os.listdir(img_dir)), sorted(os.listdir(gt_dir))
    X = np.zeros((len(ids), size, size, 3), np.float32)
    Y = np.zeros((len(gids), size, size, 1), np.float32)
    print("Getting and resizing images ...", flush=True)
    for n, (a, b) in enumerate(zip(ids, gids)):
        im = np.asarray(Image.open(os.path.join(img_dir, a)).convert("RGB"), np.float32)
        X[n] = resize(im, (size, size, 3), mode="constant", preserve_range=True) / 255.
        m = np.asarray(Image.open(os.path.join(gt_dir, b)).convert("L"), np.float32)
        Y[n] = resize(m[..., None], (size, size, 1), mode="constant", preserve_range=True) / 255.
    print("Done!", flush=True)
    return X, Y


def dice_loss(pred, target, smooth=1e-6):
    """1 - squared-denominator Dice over the whole batch, as the notebook."""
    inter = (pred * target).sum()
    den = (pred ** 2).sum() + (target ** 2).sum()
    return 1 - (2. * inter + smooth) / (den + smooth)


def dice_coef(pred, target, smooth=1e-6):
    return ((2. * (pred * target).sum() + smooth) /
            (pred.sum() + target.sum() + smooth)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/LF-Net_DRIVE_torch.pt")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)

    X, Y = get_data(a.data)
    from sklearn.model_selection import train_test_split
    Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.1, random_state=a.seed)
    to = lambda z: torch.tensor(z).permute(0, 3, 1, 2)
    Xtr, Xva, Ytr, Yva = to(Xtr), to(Xva), to(Ytr), to(Yva)
    print("train %d | val %d | vessel fraction %.4f"
          % (len(Xtr), len(Xva), (Y > .5).mean()), flush=True)

    model = LFANet().to(dev)
    print("parameters:", format(sum(p.numel() for p in model.parameters()), ","), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best, best_ep, hist = float("inf"), -1, []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = len(Xtr)
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(n)
        tl = td = 0.0; nb = 0
        for i in range(0, n - a.batch + 1, a.batch):
            idx = perm[i:i + a.batch]
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            p = model(xb)
            loss = dice_loss(p, yb)
            loss.backward(); opt.step()
            tl += loss.item(); td += dice_coef(p.detach(), yb); nb += 1

        model.eval(); vl = vd = 0.0; vb = 0
        with torch.no_grad():
            for i in range(0, len(Xva), a.batch):
                xb, yb = Xva[i:i + a.batch].to(dev), Yva[i:i + a.batch].to(dev)
                p = model(xb)
                vl += dice_loss(p, yb).item(); vd += dice_coef(p, yb); vb += 1
        tl, td, vl, vd = tl / nb, td / nb, vl / vb, vd / vb
        hist.append(dict(epoch=ep, loss=tl, dice=td, val_loss=vl, val_dice=vd))

        flag = ""
        if vl < best:
            best, best_ep = vl, ep
            torch.save(model.state_dict(), a.out)
            flag = "  <- best"
        print("ep %3d/%d  loss %.4f dice %.4f | val_loss %.4f val_dice %.4f  (%.0fs)%s"
              % (ep, a.epochs, tl, td, vl, vd, time.time() - t0, flag), flush=True)

    json.dump(hist, open(os.path.splitext(a.out)[0] + "_history.json", "w"), indent=1)
    print("\nbest val_loss %.4f at epoch %d -> %s" % (best, best_ep, a.out), flush=True)


if __name__ == "__main__":
    main()
