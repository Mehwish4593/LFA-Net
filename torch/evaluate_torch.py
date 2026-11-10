"""Evaluate a PyTorch LFA-Net checkpoint under the paper's protocol.

Protocol is Prediction_code/Evaluation.m: for each image, the threshold that
maximises F1 against that image's own mask, then bwareaopen(out, 50), scored
over the full frame with no field-of-view mask, averaged per image.

Preprocessing follows the notebook's get_data(): skimage resize to 512 with
preserve_range, then /255.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_test_set, bwareaopen, f1_optimal_threshold, metrics, SIZE
from lfanet_torch import LFANet
from sklearn.metrics import average_precision_score


def predict(model, x, dev, tta=False):
    xt = torch.tensor(x).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        if not tta:
            return model(xt)[0, 0].cpu().numpy()
        acc = np.zeros((SIZE, SIZE), np.float32)
        for flip in (False, True):
            xf = x[:, ::-1] if flip else x
            for k in range(4):
                r = np.ascontiguousarray(np.rot90(xf, k, (0, 1)))
                p = model(torch.tensor(r).permute(2, 0, 1)[None].to(dev))[0, 0].cpu().numpy()
                p = np.rot90(p, -k, (0, 1))
                acc += p[:, ::-1] if flip else p
        return acc / 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="results/retrained.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = LFANet().to(dev)
    model.load_state_dict(torch.load(a.weights, map_location=dev))
    model.eval()

    X, Y, names = load_test_set(a.images, a.gt)
    print("%s | %d images | vessel fraction %.4f%s\n"
          % (os.path.basename(a.weights), len(X), Y.mean(), "  (8-way TTA)" if a.tta else ""))

    per, aps, thrs = [], [], []
    for i, x in enumerate(X):
        P = predict(model, x, dev, a.tta)
        t = f1_optimal_threshold(P, Y[i])
        thrs.append(t)
        per.append(metrics(bwareaopen(P >= t, 50).astype(np.float32), Y[i]))
        aps.append(average_precision_score(Y[i].ravel(), P.ravel()))

    m = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    m["average_precision"] = float(np.mean(aps))
    for k in ("dice", "jaccard", "sensitivity", "specificity", "accuracy", "average_precision"):
        print("  %-18s %.4f" % (k, m[k]))
    print("\n  F1-optimal thresholds: min %.4f  median %.4f  max %.4f"
          % (min(thrs), float(np.median(thrs)), max(thrs)))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"weights": a.weights, "tta": a.tta, "mean": m,
               "thresholds": thrs, "per_image": per, "images": names},
              open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
