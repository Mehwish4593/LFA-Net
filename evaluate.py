"""Evaluate LFA-Net on a test set, in Python, matching Prediction_code/Evaluation.m.

The published protocol is Prediction_code/Evaluation.m from the LFA-Net
repository, reimplemented here:

  * per-image threshold chosen to maximise F1 against that image's own mask
    (MATLAB perfcurve -> max F1 -> imbinarize)
  * bwareaopen(out, 50): drop 8-connected components smaller than 50 px
  * metrics over the full frame, no field-of-view mask
  * per-image metrics, then averaged over the test set

Preprocessing follows get_data() in LFA-Net.ipynb: skimage.transform.resize to
512x512 with preserve_range, then /255. skimage anti-aliases when downsampling
and PIL's BILINEAR does not, so the two are not interchangeable here.

Average precision is also reported. It is threshold-free, so it does not depend
on the per-image threshold choice above, and is therefore the more stable number
to compare across checkpoints.

This is a convenience port. Prediction_code/Evaluation.m remains the reference.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import resize as sk_resize
from sklearn.metrics import precision_recall_curve, average_precision_score

SIZE = 512


def load_test_set(images_dir, gt_dir):
    names = sorted(os.listdir(images_dir))
    masks = sorted(os.listdir(gt_dir))
    if len(names) != len(masks):
        raise SystemExit("%d images but %d masks" % (len(names), len(masks)))
    X, Y = [], []
    for a, b in zip(names, masks):
        img = np.asarray(Image.open(os.path.join(images_dir, a)).convert("RGB"), np.float32)
        gt = np.asarray(Image.open(os.path.join(gt_dir, b)).convert("L"), np.float32)
        X.append(sk_resize(img, (SIZE, SIZE, 3), mode="constant",
                           preserve_range=True).astype(np.float32) / 255.)
        g = sk_resize(gt[..., None], (SIZE, SIZE, 1), mode="constant",
                      preserve_range=True)[..., 0] / 255.
        Y.append((g > 0.5).astype(np.float32))
    return np.array(X), np.array(Y), names


def bwareaopen(mask, min_size=50):
    """MATLAB bwareaopen: remove 8-connected components below min_size."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:][sizes >= min_size] = True
    return keep[lab]


def f1_optimal_threshold(prob, gt):
    p, r, t = precision_recall_curve(gt.ravel(), prob.ravel())
    f1 = 2 * (p * r) / (p + r + 1e-12)
    return float(t[min(int(np.nanargmax(f1)), len(t) - 1)])


def metrics(pred, gt):
    tp = float((pred * gt).sum()); tn = float(((1 - pred) * (1 - gt)).sum())
    fp = float((pred * (1 - gt)).sum()); fn = float(((1 - pred) * gt).sum())
    e = 1e-9
    return dict(dice=2 * tp / (2 * tp + fp + fn + e),
                jaccard=tp / (tp + fp + fn + e),
                sensitivity=tp / (tp + fn + e),
                specificity=tn / (tn + fp + e),
                accuracy=(tp + tn) / (tp + tn + fp + fn + e))


def predict(model, x, tta=False):
    if not tta:
        return model.predict(x[None], verbose=0)[0, ..., 0]
    acc = np.zeros((SIZE, SIZE), np.float32)
    for flip in (False, True):
        xf = x[:, ::-1] if flip else x
        for k in range(4):
            p = model.predict(np.ascontiguousarray(np.rot90(xf, k, (0, 1)))[None],
                              verbose=0)[0, ..., 0]
            p = np.rot90(p, -k, (0, 1))
            acc += p[:, ::-1] if flip else p
    return acc / 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Model_weights/LF-Net_DRIVE.h5")
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tta", action="store_true", help="8-way test-time augmentation")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    from lfanet import LiteFusion_UNet
    model = LiteFusion_UNet()
    model.load_weights(a.weights)

    X, Y, names = load_test_set(a.images, a.gt)
    print("%s | %d images | vessel fraction %.4f%s\n"
          % (os.path.basename(a.weights), len(X), Y.mean(),
             "  (8-way TTA)" if a.tta else ""))

    per, aps = [], []
    for i, x in enumerate(X):
        P = predict(model, x, a.tta)
        binm = bwareaopen(P >= f1_optimal_threshold(P, Y[i]), 50).astype(np.float32)
        per.append(metrics(binm, Y[i]))
        aps.append(average_precision_score(Y[i].ravel(), P.ravel()))

    m = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    m["average_precision"] = float(np.mean(aps))
    for k in ("dice", "jaccard", "sensitivity", "specificity", "accuracy", "average_precision"):
        print("  %-18s %.4f" % (k, m[k]))

    json.dump({"weights": a.weights, "n_images": len(X), "tta": a.tta,
               "mean": m, "per_image": per}, open(a.out, "w"), indent=1)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
