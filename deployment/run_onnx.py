"""Run LFA-Net's ONNX model on one image or a folder.

    python run_onnx.py --onnx lfanet_drive.onnx --images path/to/images --out preds/
    python run_onnx.py --onnx lfanet_drive.onnx --images path/to/images --gt path/to/GT_BV

Needs only: pip install onnxruntime numpy pillow scikit-image
Add --gt to also score. Scoring uses the paper's protocol: a per-image
threshold chosen to maximise F1 against that image's own mask, then
components under 50 px dropped.
"""
import argparse, os, time
import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize
import onnxruntime as ort

SIZE = 512


def preprocess(path):
    """skimage resize, not PIL: skimage anti-aliases when downsampling."""
    img = np.asarray(Image.open(path).convert("RGB"), np.float32)
    return (sk_resize(img, (SIZE, SIZE, 3), mode="constant",
                      preserve_range=True) / 255.).astype(np.float32)


def load_gt(path):
    g = np.asarray(Image.open(path).convert("L"), np.float32)
    g = sk_resize(g[..., None], (SIZE, SIZE, 1), mode="constant",
                  preserve_range=True)[..., 0] / 255.
    return (g > 0.5).astype(np.float32)


def score(prob, gt):
    from scipy import ndimage
    from sklearn.metrics import precision_recall_curve, average_precision_score
    pr, rc, th = precision_recall_curve(gt.ravel(), prob.ravel())
    f1 = 2 * (pr * rc) / (pr + rc + 1e-12)
    t = float(th[min(int(np.nanargmax(f1)), len(th) - 1)])
    m = prob >= t
    lab, n = ndimage.label(m, structure=np.ones((3, 3), int))
    if n:
        sz = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:][sz >= 50] = True
        m = keep[lab]
    m = m.astype(np.float32)
    tp = (m * gt).sum()
    return (2 * tp / (m.sum() + gt.sum() + 1e-9),
            average_precision_score(gt.ravel(), prob.ravel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", required=True, help="an image file or a folder")
    ap.add_argument("--gt", default=None, help="matching masks, to also score")
    ap.add_argument("--out", default=None, help="folder for the PNG predictions")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    print(f"{a.onnx}: input {inp} {sess.get_inputs()[0].shape} -> "
          f"{sess.get_outputs()[0].name} {sess.get_outputs()[0].shape}")

    if os.path.isdir(a.images):
        names = sorted(os.listdir(a.images))
        paths = [os.path.join(a.images, n) for n in names]
    else:
        paths, names = [a.images], [os.path.basename(a.images)]
    gts = sorted(os.listdir(a.gt)) if a.gt else None

    if a.out:
        os.makedirs(a.out, exist_ok=True)

    dices, aps, times = [], [], []
    for i, p in enumerate(paths):
        x = preprocess(p)[None]
        t0 = time.perf_counter()
        prob = sess.run(None, {inp: x})[0][0, ..., 0]
        times.append((time.perf_counter() - t0) * 1000)

        if a.out:
            stem = os.path.splitext(names[i])[0]
            Image.fromarray((prob * 255).astype(np.uint8)).save(
                os.path.join(a.out, f"{stem}_Pred.png"))
        if gts:
            d, ap_ = score(prob, load_gt(os.path.join(a.gt, gts[i])))
            dices.append(d); aps.append(ap_)
            print(f"  {names[i]:<14} Dice {d:.4f}  AP {ap_:.4f}")
        else:
            print(f"  {names[i]:<14} prob min {prob.min():.3f} "
                  f"max {prob.max():.3f} mean {prob.mean():.3f}")

    print(f"\n{len(paths)} image(s), {np.median(times):.1f} ms median "
          f"({a.threads} threads)")
    if dices:
        print(f"mean Dice {np.mean(dices):.4f}   mean AP {np.mean(aps):.4f}")
    if a.out:
        print(f"predictions written to {a.out}")


if __name__ == "__main__":
    main()
