"""Run LFA-Net on a folder of fundus images and write probability maps.

    python predict.py --weights Model_weights/LF-Net_DRIVE.h5 \
                      --images path/to/images --out predictions/

Writes one 8-bit PNG per input, which is the format Prediction_code/Evaluation.m
expects (it reads <n>_Pred.png alongside <n>_GT.png).
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from skimage.transform import resize

SIZE = 512


def preprocess(path, size=SIZE):
    """As in the training notebook: resize to 512 with preserve_range, then /255.

    skimage.transform.resize anti-aliases when downsampling. PIL's BILINEAR does
    not, and DRIVE is 584x565, so the two are not interchangeable here.
    """
    img = np.asarray(Image.open(path).convert("RGB"), np.float32)
    return resize(img, (size, size, 3), mode="constant", preserve_range=True).astype(np.float32) / 255.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--native-size", action="store_true",
                    help="resize each map back to the input image's own size")
    a = ap.parse_args()

    from lfanet import LiteFusion_UNet
    model = LiteFusion_UNet()
    model.load_weights(a.weights)
    print("loaded %s (%s parameters)" % (a.weights, format(model.count_params(), ",")))

    os.makedirs(a.out, exist_ok=True)
    names = sorted(os.listdir(a.images))
    for n in names:
        src = os.path.join(a.images, n)
        x = preprocess(src)
        p = model.predict(x[None], verbose=0)[0, ..., 0]
        if a.native_size:
            h, w = Image.open(src).size[::-1]
            p = resize(p, (h, w), mode="constant", preserve_range=True)
        stem = os.path.splitext(n)[0]
        Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(a.out, "%s_Pred.png" % stem))
    print("wrote %d probability maps to %s/" % (len(names), a.out))


if __name__ == "__main__":
    main()
