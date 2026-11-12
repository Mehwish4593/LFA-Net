r"""
LFA-Net ONNX benchmark. Needs only: pip install onnxruntime numpy pillow psutil
Usage:
  python bench_lfanet_onnx.py                       # latency + memory only
  python bench_lfanet_onnx.py C:\path\to\DRIVE\test  # + Dice (FP32 and INT8) ; folder has images\ and GT_BV\
"""
import sys, os, time, json, statistics, psutil, platform
import numpy as np
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat

FP32 = "lfanet_drive.onnx"; INT8 = "lfanet_drive_int8.onnx"; H = W = 512; THREADS = 4
res = {"cpu": platform.processor(), "threads": THREADS, "input": f"{H}x{W}x3", "onnx_mb_fp32": os.path.getsize(FP32)/1e6}

def load_pair(folder, name_img, name_gt):
    img = Image.open(os.path.join(folder, "images", name_img)).convert("RGB").resize((W, H), Image.BILINEAR)
    gt  = Image.open(os.path.join(folder, "GT_BV", name_gt)).convert("L").resize((W, H), Image.BILINEAR)
    return (np.asarray(img, np.float32)/255.0)[None], (np.asarray(gt, np.float32)/255.0 > 0.5)

def dataset(folder):
    imgs = sorted(os.listdir(os.path.join(folder, "images"))); gts = sorted(os.listdir(os.path.join(folder, "GT_BV")))
    for a, b in zip(imgs, gts):
        try: yield load_pair(folder, a, b)
        except Exception as e: print("skip", a, e)

def dice(pred_bin, gt):
    tp = (pred_bin & gt).sum(); return 2*tp / (pred_bin.sum() + gt.sum() + 1e-6)

def best_thresh_dice(prob, gt):            # same protocol as the notebook: best threshold per image
    return max(dice(prob > t, gt) for t in np.linspace(0.05, 0.95, 19))

def session(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

def bench(path, tag, folder=None):
    s = session(path); inp = s.get_inputs()[0].name
    x = np.random.rand(1, H, W, 3).astype(np.float32)
    for _ in range(10): s.run(None, {inp: x})
    lat = []
    for _ in range(50):
        t = time.perf_counter(); s.run(None, {inp: x}); lat.append((time.perf_counter()-t)*1e3)
    r = {"latency_ms_median": round(statistics.median(lat), 1), "latency_ms_p90": round(float(np.percentile(lat, 90)), 1)}
    proc = psutil.Process(); base = proc.memory_info().rss; peak = base
    if folder:
        d05, dbest, n = [], [], 0
        for img, gt in dataset(folder):
            p = s.run(None, {inp: img})[0].squeeze()
            d05.append(dice(p > 0.5, gt)); dbest.append(best_thresh_dice(p, gt)); n += 1
            peak = max(peak, proc.memory_info().rss)
        r.update({"n_images": n, "dice_at_0.5": round(float(np.mean(d05)), 4), "dice_best_thresh": round(float(np.mean(dbest)), 4)})
    r["peak_rss_mb_above_baseline"] = round((peak-base)/1e6, 1)
    res[tag] = r; print(tag, json.dumps(r))

folder = sys.argv[1] if len(sys.argv) > 1 else None
bench(FP32, "fp32", folder)

if folder:
    class Calib(CalibrationDataReader):
        def __init__(self): self.it = (img for img, _ in dataset(folder)); self.n = 0
        def get_next(self):
            if self.n >= 20: return None
            self.n += 1; x = next(self.it, None)
            return None if x is None else {session(FP32).get_inputs()[0].name: x}
    quantize_static(FP32, INT8, Calib(), quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8, per_channel=True)
    res["onnx_mb_int8"] = os.path.getsize(INT8)/1e6
    bench(INT8, "int8", folder)
    res["dice_drop_fp32_to_int8_best_thresh"] = round(res["fp32"]["dice_best_thresh"] - res["int8"]["dice_best_thresh"], 4)

json.dump(res, open("bench_results.json", "w"), indent=2); print(json.dumps(res, indent=2))
