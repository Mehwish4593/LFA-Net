# LFA-Net deployment

ONNX export of LFA-Net (DRIVE weights) and a CPU benchmark script.

## Files
- lfanet_drive.onnx: FP32 model, 110,833 parameters, 0.52 MB, input 1x512x512x3 float in [0,1], output 1x512x512x1 sigmoid.
- export_lfanet.py: rebuilds the Keras model, loads LF-Net_DRIVE.h5, exports with tf2onnx (opset 17).
- lfanet_model_onnx.py: model code with exact GELU written as 0.5*x*(1+erf(x/sqrt2)) so tf2onnx can export it (TensorFlow's GELU uses Erfc, which ONNX lacks).
- bench_lfanet_onnx.py: onnxruntime latency benchmark (10 warmup, 50 timed, median and p90, 4 threads). Pass a DRIVE test folder to also compute Dice and an INT8 comparison.
- bench_results.json: results from the sandbox run.

## Results
Intel Core Ultra 7 265U laptop CPU, 4 threads, 512x512, batch 1: 46.3 ms median, 50.4 ms p90.
Keras vs ONNX output parity: max abs diff 1.8e-5.
INT8 (static, QDQ): 0.40 MB but slower on CPU; the model is activation bound at 512x512.

## Run
pip install onnxruntime numpy pillow psutil
python bench_lfanet_onnx.py
