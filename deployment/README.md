# LFA-Net deployment

ONNX export of LFA-Net (DRIVE weights), a runner and a CPU benchmark.

## Files

    lfanet_drive.onnx        FP32 model, 110,833 parameters, 0.52 MB
    run_onnx.py              run the model on an image or a folder, and score it
    export_lfanet.py         rebuild from Keras and re-export with tf2onnx (opset 17)
    lfanet_model_onnx.py     model code used by the export
    bench_lfanet_onnx.py     onnxruntime latency benchmark
    bench_results.json       output of the last benchmark run

Input is `1x512x512x3` float32 in [0,1], NHWC. Output is `1x512x512x1` after a
sigmoid. The graph input is named `input_1` and the output `final_output1`.

## Run it

Running the ONNX needs no TensorFlow:

    pip install onnxruntime numpy pillow scikit-image scipy scikit-learn

Score a test set:

    python run_onnx.py --onnx lfanet_drive.onnx \
                       --images path/to/test/images --gt path/to/test/GT_BV --out preds

Or just run one image, without masks:

    python run_onnx.py --onnx lfanet_drive.onnx --images 01_test.tif --out preds

`--images` and `--gt` are paired by sorted filename, so both folders need the
same ordering. `--out` writes one 8-bit PNG per input as `<name>_Pred.png`.
`--threads` defaults to 4.

Preprocessing inside `run_onnx.py` is `skimage.transform.resize` to 512x512
with `preserve_range`, then divide by 255, which is what the model was trained
with. PIL's `BILINEAR` is not a substitute: skimage anti-aliases when
downsampling and PIL does not, and on DRIVE's 584x565 inputs that costs
accuracy.

## Results

DRIVE, the 20 official test images with the first observer's annotations.
Per-image threshold maximising F1 against that image's own mask, then
components under 50 px dropped.

| | Dice | AP |
|---|---|---|
| lfanet_drive.onnx | 0.8088 | 0.8932 |

Keras against ONNX on the same input: max absolute difference 7.9e-06.

Latency, one 512x512 image, batch 1, 4 CPU threads: 22.0 ms median, 22.2 ms
p90, measured on an aarch64 Grace Blackwell host. `bench_results.json` holds
that run. Expect a different figure on other hardware; an Intel laptop CPU is
slower.

## Re-exporting

Only needed if the weights change. This step does need TensorFlow:

    pip install tensorflow tf-keras tf2onnx onnx

    cd Deploy
    export TF_USE_LEGACY_KERAS=1        # Windows PowerShell: $env:TF_USE_LEGACY_KERAS="1"
    python export_lfanet.py

It reads `../Model_weights/LF-Net_DRIVE.h5`, prints `params: 110833` and
overwrites `lfanet_drive.onnx`.

The weights are Keras 2 `.h5`, so the model has to be built under `tf_keras`
(legacy Keras) rather than Keras 3, which is what `TF_USE_LEGACY_KERAS=1` does.

One detail in `lfanet_model_onnx.py`: the exact GELU is written out as
`0.5*x*(1+erf(x/sqrt(2)))` rather than called through TensorFlow's own
activation. TensorFlow implements it with `Erfc`, and ONNX has no `Erfc`
operator, so the export fails otherwise.

## A note on the numbers

If a run gives 0.6985 Dice rather than 0.8088, the ONNX file is the earlier
export, taken from the original released weights before the model was
retrained. Re-export from the current `Model_weights/LF-Net_DRIVE.h5` and the
figure returns to 0.8088.

Quoting the Dice figure: the threshold is chosen per image against that image's
own ground truth. That is the convention in this literature and is what the
number above follows, but it is an upper bound rather than a deployment figure,
since a running system has no mask to tune against. The AP column needs no
threshold.
