# LFA-Net

A lightweight retinal vessel segmentation network, **110,833 parameters** (0.42 MB),
trained and evaluated on DRIVE, STARE and CHASE_DB1.

> M. Mehmood et al., *LFA-Net: A Lightweight Network with LiteFusion Attention
> for Retinal Vessel Segmentation*, ICCVDM 2025.

## Contents

    lfanet.py              the model, as a standalone module
    torch/                 PyTorch port, training and weight conversion
    predict.py             run a checkpoint over a folder of images
    evaluate.py            metrics in Python, matching Prediction_code/
    Prediction_code/       the original MATLAB evaluation (reference)
    Model_weights/         trained weights for DRIVE, STARE and CHASE_DB1
    LFA-Net.ipynb          the training notebook
    requirements.txt

## Architecture

An encoder-decoder with three downsampling stages, a fused-attention bottleneck
and three upsampling stages.

**Encoder.** Each stage is a `conv_block`: three parallel convolutions over the
same input, 1x1, 3x3, and 3x3 dilated by 2, all without bias, summed and passed
through LeakyReLU(0.1). Max-pooling then batch normalisation follow. Widths are
8, 16 and 32.

**Bottleneck (`LiteFusion_Attention`).** A 1x1 projection, then LayerNorm and a
3x3 convolution, then context aggregation, then a 1x1 projection and a residual
add, then a Mamba-inspired block (LayerNorm, depthwise 3x3 token mixer, LayerNorm,
two-layer channel mixer, both residual).

**Context aggregation.** A 3x3 and a 1x1 branch over the same input. The 1x1
branch is globally average-pooled and passed through a 1x1 sigmoid to gate the
3x3 branch. The gated result is then scaled by a second gate built from the
difference between its global maximum and global mean (`alpha = 0.25`), raised
elementwise to `gamma = 2`, and concatenated with the ungated branch. Both gates
are derived from globally pooled statistics, so the module acts per channel.

**Skip attention (`RA_attention_block`).** A 3x3 convolution, batch norm and
ReLU, from which a channel descriptor is formed by multiplying the global max
and global average pools. The descriptor rescales the spatially averaged feature
map, and the result gates the block's input.

**Decoder.** Three transposed convolutions of stride 2, each concatenated with an
`RA_attention_block` over the matching encoder stage and followed by a 3x3
convolution. The head is a 1x1 convolution with a sigmoid.

## Results

Held-out test images, scored with the protocol in `Prediction_code/Evaluation.m`:
a per-image threshold maximising F1 against that image's own mask, then
`bwareaopen(out, 50)`, over the full frame with no field-of-view mask.

Reported figures use 8-way test-time augmentation: the image is predicted at
four rotations and two flips and the maps are averaged. Single-pass numbers are
given alongside, since TTA costs eight forward passes.

| dataset | Dice | Dice, single pass | AP | AP, single pass |
|---|---|---|---|---|
| DRIVE | **0.8179** | 0.8088 | **0.9052** | 0.8932 |
| STARE | **0.8478** | 0.8393 | **0.9256** | 0.9118 |
| CHASE | **0.8392** | 0.8289 | **0.9213** | 0.9049 |

Average precision is the area under the precision-recall curve. It needs no
threshold, so it measures the ranking rather than the operating point.

These come from the weights in `Model_weights/`, which were retrained with the
recipe in `LFA-Net.ipynb`. Reproduce them with:

    python evaluate.py --weights Model_weights/LF-Net_DRIVE.h5 \
                       --images path/to/test/images --gt path/to/test/GT --tta

## Install

    pip install -r requirements.txt

The weights are Keras 2 `.h5` files, so the model is built under `tf_keras`
(legacy Keras) rather than Keras 3.

## Predict

    python predict.py --weights Model_weights/LF-Net_DRIVE.h5 \
                      --images path/to/test/images \
                      --out predictions/

Writes one 8-bit PNG per input as `<name>_Pred.png`, which is the layout
`Prediction_code/Evaluation.m` expects.

Preprocessing is the notebook's: `skimage.transform.resize` to 512x512 with
`preserve_range`, then divide by 255. Note that skimage anti-aliases when
downsampling and PIL's `BILINEAR` does not, so substituting one for the other
changes the result on DRIVE's 584x565 inputs.

## Evaluate

The reference implementation is MATLAB, in `Prediction_code/`. Run `predict.py`
first, put `<n>_Pred.png` and `<n>_GT.png` in one folder, set `path` at the top
of `Evaluation.m` and run it.

`evaluate.py` is a Python port of the same protocol, for convenience:

    python evaluate.py --weights Model_weights/LF-Net_DRIVE.h5 \
                       --images path/to/test/images \
                       --gt path/to/test/GT

Both use the protocol from the paper: a per-image threshold chosen to maximise
F1 against that image's own mask, then `bwareaopen(out, 50)` to drop connected
components below 50 pixels, scored over the full frame with no field-of-view
mask and averaged per image.

Two things are worth knowing when quoting numbers from this repository.

**The per-image threshold uses the test ground truth.** It is the convention in
much of the retinal vessel literature and it is what the paper reports, but it
is an upper bound rather than a deployment figure: a running system has no mask
to tune against. Add `--tta` for 8-way test-time augmentation.

**The model's outputs are not calibrated to 0.5.** It emits values above 0.5
over roughly a quarter of the frame, where under a tenth is vessel, and the
per-image optimal thresholds sit close to 1. Evaluating at a fixed 0.5, or
sweeping a linear grid that stops at 0.995, will substantially understate it.

## Weights

| file | dataset | parameters |
|---|---|---|
| `Model_weights/LF-Net_DRIVE.h5` | DRIVE | 110,833 |
| `Model_weights/LF-Net_STARE.h5` | STARE | 110,833 |
| `Model_weights/LF-Net_CHASE.h5` | CHASE_DB1 | 110,833 |

## Train

`LFA-Net.ipynb` holds the training run: `DiceLoss` with a squared denominator,
Adam at 1e-3, batch 16, up to 300 epochs, keeping the best validation loss. Data
is loaded by `get_data()`, which expects a folder containing `images/` and
`GT_BV/`.

## Citation

    @inproceedings{mehmood2025lfanet,
      title     = {LFA-Net: A Lightweight Network with LiteFusion Attention for
                   Retinal Vessel Segmentation},
      author    = {Mehmood, Mehwish and others},
      booktitle = {International Conference on Computer Vision and Data Mining (ICCVDM)},
      year      = {2025}
    }

## Training

TensorFlow on some machines has no CUDA support, which makes the notebook's
recipe slow on CPU. `torch/` holds a PyTorch port of the same model for GPU
training:

    python torch/verify_port.py Model_weights/LF-Net_DRIVE.h5   # checks the port
    python torch/train_torch.py --data path/to/train_aug --out LF-Net_DRIVE.pt
    python torch/torch_to_keras.py --torch-weights LF-Net_DRIVE.pt \
                                   --out Model_weights/LF-Net_DRIVE.h5

`verify_port.py` loads the Keras weights into the PyTorch model and compares the
outputs; it agrees to 1.5e-05. One detail matters there: TensorFlow's
`padding='same'` transposed convolution is not PyTorch's
`padding=1, output_padding=1`. It is `padding=0` with the result cropped to twice
the input size from the top left, and getting it wrong changes the output by
about 14 while leaving the parameter count identical.
