"""
Rebuild LFA-Net from the notebook code, load DRIVE weights, export to ONNX. Only needed if you want to
re-export yourself; lfanet_drive.onnx is already provided.
pip install tensorflow tf-keras tf2onnx onnx
Run with:  set TF_USE_LEGACY_KERAS=1  (PowerShell: $env:TF_USE_LEGACY_KERAS="1")
"""
import os; os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np, tensorflow as tf, subprocess, sys
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfanet_model_onnx.py")).read(), globals())           # defines LiteFusion_UNet with erf-GELU
model = LiteFusion_UNet((512, 512, 3), 1, feature_scale=2, dropout=0.5)
model.load_weights("../Model_weights/LF-Net_DRIVE.h5")            # the retrained DRIVE weights
print("params:", model.count_params())
model.save("lfanet_drive_keras", save_format="tf")
subprocess.check_call([sys.executable, "-m", "tf2onnx.convert", "--saved-model", "lfanet_drive_keras",
                       "--output", "lfanet_drive.onnx", "--opset", "17"])
print("exported lfanet_drive.onnx")
