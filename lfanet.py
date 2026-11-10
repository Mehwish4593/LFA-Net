import os; os.environ["TF_USE_LEGACY_KERAS"]="1"; os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, models, backend as K
from tf_keras.layers import *

def gelu_erf(x):
    """Exact GELU written with erf so it exports to ONNX.

    tf_keras's Activation('gelu') can compile to Erfc depending on the TF
    version, and ONNX has no Erfc operator: the graph then fails to load in
    ONNX Runtime with "No Op registered for Erfc". erf is in ONNX from opset 9.
    Numerically identical, since erfc(z) = 1 - erf(z).
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))


def focal_modulation(inputs, gamma=2.0, alpha=0.25):
    c = inputs.shape[-1]
    mean = layers.GlobalAveragePooling2D()(inputs)
    mx = layers.GlobalMaxPooling2D()(inputs)
    mod = (mx - mean) * alpha
    mod = layers.Reshape((1,1,c))(mod)
    mod = layers.Conv2D(c, 1, activation='sigmoid')(mod)
    s = layers.Multiply()([inputs, mod])
    return layers.Lambda(lambda x: x ** gamma)(s)

def focal_modulation_context_aggregation(inputs, filters):
    filters=int(filters)
    conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(filters, 1, activation='relu')(inputs)
    g = layers.GlobalAveragePooling2D()(conv2)
    g = layers.Reshape((1,1,filters))(g)
    g = layers.Conv2D(filters, 1, activation='sigmoid')(g)
    g = layers.Multiply()([conv1, g])
    FM = focal_modulation(g)
    return layers.Concatenate()([conv1, FM])

class VisionMambaBlock(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1, **kw):
        super().__init__(**kw)
        self.norm1 = layers.LayerNormalization()
        self.token_mixer = keras.Sequential([layers.DepthwiseConv2D(3, padding='same'), layers.Activation(gelu_erf), layers.Dropout(dropout_rate)])
        self.norm2 = layers.LayerNormalization()
        self.channel_mixer = keras.Sequential([layers.Dense(dim*4), layers.Activation(gelu_erf), layers.Dropout(dropout_rate), layers.Dense(dim), layers.Dropout(dropout_rate)])
    def call(self, x):
        sc=x; x=self.norm1(x); x=self.token_mixer(x)+sc
        sc=x; x=self.norm2(x); x=self.channel_mixer(x)+sc
        return x

def LiteFusion_Attention(t, filters, padding="same"):
    t = layers.Conv2D(filters, 1, padding=padding)(t)
    x = layers.LayerNormalization()(t)
    x = layers.Conv2D(filters, 3, padding=padding)(x)
    x = focal_modulation_context_aggregation(x, filters)
    x = layers.Conv2D(filters, 1, padding=padding)(x)
    out = layers.Add()([x, t])
    return VisionMambaBlock(dim=filters)(out)

def RA_attention_block(inputs, n_classes, k):
    shape = K.int_shape(inputs)
    F = Conv2D(k*n_classes, 3, padding='same')(inputs)
    F = BatchNormalization()(F); F1 = Activation('relu')(F)
    x1 = GlobalMaxPool2D()(F1); x2 = GlobalAveragePooling2D()(F1)
    x = Multiply()([x1,x2]); x = Reshape((n_classes,k))(x)
    S = Lambda(lambda x: K.mean(x,axis=-1,keepdims=False))(x)
    x = Reshape((shape[1],shape[2],n_classes,k))(F1)
    x = Lambda(lambda x: K.mean(x,axis=-1,keepdims=False))(x)
    x = Multiply()([S,x])
    M = Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))(x)
    return Multiply()([inputs,M])

def conv_block(inp, out_channels, dropout_rate=0.5):
    a = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)(inp)
    b = layers.Conv2D(out_channels, 3, padding='same', use_bias=False)(inp)
    c = layers.Conv2D(out_channels, 3, padding='same', use_bias=False, dilation_rate=2)(inp)
    x = a+b+c
    return layers.LeakyReLU(alpha=0.1)(x)

def LiteFusion_UNet(input_shape=(512,512,3), num_classes=1, feature_scale=2, dropout=0.5):
    f = [int(x/feature_scale) for x in [16,32,64]]
    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, f[0], dropout); c1 = layers.MaxPooling2D(2)(c1); c1 = layers.BatchNormalization()(c1)
    c2 = conv_block(c1, f[1], dropout);     c2 = layers.MaxPooling2D(2)(c2); c2 = layers.BatchNormalization()(c2)
    c3 = conv_block(c2, f[2], dropout);     c3 = layers.MaxPooling2D(2)(c3); c3 = layers.BatchNormalization()(c3)
    att = LiteFusion_Attention(c3, filters=32)
    a1 = RA_attention_block(att,1,16); fused = layers.Concatenate()([a1, att])
    d1 = layers.Conv2DTranspose(f[2], 3, strides=2, padding='same', activation='relu')(fused)
    a2 = RA_attention_block(c2,1,16); d1 = layers.Concatenate()([a2, d1])
    d1 = layers.Conv2D(f[2], 3, padding='same', activation='relu')(d1)
    d2 = layers.Conv2DTranspose(f[2], 3, strides=2, padding='same', activation='relu')(d1)
    a3 = RA_attention_block(c1,1,16); d2 = layers.Concatenate()([a3, d2])
    d2 = layers.Conv2D(f[2], 3, padding='same', activation='relu')(d2)
    d3 = layers.Conv2DTranspose(f[0], 3, strides=2, padding='same', activation='relu')(d2)
    d3 = layers.Conv2D(f[0], 3, padding='same', activation='relu')(d3)
    out = layers.Conv2D(1, 1, activation='sigmoid', name='final_output1')(d3)
    return models.Model(inputs, out)

if __name__=="__main__":
    m = LiteFusion_UNet()
    print("params", m.count_params())
    m.load_weights("/home/claude/LFA-Net/w/Model_weights/LF-Net_DRIVE.h5")
    print("weights loaded OK")
    m.save_weights("/home/claude/edge/lfanet_drive_tfkeras.h5")
