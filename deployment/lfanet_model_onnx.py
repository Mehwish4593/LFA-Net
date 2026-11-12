import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import numpy as np
import numpy as geek
def sem_attention_block(inputs):
    shape=inputs.shape
    F2=Conv2D(shape[3],1, padding='same') (inputs)
    F2=BatchNormalization() (F2)
    F2=Activation('relu') (F2)
    x1=MaxPooling2D(pool_size=(shape[1],shape[2]),padding='same')(F2)
    x2=AveragePooling2D(pool_size=(shape[1],shape[2]),padding='same')(F2)
    con1 = Concatenate()([x1,x2])
    F3=Conv2D(shape[3],1, padding='same') (con1)
    F3=BatchNormalization() (F3)
    F33=Activation('relu') (F3)
    xa=GlobalAveragePooling2D(keepdims=True)(inputs)
    xa=Conv2D(shape[3],1, padding='same') (xa)
    xaa=BatchNormalization() (xa)
    xaa=Activation('sigmoid') (xaa)
    xm=Multiply()([F33,xaa])
    xam=Add()([inputs,xm])
    return xam
import tensorflow as tf
from tensorflow.keras import layers

# -----------------------------------------------
# FOCAL MODULATION BLOCK
# -----------------------------------------------
def focal_modulation(inputs, gamma=2.0, alpha=0.25):
    num_channels = inputs.shape[-1]

    mean = layers.GlobalAveragePooling2D()(inputs)
    max_val = layers.GlobalMaxPooling2D()(inputs)

    modulation = (max_val - mean) * alpha
    modulation = layers.Reshape((1, 1, num_channels))(modulation)
    modulation = layers.Conv2D(filters=num_channels, kernel_size=1, activation='sigmoid')(modulation)

    scaled_inputs = layers.Multiply()([inputs, modulation])
    outputs = layers.Lambda(lambda x: x ** gamma)(scaled_inputs)
    return outputs

# -----------------------------------------------
# FOCAL MODULATION + CONTEXT AGGREGATION BLOCK
# -----------------------------------------------
def focal_modulation_context_aggregation(inputs, filters):
    filters = int(filters)

    conv1 = layers.Conv2D(filters, kernel_size=3, padding='same', activation='relu')(inputs)

    conv2 = layers.Conv2D(filters, kernel_size=1, activation='relu')(inputs)
    global_context = layers.GlobalAveragePooling2D()(conv2)
    global_context = layers.Reshape((1, 1, filters))(global_context)
    global_context = layers.Conv2D(filters, kernel_size=1, activation='sigmoid')(global_context)
    global_context = layers.Multiply()([conv1, global_context])

    FM = focal_modulation(global_context)

    concatenated = layers.Concatenate()([conv1, FM])
    return concatenated

# -----------------------------------------------
# MAMBA-INSPIRED BLOCK
# -----------------------------------------------
class VisionMambaInspired(tf.keras.layers.Layer):
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization()
        self.token_mixer = tf.keras.Sequential([
            layers.DepthwiseConv2D(kernel_size=3, padding='same'),
            layers.Lambda(lambda t: 0.5 * t * (1.0 + tf.math.erf(t / 1.4142135623730951))),
            layers.Dropout(dropout_rate),
        ])
        self.norm2 = layers.LayerNormalization()
        self.channel_mixer = tf.keras.Sequential([
            layers.Dense(dim * 4),
            layers.Lambda(lambda t: 0.5 * t * (1.0 + tf.math.erf(t / 1.4142135623730951))),
            layers.Dropout(dropout_rate),
            layers.Dense(dim),
            layers.Dropout(dropout_rate),
        ])

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.token_mixer(x) + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.channel_mixer(x) + shortcut
        return x

# -----------------------------------------------
# FINAL COMBINED BLOCK FOR BOTTLENECK
# -----------------------------------------------
def LiteFusion_Attention(input_tensor, filters, padding="same"):
    # Project input channels
    input_tensor = layers.Conv2D(filters, (1, 1), padding=padding)(input_tensor)

    # LayerNorm + Conv
    x = layers.LayerNormalization()(input_tensor)
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)

    # Context aggregation with focal modulation
    x = focal_modulation_context_aggregation(x, filters)

    # Project back to match residual shape
    x = layers.Conv2D(filters, (1, 1), padding=padding)(x)

    # Residual connection
    out = layers.Add()([x, input_tensor])

    # Vision Mamba Block
    out = VisionMambaInspired(dim=filters)(out)

    return out
input_shape = (512,512,3)
# inputs = layers.Input(shape=input_shape)
# p = LiteFusion_Attention(inputs, filters=24)
# output = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid',name = 'final_output1')(p)
# model = models.Model(inputs=inputs, outputs=[output])
# model = LiteFusion_Attention(inputs, filters=32)
# model.summary()
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate
from tensorflow.keras import backend as K


#  import numpy as np
def RA_attention_block(inputs,n_classes,k):
    shape=K.int_shape(inputs)
    # F=Conv2D(k*n_classes,1, padding='same') (inputs)
    F=Conv2D(k*n_classes,3, padding='same') (inputs)
    F=BatchNormalization() (F)
    F1=Activation('relu') (F)

    F2=F1

    x1=GlobalMaxPool2D()(F2)
    x2=GlobalAveragePooling2D()(F2)
    x = Multiply() ([x1,x2])
    x=Reshape((n_classes,k))(x)
    S=Lambda(lambda x: K.mean(x,axis=-1,keepdims=False)) (x)

    # x2=Reshape((n_classes,k)) (x)
    # S2=Lambda(lambda x: K.mean(x,axis=-1,keepdims=False))  (x2)
    # S= S1+S2
    x=Reshape((shape[1],shape[2],n_classes,k)) (F1)
    x=Lambda(lambda x: K.mean(x,axis=-1,keepdims=False))  (x)
    x=Multiply()([S,x])
    M=Lambda(lambda x: K.mean(x,axis=-1,keepdims=True))  (x)

    semantic=Multiply()([inputs,M])
    return semantic

def conv_block(input, out_channels, dropout_rate=0.5):
    # Multi-scale convolutions
    conv1x1 = layers.Conv2D(out_channels, kernel_size=1, padding='same', use_bias=False)(input)
    conv3x3 = layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False)(input)
    conv3x3_dilated = layers.Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False, dilation_rate=2)(input)
    
    # Fuse multi-scale outputs by summing them
    x = conv1x1 + conv3x3 + conv3x3_dilated
    x = layers.LeakyReLU(alpha=0.1)(x)
#     x = Dropout(dropout_rate)(x)
    return x


def LiteFusion_UNet(input_shape, num_classes, feature_scale=2, dropout=0.5):
    # Define the input layer
    filters = [int(x / feature_scale) for x in [16,32,64]]
    inputs = layers.Input(shape=input_shape)

    # Encoder - Convolutional Blocks
    conv_block1 = conv_block(inputs, filters[0], dropout)
#     conv_block1 = layers.Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv_block1 = layers.MaxPooling2D(pool_size=(2, 2))(conv_block1)
    conv_block1 = layers.BatchNormalization()(conv_block1)

    conv_block2 = conv_block(conv_block1, filters[1], dropout)
#     conv_block2 = layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(conv_block1)
    conv_block2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_block2)
    conv_block2 = layers.BatchNormalization()(conv_block2)

    conv_block3 = conv_block(conv_block2, filters[2], dropout)
#     conv_block3 = layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(conv_block2)
    conv_block3 = layers.MaxPooling2D(pool_size=(2, 2))(conv_block3)
    conv_block3 = layers.BatchNormalization()(conv_block3)

    attention_output = LiteFusion_Attention(conv_block3, filters=32)


    # Concatenate the Encoder Convolutional and Transformer blocks

    att1 = RA_attention_block(attention_output,1,16)
    fused_features = layers.Concatenate()([att1, attention_output])

    # Decoder - Upsampling Blocks
    decoder_block1 = layers.Conv2DTranspose(filters[2], kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(fused_features)
    att2 = RA_attention_block(conv_block2,1,16)
    decoder_block1 = layers.Concatenate()([att2, decoder_block1])
    decoder_block1 = layers.Conv2D(filters[2], kernel_size=(3, 3), padding='same', activation='relu')(decoder_block1)

    decoder_block2 = layers.Conv2DTranspose(filters[2], kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(decoder_block1)

    att3=RA_attention_block(conv_block1,1,16)
    decoder_block2 = layers.Concatenate()([att3, decoder_block2])
    decoder_block2 = layers.Conv2D(filters[2], kernel_size=(3, 3), padding='same', activation='relu')(decoder_block2)


    decoder_block3 = layers.Conv2DTranspose(filters[0], kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(decoder_block2)
    decoder_block3 = layers.Conv2D(filters[0], kernel_size=(3, 3), padding='same', activation='relu')(decoder_block3)
    # Decoder - Output Block
    output = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid',name = 'final_output1')(decoder_block3)
#     output_OD = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name = 'final_output2')(decoder_block3)

    # Create the model
    model = models.Model(inputs=inputs, outputs=[output])

    return model

# Define the input shape for binary semantic segmentation
input_shape = (512,512,3)  # Example input shape for RGB images

# Create the TransFuse Encoder-Decoder model with 3 MaxPooling layers for binary semantic segmentation


