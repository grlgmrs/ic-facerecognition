from keras.src.models import Model
from keras.src.layers import (
    Input,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    Flatten,
)
from euclidian_distance import EuclidianDistance
import config
import keras


# Esse aqui é basicamente pra reconhecer imagens blurry de deblurry
# def get_feature_extractor(inputShape, embeddingDim=48):
#     inputs = Input(inputShape)

#     x = Conv2D(64, (2, 2), padding="same", activation="relu")(inputs)
#     x = MaxPooling2D(pool_size=(2, 2))(x)
#     x = Dropout(0.3)(x)

#     x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
#     x = MaxPooling2D(pool_size=2)(x)
#     x = Dropout(0.3)(x)

#     pooledOutput = GlobalAveragePooling2D()(x)
#     outputs = Dense(embeddingDim)(pooledOutput)

#     model = Model(inputs, outputs)

#     return model


def get_feature_extractor(inputShape, embeddingDim=48):
    inputs = Input(inputShape)
    base = keras.applications.ResNet50V2(
        weights="imagenet", include_top=False, input_tensor=inputs
    )

    for layer in base.layers:
        layer.trainable = False

    x = base.output

    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)

    model = Model(inputs, outputs)

    return model


def build_siamese_network():
    print("[INFO] building siamese network...")
    imgSharpInput = Input(shape=config.IMG_SHAPE)
    imgBlurInput = Input(shape=config.IMG_SHAPE)

    feature_extractor = get_feature_extractor(config.IMG_SHAPE)
    featsSharp = feature_extractor(imgSharpInput)
    featsBlur = feature_extractor(imgBlurInput)

    distance = EuclidianDistance()([featsSharp, featsBlur])
    outputs = Dense(1, activation="sigmoid")(distance)
    model = Model(inputs=[imgSharpInput, imgBlurInput], outputs=outputs)

    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
