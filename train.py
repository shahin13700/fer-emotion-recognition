"""
Base model training for EdgeVision (MiniXception on FER2013).

This mirrors notebooks/exploration_and_training.ipynb, which produced the shipped
model/emotion_model.keras (L2(0.01) on the two stem Conv2D layers and the Dense
classifier, class-balanced weights, checkpoint on val_accuracy, early stopping on val_loss). Run it to reproduce
a comparable model; results vary run to run because no seed is fixed.
"""
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------
# 1. Setup Directories
# ---------------------------------------------------------
def setup_dirs():
    """Creates necessary directories before training."""
    directories = ['model', 'outputs']
    for d in directories:
        os.makedirs(d, exist_ok=True)
        print(f"Directory ready: {d}/")

# ---------------------------------------------------------
# 2. Build the MiniXception Architecture
# ---------------------------------------------------------
def build_minixception(input_shape=(48, 48, 1), num_classes=7, l2_weight=0.01):
    """
    Builds the MiniXception lightweight CNN architecture (~51k parameters).
    Uses depthwise separable convolutions to reduce parameters and combat overfitting.
    L2 regularization is applied where the shipped weights have it: the two stem Conv2D
    layers and the Dense classifier (Keras 3 SeparableConv2D has no kernel_regularizer).
    """
    reg = l2(l2_weight)
    img_input = Input(shape=input_shape)

    # Base convolution block
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=reg)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ---------------------------------------------------------
    # Module 1
    # ---------------------------------------------------------
    residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])

    # ---------------------------------------------------------
    # Module 2
    # ---------------------------------------------------------
    residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])

    # ---------------------------------------------------------
    # Module 3
    # ---------------------------------------------------------
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])

    # ---------------------------------------------------------
    # Module 4
    # ---------------------------------------------------------
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.add([x, residual])

    # ---------------------------------------------------------
    # Classification Head
    # ---------------------------------------------------------
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=reg)(x)

    model = Model(img_input, output)
    return model

# ---------------------------------------------------------
# 3. Main Training Execution
# ---------------------------------------------------------
def main():
    setup_dirs()
    print("Preparing Datasets...")

    batch_size = 64
    target_size = (48, 48)
    epochs = 100
    dataset_path = 'dataset/train' # Test set is excluded per instructions

    # We use two generators with validation_split from the SAME parent folder.
    # This prevents augmenting validation data, which ruins evaluation metrics.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2  # 20% validation split
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("\nCalculating class weights based on train generator distributions...")
    classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weight_dict = dict(zip(np.unique(classes), class_weights))
    print("Class weights dict:", class_weight_dict)

    print("\nBuilding MiniXception model...")
    model = build_minixception(input_shape=(48, 48, 1), num_classes=7)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks configuration
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath='model/emotion_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\nStarting Training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Save class indices for demo.py
    indices_path = 'outputs/class_indices.json'
    with open(indices_path, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"\nClass indices saved into {indices_path}.")

    # Save training history with pickle
    history_path = 'outputs/history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\nTraining complete. History successfully saved into {history_path}.")

if __name__ == '__main__':
    main()
