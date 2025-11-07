import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# --- AUTO-DETECT PATHS ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "images", "fer2013", "train")
TEST_DIR  = os.path.join(BASE_DIR, "data", "images", "fer2013", "test")
OUT_MODEL = os.path.join(BASE_DIR, "models", "image_emotion.h5")

print("📂 Training folder:", TRAIN_DIR)
print("📂 Testing folder :", TEST_DIR)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training folder not found: {TRAIN_DIR}")

IMG_SIZE = 48
BATCH_SIZE = 64

def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

    train_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='sparse',
        batch_size=BATCH_SIZE
    )

    num_classes = len(train_gen.class_indices)
    print("Detected emotion classes:", train_gen.class_indices)

    model = build_model((IMG_SIZE, IMG_SIZE, 1), num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(OUT_MODEL, save_best_only=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=callbacks)
    print("✅ Training finished.")

    loss, acc = model.evaluate(test_gen)
    print(f"🧪 Test Accuracy: {acc:.3f}")

    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    model.save(OUT_MODEL)
    print("✅ Model saved to", OUT_MODEL)

if __name__ == "__main__":
    main()
