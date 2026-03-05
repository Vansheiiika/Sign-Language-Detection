import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(26, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.save("outputs/baseline_cnn_best.h5")

print("✅ Model created successfully!")
