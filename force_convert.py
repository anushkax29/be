import tensorflow as tf
import h5py
import numpy as np

def force_convert_h5_to_keras(h5_path, output_path):
    # Create a dummy model with matching architecture
    def build_replacement_model():
        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Load weights from original model
    with h5py.File(h5_path, 'r') as f:
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        weights = [np.array(f[layer_name]) for layer_name in layer_names]
    
    # Create and save new model
    new_model = build_replacement_model()
    new_model.set_weights(weights)
    new_model.save(output_path)
    print(f"✅ Model forcibly converted to {output_path}")

# Usage
force_convert_h5_to_keras(
    r"C:\Users\mishr\Desktop\addahfe\server\color_recommender_cnn.h5",
    r"C:\Users\mishr\Desktop\addahfe\server\converted_model.keras"
)