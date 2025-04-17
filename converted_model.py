from keras.models import load_model

# Load your old .h5 model
model = load_model("color_recommender_cnn.h5", compile=False)

# Save it as a modern .keras model
model.save("converted_model.keras", save_format="keras")

print("✅ Model converted and saved as converted_model.keras!")
