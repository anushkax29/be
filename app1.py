from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from colorthief import ColorThief
import webcolors
import os
from keras.models import load_model
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from flask_cors import CORS
from keras.models import load_model


MODEL_PATH = 'color_recommender_cnn.h5'

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")



# --- Flask Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load CNN Model ---
MODEL_PATH = 'color_recommender_cnn.h5'
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Helper Functions ---
def get_dominant_color(image_path):
    color_thief = ColorThief(image_path)
    return color_thief.get_color(quality=1)

def get_color_name(rgb_tuple):
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        return "unknown"

def recommend_outfit(color):
    color_map = {
        "blue": "Try pairing with white or beige bottoms.",
        "black": "Black goes with almost anything! Try red or gray.",
        "red": "Red pairs well with black or denim.",
        "white": "White works with any color, but try dark tones for contrast."
    }
    return color_map.get(color, "Neutral tones like black, white, or denim always work!")

def preprocess_image(img_path):
    img = Image.open(img_path).resize((128, 128))
    img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# --- Routes ---

@app.route('/color-recommendation', methods=['POST'])
def color_recommendation():
    if 'image' not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    image = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    dominant_rgb = get_dominant_color(path)
    color_name = get_color_name(dominant_rgb)
    outfit = recommend_outfit(color_name)

    return jsonify({
        "dominant_rgb": dominant_rgb,
        "color_name": color_name,
        "suggestion": outfit
    })

@app.route('/validate-inputs', methods=['POST'])
def validate_inputs():
    image = request.files.get('image')
    occasion = request.form.get('occasion', '').lower()
    weather = request.form.get('weather', '').lower()

    if not image:
        return jsonify({"error": "No image provided"}), 400

    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    try:
        img = Image.open(filepath)
        img.verify()
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    allowed_occasions = ["casual", "formal", "semi formal", "party"]
    allowed_weather = ["summer", "winter", "rainy", "fall", "spring"]

    if occasion not in allowed_occasions:
        return jsonify({"error": f"Invalid occasion. Choose from {allowed_occasions}"}), 400
    if weather not in allowed_weather:
        return jsonify({"error": f"Invalid weather. Choose from {allowed_weather}"}), 400

    return jsonify({
        "message": "Valid inputs",
        "occasion": occasion,
        "weather": weather
    })

@app.route('/generate-pinterest-url', methods=['POST'])
def generate_url():
    data = request.json
    occasion = data.get('occasion')
    weather = data.get('weather')
    colors = data.get('colors', [])

    if not occasion or not weather or not colors:
        return jsonify({"error": "Missing occasion, weather or colors"}), 400

    query = f"{occasion} {weather} outfit in {' and '.join(colors)}"
    url = f"https://in.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"
    return jsonify({"url": url})

@app.route('/fetch-images', methods=['POST'])
def fetch_images():
    url = request.json.get('url')
    if not url:
        return jsonify({"error": "URL not provided"}), 400

    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")

        CHROME_DRIVER_PATH = r"C:\Users\mishr\Desktop\addahfe\server\chromedriver\chromedriver.exe"
        service = Service(CHROME_DRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        time.sleep(3)

        images = driver.find_elements(By.TAG_NAME, "img")
        image_urls = []

        for img in images:
            src = img.get_attribute("src")
            if src and "pinimg.com" in src and "236x" in src:
                image_urls.append(src)
            if len(image_urls) >= 6:
                break

        driver.quit()
        return jsonify({"images": image_urls})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
@app.route('/predict-outfit', methods=['POST'])
def predict_outfit():
    if not model:
        return jsonify({"error": "Model not available"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    image = request.files['image']
    path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(path)

    try:
        processed = preprocess_image(path)
        prediction = model.predict(processed)
        class_index = np.argmax(prediction[0])
        return jsonify({
            "predicted_class_index": int(class_index),
            "confidence": float(np.max(prediction[0]))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Main ---
if __name__ == '__main__':
    app.run(debug=True)