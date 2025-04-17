from flask import Flask, request, jsonify
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/validate-inputs', methods=['POST'])
def validate_inputs():
    image = request.files.get('image')
    occasion = request.form.get('occasion', '').lower()
    weather = request.form.get('weather', '').lower()

    if not image:
        return jsonify({"error": "No image provided"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

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

if __name__ == '__main__':
    app.run(debug=True)