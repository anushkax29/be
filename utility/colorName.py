from flask import Flask, request, jsonify
from PIL import Image
from colorthief import ColorThief
import webcolors
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

if __name__ == '__main__':
    app.run(debug=True)