from flask import Flask, request, jsonify

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)