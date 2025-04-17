from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

app = Flask(__name__)

CHROME_DRIVER_PATH = r"C:\Users\mishr\Desktop\addahfe\server\chromedriver\chromedriver.exe"

def fetch_outfit_images(url, max_images=6):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

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
        if len(image_urls) >= max_images:
            break

    driver.quit()
    return image_urls

@app.route('/fetch-images', methods=['POST'])
def fetch_images():
    url = request.json.get('url')
    if not url:
        return jsonify({"error": "URL not provided"}), 400

    try:
        images = fetch_outfit_images(url)
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
