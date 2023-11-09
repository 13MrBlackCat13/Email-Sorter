import os
import requests
import logging
import shutil
from dotenv import load_dotenv

# Configuration
from config import Config

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility functions
def read_eml_file(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except IOError as e:
        logging.error(f"Error reading file {path}: {e}")
        return None

def move_file_to_category(file_path, file_name, category):
    sorted_folder = os.path.join("Sorted", category)
    if not os.path.exists(sorted_folder):
        os.makedirs(sorted_folder)

    new_path = os.path.join(sorted_folder, file_name)
    shutil.move(file_path, new_path)
    logging.info(f"Moved {file_name} to {sorted_folder}")

# API interaction
def process_file(file_path, file_name):
    eml_content = read_eml_file(file_path)
    if eml_content is None:
        return

    data = {"model_name": "all", "text_only": "true"}

    try:
        response = requests.post(Config.API_URL, files={"file": (file_name, eml_content, "message/rfc822")}, params=data)
        if response.status_code == 200:
            handle_response(response.json(), file_path, file_name)
        else:
            logging.error(f"Error processing {file_name}: {response.text}")
    except requests.RequestException as e:
        logging.error(f"Request failed: {e}")

def handle_response(response_data, file_path, file_name):
    avg_probabilities = response_data[-1]
    category_probabilities = {k: v for k, v in avg_probabilities.items() if k != 'Model'}

    for category, probability in category_probabilities.items():
        if float(probability) >= Config.PROBABILITY_THRESHOLD:
            move_file_to_category(file_path, file_name, category)
            break

# Main execution
def main():
    for root, dirs, files in os.walk(Config.UNSORTED_DIR):
        for file in files:
            if file.endswith(".eml"):
                process_file(os.path.join(root, file), file)

if __name__ == "__main__":
    main()
