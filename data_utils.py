
import os
import pandas as pd
import logging

def load_data(data_dir):
    categories = os.listdir(data_dir)
    messages = []
    categories_list = []
    for category in categories:
        for message in os.listdir(os.path.join(data_dir, category)):
            try:
                # Try reading in utf-8
                with open(os.path.join(data_dir, category, message), 'r', encoding='utf-8') as f:
                    messages.append(f.read())
                    categories_list.append(category)
            except UnicodeDecodeError:
                try:
                    # Try reading in utf-16 without BOM
                    with open(os.path.join(data_dir, category, message), 'r', encoding='utf-16le', errors='ignore') as f:
                        messages.append(f.read())
                        categories_list.append(category)
                except UnicodeError:
                    logging.error(f"Error decoding file: {os.path.join(data_dir, category, message)}")

    data = pd.DataFrame({'text': messages, 'category': categories_list})
    return data
