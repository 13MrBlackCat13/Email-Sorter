import os
import email
from bs4 import BeautifulSoup
import re
import shutil

def create_directory_structure():
    directories = [
        'data/raw', 'data/processed', 'data/raw/Unsorted',
        'models', 'logs',
        'data/raw/Входящие', 'data/raw/Рассылки', 'data/raw/Социальные сети',
        'data/raw/Чеки_Квитанции', 'data/raw/Новости', 'data/raw/Доставка',
        'data/raw/Госписьма', 'data/raw/Учёба', 'data/raw/Игры',
        'data/raw/Spam/Мошенничество', 'data/raw/Spam/Обычный'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def process_email(content):
    # Извлекаем текст из email
    msg = email.message_from_string(content)
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif part.get_content_type() == "text/html":
                soup = BeautifulSoup(part.get_payload(decode=True).decode('utf-8', errors='ignore'), 'html.parser')
                text += soup.get_text()
    else:
        text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

    # Очищаем текст
    text = re.sub(r'http\S+', '', text)  # Удаляем URL
    text = re.sub(r'\S+@\S+', '', text)  # Удаляем email адреса
    text = re.sub(r'[^\w\s]', '', text)  # Удаляем специальные символы
    text = re.sub(r'\s+', ' ', text).strip()  # Удаляем лишние пробелы

    return text


def extract_text_from_email(msg):
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            elif part.get_content_type() == "text/html":
                text += extract_text_from_html(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
    else:
        text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    return text


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()


def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Zа-яА-Я\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def process_dataset(raw_data_dir, processed_data_dir):
    for category in os.listdir(raw_data_dir):
        category_path = os.path.join(raw_data_dir, category)
        if os.path.isdir(category_path):
            processed_category_path = os.path.join(processed_data_dir, category)
            os.makedirs(processed_category_path, exist_ok=True)

            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                try:
                    processed_text = process_email(file_path)
                    processed_file_path = os.path.join(processed_category_path, f"{os.path.splitext(filename)[0]}.txt")
                    with open(processed_file_path, 'w', encoding='utf-8') as f:
                        f.write(processed_text)
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")


def move_email(email_id, category):
    source_path = os.path.join('data/unsorted', email_id)
    dest_path = os.path.join('data/sorted', category, email_id)

    if not os.path.exists(source_path):
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(source_path, dest_path)
    return True