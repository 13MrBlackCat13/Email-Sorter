import os
import sys
import email
import chardet
import re
from bs4 import BeautifulSoup

def get_current_path():
    """Получает путь к текущему скрипту."""
    return os.path.dirname(os.path.realpath(__file__))

def get_email_body(email_content):
    """Извлекает текстовое тело письма из его содержимого."""
    msg = email.message_from_bytes(email_content)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            payload = part.get_payload(decode=True)
            if content_type == "text/plain":
                encoding = chardet.detect(payload)['encoding']
                if encoding:
                    body = payload.decode(encoding, errors="ignore")
                else:
                    body = payload.decode("utf-8", errors="ignore")
                break
            elif content_type == "text/html":
                soup = BeautifulSoup(payload, "html.parser")
                body = soup.get_text()
                break
    else:
        encoding = chardet.detect(payload)['encoding']
        if encoding:
            body = payload.decode(encoding, errors="ignore")
        else:
            body = payload.decode("utf-8", errors="ignore")
    return body

def get_email_subject_from_filename(filename):
    """Извлекает тему письма из названия файла."""
    # Разделяем имя файла на составные части
    parts = filename.split(" - ")

    # Извлекаем тему письма (первая часть) и удаляем дату и левые цифры
    subject = parts[0].strip()
    return subject

def remove_empty_lines(text):
    """Удаляет лишние пустые строки (энтеры) между строками текста."""
    return re.sub(r'\n\s*\n', '\n\n', text)

def main():
    # Получаем путь к папке с письмами
    emails_folder = os.path.join(get_current_path(), "emails")

    # Создаем папку text
    text_folder = os.path.join(os.path.dirname(emails_folder), "text")
    if not os.path.exists(text_folder):
        os.mkdir(text_folder)

    # Итерируем по папкам с письмами
    for category in os.listdir(emails_folder):
        # Получаем путь к папке категории
        category_folder = os.path.join(emails_folder, category)

        # Подсчет номера для файла в категории
        file_number = 1

        # Итерируем по письмам в папке категории
        for email_filename in os.listdir(category_folder):
            # Получаем путь к файлу с письмом
            email_file = os.path.join(category_folder, email_filename)

            # Открываем файл письма в двоичном режиме
            with open(email_file, "rb") as f:
                email_content = f.read()

            # Получаем текстовое тело письма
            body = get_email_body(email_content)

            # Удаляем лишние пустые строки (энтеры) между строками текста
            body = remove_empty_lines(body)

            # Получаем тему письма из названия файла
            subject = get_email_subject_from_filename(email_filename)

            # Проверяем наличие заголовка From
            from_header = b"From: "
            if from_header in email_content:
                sender = email_content.split(from_header)[1]
                sender = sender.split(b"\n")[0]
                sender = sender.decode("utf-8")
            else:
                sender = ""

            # Создаем текстовый файл для письма с учетом категории и номера
            text_subfolder = os.path.join(text_folder, category)
            if not os.path.exists(text_subfolder):
                os.makedirs(text_subfolder)

            # Изменяем расширение файла на .txt и добавляем номер
            txt_filename = f"{subject} - {file_number}.txt"
            text_file = os.path.join(text_subfolder, txt_filename)
            with open(text_file, "w", encoding="utf-8") as f:
                f.write("From: " + sender + "\nSubject: " + subject + "\n\n" + body)

            # Увеличиваем номер для следующего файла
            file_number += 1

    print("Done!")

if __name__ == "__main__":
    main()