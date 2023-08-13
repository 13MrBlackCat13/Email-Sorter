import os
import requests

# Параметры для API
API_URL = "http://localhost:5000/api/emailcategory/eml"
SUPER_TOKEN = "devtest"
TOKEN_GROUP = "group1"  # Выберите группу токена, которую хотите использовать

# Путь к папке Unsorted
unsorted_dir = "Unsorted"

probability_threshold = 0.6

# Пройдемся по всем .eml файлам в папке Unsorted
for root, dirs, files in os.walk(unsorted_dir):
    for file in files:
        if file.endswith(".eml"):
            eml_path = os.path.join(root, file)

            # Чтение содержимого .eml файла
            with open(eml_path, "rb") as f:
                eml_content = f.read()

            # Параметры для запроса к API
            data = {
                "model_name": "all",  # Используем все модели для анализа
                "text_only": "true"
            }

            # Отправляем запрос к API
            response = requests.post(API_URL, files={"file": (file, eml_content, "message/rfc822")}, params=data)

            if response.status_code == 200:
                results = response.json()  # Получаем все результаты из списка
                avg_probabilities = results[-1]  # Берем последний элемент, который содержит средние вероятности
                category_probabilities = {k: v for k, v in avg_probabilities.items() if k != 'Model'}
                
                # Пройдемся по категориям и проверим, превышает ли средняя вероятность порог
                for category, probability in category_probabilities.items():
                    if float(probability) >= probability_threshold:
                        sorted_folder = os.path.join("Sorted", category)

                        # Если папка для категории не существует, создаем её
                        if not os.path.exists(sorted_folder):
                            os.makedirs(sorted_folder)

                        # Перемещаем .eml файл в соответствующую папку
                        new_path = os.path.join(sorted_folder, file)
                        os.rename(eml_path, new_path)
                        print(f"Moved {file} to {sorted_folder}")
                        break  # Перемещаем файл только в первую папку, которая подходит

            else:
                print(f"Error processing {file}: {response.text}")
