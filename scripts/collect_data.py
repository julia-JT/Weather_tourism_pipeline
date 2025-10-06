import os
import json
import requests
from datetime import datetime

# Список городов
cities = ["Moscow", "Saint Petersburg", "Sochi", "Kazan", "Novosibirsk"]

# Путь к папке raw для хранения данных (относительно scripts/ к data/raw/openweather_api)
raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'openweather_api')
log_file_path = os.path.join(raw_dir, "data_collection.txt")

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)  # вывод в консоль (GitHub Actions лог)
    os.makedirs(raw_dir, exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry + "\n")

def collect_and_save_weather_data(cities, api_key):
    current_datetime = datetime.now()
    timestamp_str = current_datetime.strftime("%Y%m%d_%H%M")
    year = current_datetime.strftime("%Y")
    month = current_datetime.strftime("%m")
    day = current_datetime.strftime("%d")

    for city in cities:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city,
            'appid': api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                # Добавляем метаданные
                data['city'] = city
                data['timestamp'] = current_datetime.isoformat()
                data['source'] = 'openweathermap.org'

                city_safe = city.replace(" ", "_")
                dir_path = os.path.join(raw_dir, year, month, day)
                os.makedirs(dir_path, exist_ok=True)

                filename = f"weather_{city_safe}_{timestamp_str}.json"
                filepath = os.path.join(dir_path, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)

                log_message(f"SUCCESS: Данные для {city} сохранены в {filepath}")
            else:
                log_message(f"ERROR: Для {city} получен статус {response.status_code}")
        except Exception as e:
            log_message(f"EXCEPTION: Ошибка при получении данных для {city}: {e}")

if __name__ == "__main__":
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        raise ValueError("API ключ не найден! Установите переменную окружения OPENWEATHER_API_KEY")
    collect_and_save_weather_data(cities, api_key)
