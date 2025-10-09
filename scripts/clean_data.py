import json
import os
import pandas as pd
from datetime import datetime, timedelta

# Папки (относительные пути от scripts/ к data/)
raw_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'openweather_api')
cleaned_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cleaned')
log_dir = cleaned_dir  # Лог в той же папке, что и cleaned

# Создаем папки, если не существуют
os.makedirs(cleaned_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Стандартизация городов на русский
city_mapping = {
    "Moscow": "Москва",
    "Saint Petersburg": "Санкт-Петербург",
    "Sochi": "Сочи",
    "Kazan": "Казань",
    "Novosibirsk": "Новосибирск"
}

# Функция для преобразования давления из hPa в мм.рт.ст.
def hpa_to_mmhg(hpa):
    return round(hpa * 0.750062)

# Функция для обработки данных из JSON (текущая погода: один объект, не список)
def process_json_file(data):
    records = []
    collection_time = data.get('timestamp')  # Время загрузки в ISO формате
    
    if not collection_time:
        print("ERROR: Отсутствует timestamp в данных")
        return records
    
    try:
        city = city_mapping.get(data['city'], data['city'])
        temp = round(float(data['main']['temp']))
        feels_like = round(float(data['main']['feels_like']))
        humidity = data['main']['humidity']
        pressure = hpa_to_mmhg(data['main']['pressure'])
        wind_speed = data['wind']['speed']
        weather_description = data['weather'][0]['description']
        
        # Новые поля
        visibility = data.get('visibility', None)
        pop = None  # Для текущей погоды pop нет, но оставлено для совместимости
        clouds = data['clouds']['all'] if 'clouds' in data else None
        temp_min = round(float(data['main']['temp_min'])) if 'temp_min' in data['main'] else None
        temp_max = round(float(data['main']['temp_max'])) if 'temp_max' in data['main'] else None
        
        # Проверка температуры
        if not (-50 <= temp <= 60):
            return records  # Пропускаем запись
        
        # Формат collection_time: DD.MM.YYYY hh:mm:ss
        dt_obj = datetime.fromisoformat(collection_time)
        collection_time_formatted = dt_obj.strftime("%d.%m.%Y %H:%M:%S")
        
        records.append({
            'city_name': city,
            'temperature': temp,
            'feels_like': feels_like,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'weather_description': weather_description,
            'visibility': visibility,
            'pop': pop,
            'clouds': clouds,
            'temp_min': temp_min,
            'temp_max': temp_max,
            'collection_time': collection_time_formatted,
            'timestamp': collection_time  # Для извлечения даты файла
        })
    except KeyError as e:
        print(f"ERROR: Отсутствующий ключ в данных: {e}")
    
    return records

# Основная функция (чтение JSON, очистка, преобразование, обогащение и сохранение в CSV)
def clean_weather_data():
    problems = []
    rules = [
        "Стандартизация названия города на русский язык",
        "Округление температуры, feels_like до целого числа",
        "Конвертация давления из hPa в мм.рт.ст.",
        "Фильтрация записей с температурой вне диапазона -50..+60°C",
        "Форматирование collection_time в DD.MM.YYYY hh:mm:ss",
        "Обогащение новыми полями (visibility, clouds, temp_min, temp_max)"
    ]
    
    # Определяем даты сегодня и вчера
    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    
    # Словарь для хранения записей по датам
    records_by_date = {today: [], yesterday: []}
    counts_original = {today: 0, yesterday: 0}
    
    # Найти все JSON файлы (только чтение, без изменений)
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.json'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Получаем дату из timestamp внутри JSON
                    timestamp_str = data.get('timestamp')
                    if not timestamp_str:
                        problems.append(f"Файл {file} пропущен: отсутствует timestamp")
                        continue
                    dt_obj = datetime.fromisoformat(timestamp_str)
                    file_date = dt_obj.date()
                    
                    # Фильтруем только сегодня и вчера
                    if file_date not in records_by_date:
                        continue
                    
                    counts_original[file_date] += 1  # Каждый JSON — одна запись (текущая погода)
                    records = process_json_file(data)
                    records_by_date[file_date].extend(records)
                except Exception as e:
                    problems.append(f"Ошибка чтения файла {filepath}: {e}")
    
    # Сохраняем по отдельности для каждой даты
    for dt in [yesterday, today]:
        day_records = records_by_date[dt]
        total_cleaned = len(day_records)
        total_original = counts_original[dt]
        
        if not day_records:
            print(f"Нет данных для даты {dt.strftime('%Y-%m-%d')}")
            continue
        
        # Формат даты для имени файла
        date_str = dt.strftime("%Y%m%d")
        
        # Сохранить CSV
        df = pd.DataFrame(day_records)
        csv_filename = f"weather_cleaned_{date_str}.csv"
        csv_path = os.path.join(cleaned_dir, csv_filename)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Лог
        log_filename = f"cleaning_log_{date_str}.txt"
        log_path = os.path.join(log_dir, log_filename)
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write(f"Количество исходных записей: {total_original}\n")
            log_file.write(f"Количество очищенных записей: {total_cleaned}\n")
            log_file.write("Типы примененных правил:\n")
            for rule in rules:
                log_file.write(f"- {rule}\n")
            log_file.write("Найденные проблемы:\n")
            for problem in problems:
                log_file.write(f"- {problem}\n")
        
        print(f"Очищенные данные за {dt.strftime('%Y-%m-%d')} сохранены в {csv_path}")
        print(f"Лог сохранен в {log_path}")

# Запуск
if __name__ == "__main__":
    clean_weather_data()
