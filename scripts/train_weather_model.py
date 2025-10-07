import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle

# Папки
enriched_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'enriched')
models_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
forecasts_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'forecasts')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(forecasts_dir, exist_ok=True)

# Функция для чтения и объединения всех enriched файлов (используя collection_time как as_of_date)
def load_and_merge_enriched_data():
    all_files = [f for f in os.listdir(enriched_dir) if f.endswith('.csv')]
    df_list = []
    
    for file in all_files:
        file_path = os.path.join(enriched_dir, file)
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Использовать collection_time как as_of_date (предполагаем, что оно в формате datetime или строке)
        if 'collection_time' in df.columns:
            df['as_of_date'] = pd.to_datetime(df['collection_time'])  # Преобразовать в datetime
        else:
            print(f"Поле 'collection_time' отсутствует в {file}. Пропускаем файл.")
            continue
        
        df_list.append(df)
    
    # Объединить все данные
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values('as_of_date').drop_duplicates()  # Сортировка и удаление дубликатов
    return df_all

# Функция для подготовки данных с таргетом на завтра
def prepare_data_for_forecast(df):
    # Сортировка по дате и городу (предполагаем, что есть столбец 'city_name')
    df = df.sort_values(['city_name', 'as_of_date'])
    
    # Создать таргет: comfort_index на следующий день
    df['next_day_comfort_index'] = df.groupby('city_name')['comfort_index'].shift(-1)
    df = df.dropna(subset=['next_day_comfort_index'])  # Удалить строки без таргета
    
    # Признаки (адаптируй под твои столбцы)
    features = ['temperature', 'humidity', 'wind_speed', 'precipitation']  # Примеры признаков
    target = 'next_day_comfort_index'
    
    # Фильтр: только строки с полными данными
    df = df.dropna(subset=features + [target])
    
    return df[features], df[target], df

# Основная функция обучения и прогноза
def train_and_forecast():
    # Шаг 1: Загрузить и объединить данные
    df_all = load_and_merge_enriched_data()
    if df_all.empty:
        print("Нет данных в enriched_dir. Проверь папку и поле 'collection_time'.")
        return
    
    # Шаг 2: Подготовить данные
    X, y, df = prepare_data_for_forecast(df_all)
    if X.empty or y.empty:
        print("Недостаточно данных для обучения.")
        return
    
    # Шаг 3: Разделить на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Шаг 4: Обучить модель (линейная регрессия для примера)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE на тесте: {mae:.2f}")
    
    # Шаг 5: Сохранить модель
    model_path = os.path.join(models_dir, 'weather_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Модель сохранена в {model_path}")
    
    # Шаг 6: Прогноз на завтра
    # Использовать последние данные (максимальный as_of_date)
    latest_data = df[df['as_of_date'] == df['as_of_date'].max()]
    if latest_data.empty:
        print("Нет последних данных для прогноза.")
        return
    
    # Признаки для прогноза (последние значения)
    latest_features = latest_data[features].iloc[0:1]  # Берем первую строку (предполагаем один город или усредни)
    forecast_comfort_index = model.predict(latest_features)[0]
    
    # Дата завтра
    tomorrow = (df['as_of_date'].max() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Сохранить прогноз
    forecast_df = pd.DataFrame({
        'forecast_date': [tomorrow],
        'predicted_comfort_index': [forecast_comfort_index],
        'based_on_date': [df['as_of_date'].max().strftime('%Y-%m-%d %H:%M:%S')]  # Изменено на YYYY-MM-DD hh:mm:ss
    })
    forecast_path = os.path.join(forecasts_dir, 'forecast_tomorrow.csv')
    forecast_df.to_csv(forecast_path, index=False, encoding='utf-8')
    print(f"Прогноз на завтра ({tomorrow}): comfort_index = {forecast_comfort_index:.2f}")
    print(f"Прогноз сохранён в {forecast_path}")

# Запуск
if __name__ == "__main__":
    train_and_forecast()
