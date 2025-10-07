import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression  # Импорт из scikit-learn (sklearn deprecated, но работает)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  # Для визуализаций
import pickle  # Уже есть, но для ясности

# Папки
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
enriched_dir = os.path.join(data_dir, 'enriched')
models_dir = os.path.join(data_dir, 'models')
forecasts_dir = os.path.join(models_dir, 'forecast')
visualizations_dir = os.path.join(data_dir, 'visualizations')
os.makedirs(forecasts_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для загрузки данных
def load_data_from_directory(directory):
    all_data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            # Извлекаем город из имени файла
            city_name = file.split('_')[0]
            df['city'] = city_name
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return pd.DataFrame()

# Функция для вычисления comfort index (пример: простой расчёт на основе температуры и влажности)
def calculate_comfort_index(temp, humidity):
    # Пример формулы: comfort_index = temp - (humidity / 10)  # Упрощённая формула, адаптируйте по необходимости
    return temp - (humidity / 10)

# Функция для обучения модели и прогноза
def train_and_forecast(df, city, tomorrow_date):
    if df.empty:
        print(f"Нет данных для города {city}")
        return pd.DataFrame()
    
    df_city = df[df['city'] == city]
    if df_city.empty:
        print(f"Нет данных для города {city}")
        return pd.DataFrame()
    
    # Подготовка данных (пример для температуры)
    df_city['date'] = pd.to_datetime(df_city['date'])
    df_city = df_city.sort_values('date')
    df_city['day_of_year'] = df_city['date'].dt.dayofyear
    X = df_city[['day_of_year']]
    y = df_city['temperature']
    
    if len(y) < 2:
        print(f"Недостаточно данных для модели в городе {city}")
        return pd.DataFrame()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Прогноз на завтра
    tomorrow_day = pd.to_datetime(tomorrow_date).dayofyear
    predicted_temp = model.predict([[tomorrow_day]])[0]
    
    # Сохранение модели
    model_path = os.path.join(models_dir, f'{city}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Возврат прогноза как DataFrame
    forecast_df = pd.DataFrame({
        'city': [city],
        'forecast_date': [tomorrow_date],
        'predicted_temp': [predicted_temp],
        'model_type': ['LinearRegression']
    })
    return forecast_df

# Функция для создания визуализаций
def create_visualizations(df):
    if df.empty:
        print("Нет данных для визуализаций.")
        return
    
    # Визуализация температуры по городам
    plt.figure(figsize=(10, 6))
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        plt.plot(city_data['date'], city_data['temperature'], label=city)
    plt.title('Temperature Over Time by City')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig(os.path.join(visualizations_dir, 'temperature.png'))
    plt.close()
    
    # Визуализация comfort index (если есть humidity)
    if 'humidity' in df.columns:
        df['comfort_index'] = df.apply(lambda row: calculate_comfort_index(row['temperature'], row['humidity']), axis=1)
        plt.figure(figsize=(10, 6))
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            plt.plot(city_data['date'], city_data['comfort_index'], label=city)
        plt.title('Comfort Index Over Time by City')
        plt.xlabel('Date')
        plt.ylabel('Comfort Index')
        plt.legend()
        plt.savefig(os.path.join(visualizations_dir, 'comfort_index.png'))
        plt.close()
    else:
        print("Колонка 'humidity' отсутствует, пропускаем comfort_index.")
    
    print("Визуализации сохранены в data/visualizations/")

# Основная функция
def main():
    df = load_data_from_directory(enriched_dir)
    if df.empty:
        print("Нет данных для обработки.")
        return
    
    tomorrow = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    as_of_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_forecasts = []
    cities = df['city'].unique()
    for city in cities:
        forecast_df = train_and_forecast(df, city, tomorrow)
        if not forecast_df.empty:
            forecast_df['as_of_date'] = as_of_date
            all_forecasts.append(forecast_df)
    
    if all_forecasts:
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)
        forecast_file = os.path.join(forecasts_dir, 'Forecast.csv')
        file_exists = os.path.exists(forecast_file)
        combined_forecast.to_csv(forecast_file, mode='a', header=not file_exists, index=False)
        print(f"Прогнозы сохранены в {forecast_file} с as_of_date {as_of_date}")
    else:
        print("Нет прогнозов для сохранения.")
    
    # Создание визуализаций на основе всех данных
    create_visualizations(df)

if __name__ == "__main__":
    main()
