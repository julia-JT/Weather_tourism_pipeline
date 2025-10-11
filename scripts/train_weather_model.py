import pandas as pd
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess  # Добавлено для выполнения git команд

# Папки (без изменений)
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
enriched_dir = os.path.join(data_dir, 'enriched')
models_dir = os.path.join(data_dir, 'models')
forecasts_dir = os.path.join(models_dir, 'forecast')
visualizations_dir = os.path.join(data_dir, 'visualizations')
os.makedirs(forecasts_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Функция для загрузки данных (без изменений)
def load_data_from_directory(directory):
    all_data = []
    for file in os.listdir(directory):
        if file.startswith('weather_enriched_') and file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            if 'city_name' in df.columns:
                df['city'] = df['city_name']
            else:
                print(f"Предупреждение: В файле {file} нет колонки 'city_name'. Пропускаем файл.")
                continue
            df.rename(columns={'collection_time': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S', errors='coerce')
            all_data.append(df)
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['hour'] = combined_df['date'].dt.hour
        combined_df['interval'] = combined_df['hour'].apply(lambda h: 'day' if 11 <= h < 18 else 'night')
        combined_df['date_day'] = combined_df['date'].dt.date
        combined_df = combined_df.groupby(['city', 'date_day', 'interval'])['temperature'].mean().reset_index()
        combined_df = combined_df.pivot(index=['city', 'date_day'], columns='interval', values='temperature').reset_index()
        combined_df.rename(columns={'day': 'temp_day', 'night': 'temp_night'}, inplace=True)
        combined_df['date'] = pd.to_datetime(combined_df['date_day'])
        combined_df.drop('date_day', axis=1, inplace=True)
        combined_df['temp_day'] = combined_df['temp_day'].fillna(combined_df['temp_day'].mean())
        combined_df['temp_night'] = combined_df['temp_night'].fillna(combined_df['temp_night'].mean())
        return combined_df
    return pd.DataFrame()

# Функция для обучения модели и прогноза (без изменений)
def train_and_forecast(df, city, tomorrow_date):
    if df.empty:
        print(f"Нет данных для города {city}")
        return pd.DataFrame()
    
    df_city = df[df['city'] == city]
    if df_city.empty:
        print(f"Нет данных для города {city}")
        return pd.DataFrame()
    
    if 'date' not in df_city.columns or 'temp_day' not in df_city.columns or 'temp_night' not in df_city.columns:
        print(f"Необходимые колонки отсутствуют в данных для города {city}")
        return pd.DataFrame()
    
    df_city = df_city.copy()
    df_city['date'] = pd.to_datetime(df_city['date'])
    df_city = df_city.sort_values('date')
    df_city['day_of_year'] = df_city['date'].dt.dayofyear
    X = df_city[['day_of_year']]
    
    if len(df_city) < 2:
        print(f"Недостаточно данных для модели в городе {city}")
        return pd.DataFrame()
    
    y_day = df_city['temp_day']
    X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(X, y_day, test_size=0.2, random_state=42)
    model_day = LinearRegression()
    model_day.fit(X_train_day, y_train_day)
    
    y_night = df_city['temp_night']
    X_train_night, X_test_night, y_train_night, y_test_night = train_test_split(X, y_night, test_size=0.2, random_state=42)
    model_night = LinearRegression()
    model_night.fit(X_train_night, y_train_night)
    
    tomorrow_day = pd.to_datetime(tomorrow_date).dayofyear
    predicted_day = model_day.predict(pd.DataFrame([[tomorrow_day]], columns=['day_of_year']))[0]
    predicted_night = model_night.predict(pd.DataFrame([[tomorrow_day]], columns=['day_of_year']))[0]
    
    try:
        model_path_day = os.path.join(models_dir, f'{city}_model_day.pkl')
        model_path_night = os.path.join(models_dir, f'{city}_model_night.pkl')
        with open(model_path_day, 'wb') as f:
            pickle.dump(model_day, f)
        with open(model_path_night, 'wb') as f:
            pickle.dump(model_night, f)
        print(f"Модели для {city} сохранены в {models_dir}")
    except Exception as e:
        print(f"Ошибка сохранения моделей для {city}: {e}")
    
    forecast_df = pd.DataFrame({
        'city': [city],
        'forecast_date': [tomorrow_date],
        'predicted_temp_day': [round(predicted_day)],
        'predicted_temp_night': [round(predicted_night)],
        'model_type': ['LinearRegression']
    })
    return forecast_df

# Функция для создания динамических визуализаций (изменено: формат дат DD.MM.YY на оси X для всех графиков)
def create_dynamic_visualizations(df, forecast_df):
    if df.empty:
        print("Нет данных для визуализаций.")
        return
    
    if 'date' not in df.columns:
        print("Колонка 'date' отсутствует для визуализаций.")
        return
    
    print(f"Уникальные даты в enriched данных: {sorted(df['date'].unique())}")
    print(f"Диапазон дат: от {df['date'].min()} до {df['date'].max()}")
    if not forecast_df.empty:
        print(f"Прогноз на дату: {forecast_df['forecast_date'].iloc[0]}")
    
    df_combined = df.copy()
    if not forecast_df.empty:
        forecast_rows = forecast_df[['city', 'forecast_date', 'predicted_temp_day', 'predicted_temp_night']].rename(
            columns={'forecast_date': 'date', 'predicted_temp_day': 'temp_day', 'predicted_temp_night': 'temp_night'}
        )
        forecast_rows['date'] = pd.to_datetime(forecast_rows['date'])
        df_combined = pd.concat([df_combined, forecast_rows], ignore_index=True)
    
    if 'as_of_date' in forecast_df.columns:
        df_combined['as_of_date'] = forecast_df['as_of_date'].iloc[0] if not forecast_df.empty else None
    else:
        df_combined['as_of_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    df_combined = df_combined.sort_values('date')
    
    # Список цветов для консистентности (Plotly qualitative palette)
    import plotly.colors
    colors = plotly.colors.qualitative.Plotly  # ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    fig1 = go.Figure()
    for i, city in enumerate(df_combined['city'].unique()):
        color = colors[i % len(colors)]
        city_data = df_combined[(df_combined['city'] == city) & (df_combined['date'] < pd.to_datetime(datetime.now().date()))]
        fig1.add_trace(go.Scatter(x=city_data['date'], y=city_data['temp_day'], mode='lines+markers', name=f"{city} - Historical Day", line=dict(color=color)))
    fig1.update_layout(
        title="Historical Day Temperature Over Time by City",
        xaxis_title="Date",
        yaxis_title="Day Temperature (°C)",
        xaxis=dict(tickformat='%d.%m.%y'),  # Изменено: формат дат DD.MM.YY
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(label=city, method="update", args=[{"visible": [c == city for c in df_combined['city'].unique()]}, {"title": f"Historical Day Temperature - {city}"}])
                    for city in df_combined['city'].unique()
                ]
            )
        ]
    )
    fig1.write_image(os.path.join(visualizations_dir, 'historical_day_temperature.png'))  # Изменено на PNG
    
    fig2 = go.Figure()
    for i, city in enumerate(df_combined['city'].unique()):
        color = colors[i % len(colors)]
        city_data = df_combined[(df_combined['city'] == city) & (df_combined['date'] < pd.to_datetime(datetime.now().date()))]
        fig2.add_trace(go.Scatter(x=city_data['date'], y=city_data['temp_night'], mode='lines+markers', name=f"{city} - Historical Night", line=dict(color=color)))
    fig2.update_layout(
        title="Historical Night Temperature Over Time by City",
        xaxis_title="Date",
        yaxis_title="Night Temperature (°C)",
        xaxis=dict(tickformat='%d.%m.%y'),  # Изменено: формат дат DD.MM.YY
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(label=city, method="update", args=[{"visible": [c == city for c in df_combined['city'].unique()]}, {"title": f"Historical Night Temperature - {city}"}])
                    for city in df_combined['city'].unique()
                ]
            )
        ]
    )
    fig2.write_image(os.path.join(visualizations_dir, 'historical_night_temperature.png'))  # Изменено на PNG
    
    fig3 = go.Figure()
    for i, city in enumerate(df_combined['city'].unique()):
        color = colors[i % len(colors)]
        historical = df_combined[(df_combined['city'] == city) & (df_combined['date'] < pd.to_datetime(datetime.now().date()))]
        forecast = df_combined[(df_combined['city'] == city) & (df_combined['date'] >= pd.to_datetime(datetime.now().date()))]
        fig3.add_trace(go.Scatter(x=historical['date'], y=historical['temp_day'], mode='lines+markers', name=f"{city} - Historical Day", line=dict(color=color)))
        fig3.add_trace(go.Scatter(x=forecast['date'], y=forecast['temp_day'], mode='markers', marker=dict(color=color, size=10), name=f"{city} - Forecast Day"))
    fig3.update_layout(
        title="Forecasted Day Temperature Over Time by City (with as_of_date)",
        xaxis_title="Date",
        yaxis_title="Day Temperature (°C)",
        xaxis=dict(tickformat='%d.%m.%y'),  # Изменено: формат дат DD.MM.YY
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(label=city, method="update", args=[{"visible": [c == city for c in df_combined['city'].unique()] * 2}, {"title": f"Forecasted Day Temperature - {city}"}])
                    for city in df_combined['city'].unique()
                ]
            )
        ]
    )
    fig3.write_image(os.path.join(visualizations_dir, 'forecasted_day_temperature.png'))  # Изменено на PNG
    
    fig4 = go.Figure()
    for i, city in enumerate(df_combined['city'].unique()):
        color = colors[i % len(colors)]
        historical = df_combined[(df_combined['city'] == city) & (df_combined['date'] < pd.to_datetime(datetime.now().date()))]
        forecast = df_combined[(df_combined['city'] == city) & (df_combined['date'] >= pd.to_datetime(datetime.now().date()))]
        fig4.add_trace(go.Scatter(x=historical['date'], y=historical['temp_night'], mode='lines+markers', name=f"{city} - Historical Night", line=dict(color=color)))
        fig4.add_trace(go.Scatter(x=forecast['date'], y=forecast['temp_night'], mode='markers', marker=dict(color=color, size=10), name=f"{city} - Forecast Night"))
    fig4.update_layout(
        title="Forecasted Night Temperature Over Time by City (with as_of_date)",
        xaxis_title="Date",
        yaxis_title="Night Temperature (°C)",
        xaxis=dict(tickformat='%d.%m.%y'),  # Изменено: формат дат DD.MM.YY
        updatemenus=[
            dict(
                type="dropdown",
                buttons=[
                    dict(label=city, method="update", args=[{"visible": [c == city for c in df_combined['city'].unique()] * 2}, {"title": f"Forecasted Night Temperature - {city}"}])
                    for city in df_combined['city'].unique()
                ]
            )
        ]
    )
    fig4.write_image(os.path.join(visualizations_dir, 'forecasted_night_temperature.png'))  # Изменено на PNG
    
    print("Динамические визуализации сохранены в data/visualizations/ как PNG-файлы (статические изображения)")

# Новая функция для коммита и пуша изменений с исправлениями
def commit_and_push_changes():
    try:
        # Сначала pull с rebase, чтобы синхронизировать с remote
        subprocess.run(['git', 'pull', '--rebase'], check=True, capture_output=True, text=True)
        print("Git pull --rebase выполнен успешно.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при git pull --rebase: {e.stderr}")
        # Если pull fails, попробуем allow-unrelated-histories (на случай первого пуша)
        try:
            subprocess.run(['git', 'pull', '--allow-unrelated-histories'], check=True, capture_output=True, text=True)
            print("Git pull --allow-unrelated-histories выполнен успешно.")
        except subprocess.CalledProcessError as e2:
            print(f"Ошибка при git pull --allow-unrelated-histories: {e2.stderr}")
            return  # Не пушим, если pull не удался
    
    try:
        # Add все изменения (modified и untracked)
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True, text=True)
        print("Git add выполнен успешно.")
        
        # Commit с сообщением
        commit_message = f"Automated update: weather data, models, and reports on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(['git', 'commit', '-m', commit_message], check=True, capture_output=True, text=True)
        print(f"Git commit выполнен успешно: {commit_message}")
        
        # Push
        subprocess.run(['git', 'push'], check=True, capture_output=True, text=True)
        print("Git push выполнен успешно.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при git операциях: {e.stderr}")

# Основная функция (добавлен вызов commit_and_push_changes)
def main():
    print(f"Script started. Data dir: {data_dir}")
    print(f"Enriched dir: {enriched_dir}")
    print(f"Files in enriched dir: {os.listdir(enriched_dir) if os.path.exists(enriched_dir) else 'Enriched dir not found'}")
    
    df = load_data_from_directory(enriched_dir)
    if df.empty:
        print("Нет данных для обработки.")
        return
    
    print(f"Loaded data shape: {df.shape}")
    
    tomorrow = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    as_of_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    all_forecasts = []
    cities = df['city'].unique()
    print(f"Cities to process: {cities}")
    for city in cities:
        forecast_df = train_and_forecast(df, city, tomorrow)
        if not forecast_df.empty:
            forecast_df['as_of_date'] = as_of_date
            all_forecasts.append(forecast_df)
    
    combined_forecast = pd.DataFrame()
    if all_forecasts:
        combined_forecast = pd.concat(all_forecasts, ignore_index=True)
        forecast_file = os.path.join(forecasts_dir, 'Forecast.csv')
        try:
            file_exists = os.path.exists(forecast_file)
            combined_forecast.to_csv(forecast_file, mode='w', header=True, index=False)
            print(f"Прогнозы сохранены в {forecast_file} с as_of_date {as_of_date}")
        except Exception as e:
            print(f"Ошибка сохранения прогнозов: {e}")
    else:
        print("Нет прогнозов для сохранения.")
    
    create_dynamic_visualizations(df, combined_forecast)
    
    # Коммит и пуш изменений
    commit_and_push_changes()

if __name__ == "__main__":
    main()
