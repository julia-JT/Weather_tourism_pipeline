import pandas as pd
import matplotlib.pyplot as plt
import os

models_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
os.makedirs(models_dir, exist_ok=True)

# Загрузка aggregated (объединяем файлы)
aggregated_files = [f'data/aggregated/{f}' for f in os.listdir('data/aggregated') if f.endswith('.csv')]
aggregated = pd.concat([pd.read_csv(f) for f in aggregated_files], ignore_index=True)

forecasts = pd.read_csv('data/models/forecasts.csv')

# Вопрос 1: В каком городе самый высокий tourism_rating? (с ссылкой)
best_city = aggregated.loc[aggregated['tourism_rating'].idxmax()]
plt.figure(figsize=(10, 6))
plt.bar(aggregated['city'], aggregated['tourism_rating'])
plt.title(f'Рейтинг туризма по городам\nЛучший: {best_city["city"]} (Ссылка: {best_city["address_link"]})')
plt.savefig(os.path.join(models_dir, 'tourism_rating_cities.png'))

# Вопрос 2: Какие федеральные округа наиболее привлекательны? (с ссылкой)
region_avg = aggregated.groupby('region')['tourism_rating'].mean()
best_region = region_avg.idxmax()
region_links = aggregated.groupby('region')['address_link'].first()  # Первая ссылка по региону
plt.figure(figsize=(10, 6))
region_avg.plot(kind='bar')
plt.title(f'Средний рейтинг туризма по округам\nЛучший: {best_region} (Ссылка: {region_links[best_region]})')
plt.savefig(os.path.join(models_dir, 'tourism_rating_regions.png'))

# Вопрос 3: Рекомендации турагентствам (прогнозы с ссылками)
plt.figure(figsize=(10, 6))
plt.scatter(forecasts['forecast_tourism_rating'], forecasts['city'], s=forecasts['forecast_tourism_rating']*10)
plt.title('Прогноз рейтинга туризма по городам\nРекомендация: Продвигать города с рейтингом >0.8')
# Добавление ссылок в аннотации (для топ-городов)
top_forecasts = forecasts.nlargest(5, 'forecast_tourism_rating')
for _, row in top_forecasts.iterrows():
    link = aggregated.loc[aggregated['city'] == row['city'], 'address_link'].values[0]
    plt.annotate(f'Ссылка: {link}', (row['forecast_tourism_rating'], row['city']), fontsize=8)
plt.savefig(os.path.join(models_dir, 'forecast_recommendations.png'))

# Дополнительная визуализация: Влияние погоды на рейтинг (используя данные файлов)
plt.figure(figsize=(10, 6))
plt.scatter(aggregated['temp_mean'], aggregated['tourism_rating'], c=aggregated['rain_mean'], cmap='viridis')
plt.colorbar(label='Средние осадки (rain_mean)')
plt.xlabel('Средняя температура (°C)')
plt.ylabel('Рейтинг туризма')
plt.title('Влияние температуры и осадков на рейтинг туризма\nДанные из aggregated файлов')
plt.savefig(os.path.join(models_dir, 'weather_impact.png'))
plt.ylabel('Рейтинг туризма')
plt.title('Влияние температуры и осадков на рейтинг туризма\nДанные из aggregated файлов')
plt.savefig(os.path.join(models_dir, 'weather_impact.png'))
