import pandas as pd
from ctgan import CTGAN
import joblib

# Загрузка данных
data = pd.read_csv('data/original_data.csv')

# Инициализация модели CTGAN
ctgan = CTGAN(epochs=300, batch_size=500)

# Обучение модели
ctgan.fit(data)

# Сохранение обученной модели
joblib.dump(ctgan, 'model/ctgan_model.pkl')
