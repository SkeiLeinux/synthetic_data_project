import pandas as pd
import numpy as np

def generate_synthetic_data(df):
    synthetic_df = df.copy()

    # Простая заглушка: случайное перемешивание и добавление минимального шума
    synthetic_df = synthetic_df.sample(frac=1).reset_index(drop=True)

    numeric_cols = synthetic_df.select_dtypes(include=np.number).columns
    synthetic_df[numeric_cols] += np.random.normal(0, 0.01, synthetic_df[numeric_cols].shape)

    return synthetic_df

