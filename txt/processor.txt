import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def basic_statistics(self):
        """
        Считает основные статистические показатели.
        """
        stats = self.df.describe(include='all')
        return stats

    def preprocess(self):
        """
        Базовая предобработка данных:
        - Удаление дубликатов
        - Заполнение или удаление пропущенных значений
        - Конвертация типов данных
        - Биннинг квазиидентификаторов
        """

        # Удаление полных дубликатов
        self.df.drop_duplicates(inplace=True)

        # Заполнение пропущенных значений
        for column in self.df.columns:
            if self.df[column].dtype in [np.float64, np.int64]:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            else:
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)

        self.generalize_qi()

        return self.df

    def generalize_qi(self):
        """Generalize quasi-identifiers used in the neural network training."""
        df = self.df
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            bins = [0, 30, 60, 100]
            labels = ['<=30', '31-60', '61+']
            df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)

        if 'education-num' in df.columns:
            df['education-num'] = pd.to_numeric(df['education-num'], errors='coerce')
            df['edu_bin'] = df['education-num'].apply(lambda x: 'low' if x <= 10 else 'high').astype(str)

        if 'marital-status' in df.columns:
            df['marital_bin'] = df['marital-status'].apply(lambda x: 'married' if 'Married' in str(x) else 'not-married').astype(str)

        if 'race' in df.columns:
            df['race_bin'] = df['race'].apply(lambda x: x if x == 'White' else 'Non-White').astype(str)

