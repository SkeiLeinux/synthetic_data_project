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
        """

        # Удаление полных дубликатов
        self.df.drop_duplicates(inplace=True)

        # Заполнение пропущенных значений
        for column in self.df.columns:
            if self.df[column].dtype in [np.float64, np.int64]:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            else:
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)

        return self.df

