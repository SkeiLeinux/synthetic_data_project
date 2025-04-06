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

    def compare_statistics(self, synthetic_df, tolerances):
        orig_stats = self.df.describe(include='all')
        synth_stats = synthetic_df.describe(include='all')

        violations = {}

        for column, rules in tolerances.items():
            if column in orig_stats.columns:
                dtype = self.df[column].dtype

                # Проверка числовых данных (int, float)
                if np.issubdtype(dtype, np.number):
                    for metric in ['mean', '50%', 'min', 'max']:
                        orig_val = orig_stats.loc[metric, column]
                        synth_val = synth_stats.loc[metric, column]
                        abs_tol = rules.get('absolute', None)
                        perc_tol = rules.get('percent', None)

                        actual_diff = abs(orig_val - synth_val)

                        if abs_tol is not None and actual_diff > abs_tol:
                            violations[(column, metric)] = f'Превышено абсолютное отклонение: {actual_diff} > {abs_tol}'

                        if perc_tol is not None:
                            allowed_diff = orig_val * perc_tol / 100
                            if actual_diff > allowed_diff:
                                violations[(column,
                                            metric)] = f'Превышено процентное отклонение: {actual_diff:.2f} > {allowed_diff:.2f}'

                # Проверка данных типа date и timestamp
                elif np.issubdtype(dtype, np.datetime64):
                    for metric in ['50%', 'min', 'max']:
                        orig_date = pd.to_datetime(orig_stats.loc[metric, column])
                        synth_date = pd.to_datetime(synth_stats.loc[metric, column])
                        days_tol = rules.get('days', None)
                        minutes_tol = rules.get('minutes', None)

                        diff = abs(orig_date - synth_date)

                        if days_tol is not None and diff.days > days_tol:
                            violations[
                                (column, metric)] = f'Превышено отклонение в днях: {diff.days} дней > {days_tol} дней'

                        if minutes_tol is not None and diff.total_seconds() / 60 > minutes_tol:
                            violations[(column,
                                        metric)] = f'Превышено отклонение в минутах: {diff.total_seconds() / 60:.1f} мин. > {minutes_tol} мин.'

        return len(violations) == 0, violations
