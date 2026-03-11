# final_system/processor.py

from __future__ import annotations

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Предобработка табличных данных перед передачей в генератор.

    preprocess()     — очистка: дубликаты, пропуски. Не меняет список колонок.
    generalize_qi()  — добавляет bin-колонки для диагностики k/l/t-анонимности.
                       Вызывать явно после preprocess(), если нужны QI для
                       privacy_evaluator. Исходные колонки при этом сохраняются.
    drop_columns()   — удаляет колонки по списку (например, оригиналы после биннинга
                       или технические поля типа fnlwgt).
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.copy()

    def preprocess(self) -> pd.DataFrame:
        """
        Базовая очистка:
        - удаление полных дубликатов
        - заполнение пропусков (числовые → median, категориальные → mode)

        Не добавляет и не удаляет колонки — датафрейм остаётся совместимым
        с любым списком categorical_columns / continuous_columns.
        """
        self.df.drop_duplicates(inplace=True)

        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        return self.df

    def generalize_qi(self) -> pd.DataFrame:
        """
        Добавляет обобщённые (bin) колонки для квазиидентификаторов.
        Используется для диагностики k/l/t-анонимности в privacy_evaluator.

        Добавляемые колонки (если исходная колонка есть в датафрейме):
            age          → age_bin:     ['<=30', '31-60', '61+']
            education-num→ edu_bin:     ['low' (<=10), 'high' (>10)]
            marital-status→ marital_bin: ['married', 'not-married']
            race         → race_bin:    ['White', 'Non-White']

        Исходные колонки НЕ удаляются — их нужно явно передавать в CTGAN
        или убирать через drop_columns() в зависимости от задачи.
        """
        df = self.df

        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")
            df["age_bin"] = pd.cut(
                df["age"], bins=[0, 30, 60, 100], labels=["<=30", "31-60", "61+"]
            ).astype(str)

        if "education-num" in df.columns:
            df["education-num"] = pd.to_numeric(df["education-num"], errors="coerce")
            df["edu_bin"] = df["education-num"].apply(
                lambda x: "low" if x <= 10 else "high"
            ).astype(str)

        if "marital-status" in df.columns:
            df["marital_bin"] = df["marital-status"].apply(
                lambda x: "married" if "Married" in str(x) else "not-married"
            ).astype(str)

        if "race" in df.columns:
            df["race_bin"] = df["race"].apply(
                lambda x: x if x == "White" else "Non-White"
            ).astype(str)

        return self.df

    def drop_columns(self, columns: list) -> pd.DataFrame:
        """Удаляет указанные колонки (если они есть). Игнорирует отсутствующие."""
        existing = [c for c in columns if c in self.df.columns]
        self.df.drop(columns=existing, inplace=True)
        return self.df

    def basic_statistics(self) -> pd.DataFrame:
        return self.df.describe(include="all")

    def get(self) -> pd.DataFrame:
        """Возвращает текущий датафрейм."""
        return self.df
