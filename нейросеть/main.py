import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdmetrics.reports.single_table import QualityReport
# import sdmetrics
from scipy.stats import entropy

# 1. загрузка и базовая подготовка
df = pd.read_csv('adult.csv')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

# 2. чувствительный атрибут
sensitive = 'occupation'

# 3. генерализация квазиидентификаторов с приведением к str
def generalize_qi(df):
    # преобразуем age и education-num в числовой формат на всякий случай
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['education-num'] = pd.to_numeric(df['education-num'], errors='coerce')

    bins = [0, 30, 60, 100]
    labels = ['<=30', '31-60', '61+']
    # age → age_bin (astype(str) убирает category)
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)

    # education-num → edu_bin
    df['edu_bin'] = df['education-num']\
                       .apply(lambda x: 'low' if x<=10 else 'high')\
                       .astype(str)
    # marital-status → marital_bin
    df['marital_bin'] = df['marital-status']\
                       .apply(lambda x: 'married' if 'Married' in x else 'not-married')\
                       .astype(str)
    # race → race_bin
    df['race_bin'] = df['race']\
                       .apply(lambda x: x if x=='White' else 'Non-White')\
                       .astype(str)

    return df

df = generalize_qi(df)
quasi_ids = ['age_bin','edu_bin','marital_bin','race_bin','sex']
# quasi_ids = ['age_bin','sex']

# 4. разделение на train/test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[sensitive]
)

# 5. добавляем 'id'
train_df = train_df.reset_index(drop=True)
train_df['id'] = train_df.index

# 6. метаданные
metadata = Metadata.detect_from_dataframe(data=train_df, table_name='adult')
metadata.set_primary_key(table_name='adult', column_name='id')
metadata.save_to_json(filepath='metadata.json')

# 7.1. словарь доступных генераторов:
generators = {
    "ctgan": CTGANSynthesizer,
    "tvae": TVAESynthesizer,
    "copulagan": CopulaGANSynthesizer,
}

# 7.2. выбираем нужный синтезатор:
chosen_model = "copulagan"  # допустимые значения: "ctgan", "tvae", "copulagan"

# 7.3. обучаем модель
model = generators[chosen_model](
    metadata,
    epochs=400,             # количество эпох обучения, больше 1000 не рекомендуется
    batch_size=8000,        # количество "строк" за 1 шаг, больше 16000 не надо
    # generator_lr=2e-4,      # скорость обучения генератора
    # discriminator_lr=2e-4,  # скорость обучения дискриминатора
    verbose=True,           # лог в терминал
    cuda=True               # вычисления на GPU
)
model.fit(train_df)

# 8. Генерируем синтетику
syn_df = model.sample(num_rows=len(train_df)).drop(columns=['id'])
# И сразу же биннинг + astype(str)
syn_df = generalize_qi(syn_df)
syn_df.to_csv('synth.csv')

# 9. Метрики
def k_anonymity(df):
    return int(df.groupby(quasi_ids).size().min())

def l_diversity(df):
    return int(df.groupby(quasi_ids)[sensitive].nunique().min())

def t_closeness(orig, syn):
    global_dist = orig[sensitive].value_counts(normalize=True)
    def emd(group):
        p = group[sensitive].value_counts(normalize=True)\
                 .reindex(global_dist.index, fill_value=0)
        return np.abs(p - global_dist).sum() / 2
    return float(syn.groupby(quasi_ids, group_keys=False).apply(emd).max())

k_syn = k_anonymity(syn_df)
l_syn = l_diversity(syn_df)
t_syn = t_closeness(train_df, syn_df)

print(f"k‑анонимность = {k_syn} (≥10?)")
print(f"l‑разнообразие = {l_syn} (≥3?)")
print(f"t‑близость   = {t_syn:.4f} (<0.3?)")

# 10.0. Оценка «statistical fidelity» — насколько распределения и корреляции синтетики совпадают с оригиналом:
#    (drop columns=['id'], поскольку синтетический DF мы уже сохранили без id)
real_no_id = train_df.drop(columns=['id'])
syn_no_id  = syn_df.copy()

# Создаём отчёт
report = QualityReport()

# Обучаем его на реальных и синтетических данных
report.generate(
    real_data=train_df,
    synthetic_data=pd.concat([syn_df, train_df[['id']].iloc[:len(syn_df)].reset_index(drop=True)], axis=1),
    metadata=metadata.to_dict()['tables']['adult']
)

# Получаем общие результаты (в том числе распределения и зависимости)
overall_score = report.get_score()

# Общая оценка
print(f"\n=== Общая оценка качества синтетики ===")
print(f"Итоговая оценка: {overall_score:.2%}")

# Подробности по Column Shapes
print("\n— Column Shapes —")
print(report.get_details(property_name='Column Shapes'))

# Подробности по Column Pair Trends
print("\n— Column Pair Trends —")
print(report.get_details(property_name='Column Pair Trends'))

""""
# 10.1. Сравнение распределений (ColumnShapes)
shapes_score = ColumnShapes.compute(
    real_data=real_no_id,
    synthetic_data=syn_no_id,
    metadata=metadata
)
print("Column Shapes (средняя разность распределений):", shapes_score.mean())

# 10.2. Сравнение корреляций (ColumnPairTrends)
trends_score = ColumnPairTrends.compute(
    real_data=real_no_id,
    synthetic_data=syn_no_id,
    metadata=metadata
)
print("Column Pair Trends (средняя разность корреляций):", trends_score.mean())
"""