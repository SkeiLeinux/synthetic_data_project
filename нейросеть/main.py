import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from scipy.stats import entropy

# 1. Загрузка и базовая подготовка
df = pd.read_csv('adult.csv')
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

# 2. Чувствительный атрибут
sensitive = 'occupation'

# 3. Генерализация QI с приведением к str
def generalize_qi(df):
    bins = [0, 30, 60, 100]
    labels = ['<=30', '31-60', '61+']
    # age → age_bin (astype(str) убирает category)
    df['age_bin'] = pd.cut(df['age'], bins=bins, labels=labels).astype(str)

    # education-num → edu_bin
    df['edu_bin']     = df['education-num']\
                           .apply(lambda x: 'low' if x<=10 else 'high')\
                           .astype(str)
    # marital-status → marital_bin
    df['marital_bin'] = df['marital-status']\
                           .apply(lambda x: 'married' if 'Married' in x else 'not-married')\
                           .astype(str)
    # race → race_bin
    df['race_bin']    = df['race']\
                           .apply(lambda x: x if x=='White' else 'Non-White')\
                           .astype(str)
    return df

df = generalize_qi(df)
quasi_ids = ['age_bin','edu_bin','marital_bin','race_bin','sex']
# quasi_ids = ['age_bin','sex']

# 4. Train/test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df[sensitive]
)

# 5. Добавляем 'id'
train_df = train_df.reset_index(drop=True)
train_df['id'] = train_df.index

# 6. Метаданные
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=train_df)
metadata.set_primary_key(column_name='id')

# 7. Обучаем CTGAN
model = CTGANSynthesizer(
    metadata,
    epochs=400,
    batch_size=8000,
    verbose=True,
    cuda=True
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
    return float(syn.groupby(quasi_ids).apply(emd).max())

k_syn = k_anonymity(syn_df)
l_syn = l_diversity(syn_df)
t_syn = t_closeness(train_df, syn_df)

print(f"k‑анонимность = {k_syn} (≥10?)")
print(f"l‑разнообразие = {l_syn} (≥3?)")
print(f"t‑близость   = {t_syn:.4f} (<0.3?)")
