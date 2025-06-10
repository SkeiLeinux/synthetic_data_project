from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sklearn.model_selection import train_test_split
from processor import DataProcessor


def generate_synthetic_data(df, sensitive, model_name: str = "ctgan", epochs: int = 300, batch_size: int = 500, cuda: bool = True):
    """Train a generative model and return a synthetic dataset."""
    # разделение на train/test
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[sensitive]
    )

    # добавляем 'id'
    train_df = train_df.reset_index(drop=True)
    train_df['id'] = train_df.index

    metadata = Metadata.detect_from_dataframe(data=train_df, table_name='adult')
    metadata.set_primary_key(table_name='adult', column_name='id')
    metadata.save_to_json(filepath='metadata.json', mode = 'overwrite')

    generators = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
        "copulagan": CopulaGANSynthesizer,
    }

    model = generators[model_name](
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        cuda=cuda
        )
    model.fit(train_df)

    syn_df = model.sample(num_rows=len(train_df)).drop(columns=['id'])
    # И сразу же биннинг + astype(str)
    dp = DataProcessor(syn_df)
    syn_df = dp.preprocess()
    return syn_df
