from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer


def generate_synthetic_data(df, model_name: str = "copulagan", epochs: int = 300, batch_size: int = 500, cuda: bool = False):
    """Train a generative model and return a synthetic dataset."""
    df = df.reset_index(drop=True).copy()
    df['id'] = df.index

    metadata = Metadata.detect_from_dataframe(data=df, table_name='data')
    metadata.set_primary_key(table_name='data', column_name='id')

    generators = {
        "ctgan": CTGANSynthesizer,
        "tvae": TVAESynthesizer,
        "copulagan": CopulaGANSynthesizer,
    }

    SynthClass = generators.get(model_name.lower(), CopulaGANSynthesizer)
    model = SynthClass(metadata, epochs=epochs, batch_size=batch_size, verbose=True, cuda=cuda)
    model.fit(df)

    synthetic_df = model.sample(num_rows=len(df)).drop(columns=['id'])
    return synthetic_df
