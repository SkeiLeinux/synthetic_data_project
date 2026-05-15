from logger_config import setup_logger
from sdmetrics.reports.single_table import QualityReport
import pandas as pd
import os
import json

logger = setup_logger(__name__)


class DataValidator:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def check_k_anonymity(self, quasi_identifiers, k=3):
        # grouped = self.df.groupby(quasi_identifiers).size().reset_index(name='counts')
        # violations = grouped[grouped['counts'] < k]
        # return violations.empty, violations
        return int(self.df.groupby(quasi_identifiers).size().min())

    def check_l_diversity(self, quasi_identifiers, sensitive_attribute, l=2):
        # grouped = self.df.groupby(quasi_identifiers)[sensitive_attribute].nunique().reset_index(name='diversity')
        # violations = grouped[grouped['diversity'] < l]
        # return violations.empty, violations
        return int(self.df.groupby(quasi_identifiers)[sensitive_attribute].nunique().min())


    def check_t_closeness(self, quasi_identifiers, sensitive_attribute, original_df, t=0.9):
        global_dist = original_df[sensitive_attribute].value_counts(normalize=True)

        def emd(group):
            p = group[sensitive_attribute].value_counts(normalize=True).reindex(global_dist.index, fill_value=0)
            return abs(p - global_dist).sum() / 2

        max_emd = self.df.groupby(quasi_identifiers, group_keys=False).apply(emd).max()
        return max_emd


    def check_k_l_t(self, quasi_identifiers, sensitive_attribute, original_df, k, l, t):
        checked_k = self.check_k_anonymity(quasi_identifiers)
        checked_l = self.check_l_diversity(quasi_identifiers, sensitive_attribute)
        checked_t = self.check_t_closeness(quasi_identifiers, sensitive_attribute, original_df)

        logger.info(f"k‑анонимность = {checked_k} (≥{k}?)")
        logger.info(f"l‑разнообразие = {checked_l} (≥{l}?)")
        logger.info(f"t‑близость = {checked_t:.4f} (<{t}?)")
        if checked_k >= k and checked_l >= l and checked_t < t:
            return True
        else:
            return False

    def generate_quality_report(self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame,
                                metadata_path='metadata.json', process_id=None, data_manager=None):
        if not os.path.exists(metadata_path):
            logger.error(f"Файл {metadata_path} не найден. Невозможно сгенерировать QualityReport.")
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)['tables']['adult']

        report = QualityReport()
        real_df = real_df.reset_index(drop=True)
        real_df['id'] = real_df.index
        synthetic_df_with_id = pd.concat(
            [synthetic_df.reset_index(drop=True), real_df[['id']].iloc[:len(synthetic_df)].reset_index(drop=True)],
            axis=1
        )
        report.generate(real_data=real_df, synthetic_data=synthetic_df_with_id, metadata=metadata)

        score = report.get_score()
        details_shapes = report.get_details(property_name='Column Shapes')
        details_pairs = report.get_details(property_name='Column Pair Trends')

        logger.info(f"\n=== Общая оценка качества синтетики: {score:.2%} ===")
        logger.info("\n— Column Shapes —")
        logger.info(f"{details_shapes}")
        logger.info("\n— Column Pair Trends —")
        logger.info(f"{details_pairs}")

        if process_id and data_manager:
            metadata_value = {
                "overall_score": round(score, 4),
                "column_shapes_score": round(details_shapes['Score'].mean(), 4),
                "column_pair_trends_score": round(details_pairs['Score'].mean(), 4)
            }
            data_manager.insert_metadata(process_id, '44444444-4444-4444-4444-444444444444', metadata_value)
