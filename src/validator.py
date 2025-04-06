class DataValidator:
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def check_k_anonymity(self, quasi_identifiers, k=3):
        grouped = self.df.groupby(quasi_identifiers).size().reset_index(name='counts')
        violations = grouped[grouped['counts'] < k]
        return violations.empty, violations

    def check_l_diversity(self, quasi_identifiers, sensitive_attribute, l=2):
        grouped = self.df.groupby(quasi_identifiers)[sensitive_attribute].nunique().reset_index(name='diversity')
        violations = grouped[grouped['diversity'] < l]
        return violations.empty, violations

    def check_t_closeness(self, quasi_identifiers, sensitive_attribute, original_df, t=0.2):
        synthetic_dist = self.df[sensitive_attribute].value_counts(normalize=True)
        original_dist = original_df[sensitive_attribute].value_counts(normalize=True)
        total_variation = abs(synthetic_dist - original_dist).sum() / 2
        return total_variation <= t, total_variation
