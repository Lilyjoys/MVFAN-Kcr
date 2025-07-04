import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


positive_data = pd.read_csv('onehot+esm2/onehot+anova/train-onehot+anova+.csv')
negative_data = pd.read_csv('onehot+esm2/onehot+anova/train-onehot+anova-.csv')


positive_data['label'] = 1
negative_data['label'] = 0


combined_data = pd.concat([positive_data, negative_data])

# Analyze potential stratification features
print("Categorical features in the dataset:")
categorical_features = []
for col in combined_data.columns:
    if combined_data[col].dtype == 'object' or combined_data[col].nunique() < 20:
        categorical_features.append(col)
        print(f"- {col}: {combined_data[col].nunique()} unique values")

# Compute correlation between features and labels, and select the best feature
print("\nCorrelation between features and label:")
best_feature = None
best_chi2 = 0

for feature in categorical_features:
    if feature != 'label':
        # Perform chi-squared test
        contingency_table = pd.crosstab(combined_data[feature], combined_data['label'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        print(f"- {feature}: Chi-squared = {chi2:.2f}, p-value = {p:.4f}")

        # Select the feature with the highest chi-squared value
        if chi2 > best_chi2:
            best_chi2 = chi2
            best_feature = feature

if best_feature:
    print(f"\nAutomatically selected stratification feature: {best_feature} (Chi-squared = {best_chi2:.2f})")
    stratify_feature = best_feature
else:
    print("Error: No suitable stratification feature found.")
    exit(1)

#Compute sample distribution within each stratum
strata = combined_data[stratify_feature].unique()
strata_distribution = {}
for stratum in strata:
    stratum_data = combined_data[combined_data[stratify_feature] == stratum]
    positive_count = len(stratum_data[stratum_data['label'] == 1])
    negative_count = len(stratum_data[stratum_data['label'] == 0])
    strata_distribution[stratum] = {'positive': positive_count, 'negative': negative_count}

# Determine target sample size per stratum (use the smaller class size)
target_samples_per_stratum = {}
for stratum, counts in strata_distribution.items():
    target_samples_per_stratum[stratum] = min(counts['positive'], counts['negative'])

#  Perform stratified downsampling
balanced_strata = []
for stratum in strata:
    stratum_data = combined_data[combined_data[stratify_feature] == stratum]
    positive_samples = stratum_data[stratum_data['label'] == 1]
    negative_samples = stratum_data[stratum_data['label'] == 0]

    # Downsample the majority class
    target_size = target_samples_per_stratum[stratum]
    if len(positive_samples) > target_size:
        positive_samples = positive_samples.sample(n=target_size, random_state=42)
    if len(negative_samples) > target_size:
        negative_samples = negative_samples.sample(n=target_size, random_state=42)

    # Combine the downsampled data
    balanced_stratum = pd.concat([positive_samples, negative_samples])
    balanced_strata.append(balanced_stratum)


balanced_data = pd.concat(balanced_strata)


balanced_positive_data = balanced_data[balanced_data['label'] == 1].drop(columns=['label'])
balanced_negative_data = balanced_data[balanced_data['label'] == 0].drop(columns=['label'])

balanced_positive_data.to_csv('onehot+esm2/onehot+anova/onehot+anova+SS2.csv', index=False)
balanced_negative_data.to_csv('onehot+esm2/onehot+anova/onehot+anova-SS2.csv', index=False)


print("Stratified downsampling results:")
for stratum, counts in strata_distribution.items():
    original_pos = counts['positive']
    original_neg = counts['negative']
    target = target_samples_per_stratum[stratum]
    print(f"Stratum '{stratum}': Positive {original_pos} -> {target}, Negative {original_neg} -> {target}")

print("\nFinal balanced dataset:")
print("Number of positive samples:", len(balanced_positive_data))
print("Number of negative samples:", len(balanced_negative_data))
