import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Load positive and negative training and testing datasets
positive_train_data = pd.read_csv('480/kcr_cv+480.csv')
negative_train_data = pd.read_csv('480/kcr_cv-480.csv')
positive_test_data = pd.read_csv('480/Kcr_IND+480.csv')
negative_test_data = pd.read_csv('480/Kcr_IND-480.csv')

# Add binary labels
positive_train_data['label'] = 1
negative_train_data['label'] = 0
positive_test_data['label'] = 1
negative_test_data['label'] = 0

# Concatenate training and testing data
train_data = pd.concat([positive_train_data, negative_train_data], axis=0)
test_data = pd.concat([positive_test_data, negative_test_data], axis=0)

# Split features and labels
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

# Feature selection using ANOVA F-test (select top 300)
selector = SelectKBest(score_func=f_classif, k=300)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Separate selected features into positive and negative samples
positive_train_selected = X_train_selected[:len(positive_train_data)]
negative_train_selected = X_train_selected[len(positive_train_data):]
positive_test_selected = X_test_selected[:len(positive_test_data)]
negative_test_selected = X_test_selected[len(positive_test_data):]

# Convert to DataFrames and save as CSV
positive_train_df = pd.DataFrame(positive_train_selected)
negative_train_df = pd.DataFrame(negative_train_selected)
positive_test_df = pd.DataFrame(positive_test_selected)
negative_test_df = pd.DataFrame(negative_test_selected)

positive_train_df.to_csv('480/anova/train_anova+.csv', index=False)
negative_train_df.to_csv('480/anova/train_anova-.csv', index=False)
positive_test_df.to_csv('480/anova/test_anova+.csv', index=False)
negative_test_df.to_csv('480/anova/test_anova-.csv', index=False)

print("Top 300 features selected and saved by class to output files.")
