# Code to identify features that impact 'is_viral' the most.
# Conclusion: The 'views' feature has an overwhelmingly high impact.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

# 1. Load Data
df = pd.read_csv('social_media_viral_content_dataset.csv')

# 2. Check the first 5 rows of the data
print("--- Data Sample (Head) ---")
print(df.head())

# 3. Check data summary (column names, data types, missing values, etc.)
print("\n--- Data Summary Info (Info) ---")
df.info()

# 4. Check descriptive statistics for numerical data (mean, std, quartiles, etc.)
pd.options.display.float_format = '{:,.0f}'.format
print("\n--- Descriptive Statistics (Describe) ---")
print(df.describe())

# 5. Check for missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# 6. Check the shape of the dataframe (rows, columns)
print("\n--- Data Shape ---")
print(df.shape)

# Copy original data for visualization (Box Plot)
df_viz = df.copy()
df_viz['is_viral_status'] = df_viz['is_viral'].map({0: 'Non-Viral', 1: 'Viral'})

# 7. Normalize Numerical Data (RobustScaler)
# Apply RobustScaler to handle scale differences and outliers between variables
scaler = RobustScaler()
performance_cols = ['views', 'likes', 'comments', 'shares']
df[performance_cols] = scaler.fit_transform(df[performance_cols])

# 8. Preprocessing and Feature Selection
# Apply One-Hot Encoding to convert categorical variables into numerical values
categorical_cols = ['platform', 'content_type', 'topic', 'language', 'region']
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Columns to exclude from analysis (IDs, dates, raw hashtags, and derived rates)
cols_to_drop = ['post_id', 'post_datetime', 'hashtags', 'engagement_rate', 'is_viral']
X = df_encoded.drop(columns=cols_to_drop)
y = df_encoded['is_viral']

# 9. Impact (Importance) Analysis - Using Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Organize feature importance into a DataFrame
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# 10. Plot 1: Feature Importance
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
plt.title('Feature Importance for Original Virality Label', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
print("Showing Plot 1: Feature Importance. Close the window to see the next plot.")
plt.show()

# 11. Plot 2: Raw Views Distribution (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='is_viral_status', y='views', data=df_viz, palette='coolwarm')
plt.title('Raw Views Distribution by Virality Status', fontsize=16, fontweight='bold')
plt.xlabel('Viral Status (Original)', fontsize=12)
plt.ylabel('Actual View Counts', fontsize=12)
plt.tight_layout()
print("Showing Plot 2: Raw Views Distribution.")
plt.show()

# 12. Print Detailed Numerical Results
print("\n--- Detailed Feature Importance (Original is_viral) ---")
print(importances.head(10))