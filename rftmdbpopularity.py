import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import ast

train = pd.read_csv('movie_data_version1.csv')
y_label = 'tmdb_popularity'
label = train[y_label]
data = train.drop(columns=[y_label])
data = data.drop(columns='age_certification')
data['genres'] = data['genres'].apply(ast.literal_eval)
encoded_genres = data['genres'].apply(pd.Series).stack().str.get_dummies().sum(level=0)
data_encoded = pd.concat([data, encoded_genres], axis=1)
data_encoded = data_encoded.drop(columns=['genres'])
data_encoded.fillna(data_encoded.mean(), inplace=True)
label.fillna(label.mean(), inplace=True)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(data_encoded, label)

feature_importances = model.feature_importances_

sorted_indices = feature_importances.argsort()[::-1]
sorted_importances = feature_importances[sorted_indices]
sorted_feature_names = data_encoded.columns[sorted_indices]

# Create a bar plot for feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names[:50], sorted_importances[:50])
plt.xlabel('Feature Importance')
plt.ylabel('Feature Name')
plt.title('Top 50 Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the figure for top 50 feature importances as an image
plt.savefig('top_50_feature_importances_rf_tmdb.png', bbox_inches='tight')
plt.close()
