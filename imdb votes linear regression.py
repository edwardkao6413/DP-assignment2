from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import ast
import statsmodels.api as sm


train = pd.read_csv('movie_data_version1.csv')
y_label = 'imdb_votes'
label = train[y_label]
data = train.drop(columns=[y_label])

#research question does not include analysis of age certification
data = train.drop(columns= 'age_certification')

#encoding genre feature
data['genres'] = data['genres'].apply(ast.literal_eval)
encoded_genres = data['genres'].apply(pd.Series).stack().str.get_dummies().sum(level=0)
data_encoded = pd.concat([data, encoded_genres], axis=1)
data_encoded = data_encoded.drop(columns=['genres'])
data_encoded.fillna(data_encoded.mean(), inplace=True)  # Replace NaN values with column means
data_encoded = data_encoded.drop(columns=[y_label])

label.fillna(label.mean(), inplace=True)  # Replace NaN values in label with column mean

model = LinearRegression()
model.fit(data_encoded, label)

coefficients = model.coef_
print("Coefficient weight; ", coefficients)
feature_names = data_encoded.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, coefficients)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Linear Regression Coefficients')
plt.gca().invert_yaxis()
plt.yticks(rotation=45)
plt.yticks(fontsize=4)

plt.savefig('all_coefficients_plot_imdb_votes.png', bbox_inches='tight')
plt.show()

#need to make a top 50 and bottom 50
top_50_coefficients = coefficients.argsort()[-50:][::-1]
bottom_50_coefficients = coefficients.argsort()[:50]

plt.figure(figsize=(10, 6))
plt.barh(feature_names[top_50_coefficients], coefficients[top_50_coefficients])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Top 50 Linear Regression Coefficients')
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the figure for top 50 coefficients as an image
plt.savefig('top_50_coefficients_imdb_votes.png', bbox_inches='tight')
plt.close()

# Create a figure for the bottom 50 coefficients
plt.figure(figsize=(10, 6))
plt.barh(feature_names[bottom_50_coefficients], coefficients[bottom_50_coefficients])
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Bottom 50 Linear Regression Coefficients')
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the figure for bottom 50 coefficients as an image
plt.savefig('bottom_50_coefficients_imdb_votes.png', bbox_inches='tight')
plt.close()

regre_ols = sm.OLS( endog=np.array(label, dtype = float), exog = np.array(data_encoded, dtype = float)).fit()
output_file_path = 'regression_results_imdb_votes.txt'
with open(output_file_path, 'w') as f:
    f.write(regre_ols.summary().as_text())



