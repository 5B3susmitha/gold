import pandas as pd
import pickle

# Read the dataset
dataset = pd.read_csv('gold_dataset.csv')

# Separate the features and the target variable
X = dataset[['eur_usd', 'spi', 'spx']]
y = dataset['gold_price']

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Fitting the model with training data
regressor.fit(X, y)

# Saving the model to disk
pickle.dump(regressor, open('gold_price_prediction_model.pkl', 'wb'))

# Loading the model from disk
model = pickle.load(open('gold_price_prediction_model.pkl', 'rb'))

# Example usage: making a prediction
prediction = model.predict([[1.2, 150, 3000]])
print(prediction)
