import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('gld_price_dataset.csv')

# Separate the features (inputs) and the target variable (gold price)
X = data[['SPX', 'USO', 'SLV', 'EUR/USD']]
y = data['GLD']

# Train the Random Forest Regression model
model = RandomForestRegressor()
model.fit(X, y)

# Save the trained model using pickle
with open('gld_price_model.pickle', 'wb') as f:
    pickle.dump(model, f)

# Load the trained model from pickle
with open('gld_price_model.pickle', 'rb') as f:
    loaded_model = pickle.load(f)

# Example input data for prediction
new_data = pd.DataFrame([[4000, 40, 25, 1.2]], columns=['SPX', 'USO', 'SLV', 'EUR/USD'])

# Predict the gold price
predicted_price = loaded_model.predict(new_data)[0]
print('Predicted GLD Price:', predicted_price)
