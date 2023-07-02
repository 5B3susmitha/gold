import requests

url = 'http://localhost:5000/predict_api'  # Replace with your API endpoint URL

# Make a POST request with the input values
r = requests.post(url, json={'eur_usd': 1.2, 'spi': 150, 'spx': 3000})

# Print the response
print(r.json())