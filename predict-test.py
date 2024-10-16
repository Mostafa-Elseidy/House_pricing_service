import requests

url = 'http://localhost:9696/predict'

new_house_id = 'xyz-123'
new_house = {
    'Square_Feet': 4272,
    'Bedrooms': 3,
    'Age': 31,
    'Location_Rating': 7.108030
}


response = requests.post(url, json=new_house).json()
# print(response)

print(f"House Price Prediciton is {response['predicted_price']}")
