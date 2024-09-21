import json
import requests

# URL of the FastAPI endpoint
url = 'http://127.0.0.10:8080/fake_news_prediction'

# Input data for the model
input_data_for_model = {
    'title': 'NASAs Perseverance Rover Successfully Lands on Mars',
    'author': 'Emily Thompson',
    'text' : 'NASAs Perseverance rover has successfully landed on Mars, marking a historic achievement in space exploration. The rover will now begin its mission to search for signs of ancient life and collect samples of Martian rock and soil.'
}

input_json = json.dumps(input_data_for_model)

# Send a POST request to the FastAPI endpoint
response = requests.post(url, data=input_json, headers={'Content-Type': 'application/json'})

# Print the response from the server
print(response.text)
