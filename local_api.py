import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000") # Your code here

# TODO: print the status code
print(f"GET status code: {r.status_code}")
# TODO: print the welcome message
print(f"Welcome: {r.json()['message']}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/", json=data) # Your code here

# TODO: print the status code
print(f"POST Status Code: {r.status_code}")
# TODO: print the result
print(f"POST Request Result: {r.json()['result']}")
