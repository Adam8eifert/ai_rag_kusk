import requests

API_URL = "http://localhost:8000/ask"

while True:
    question = input("\nZadej dotaz (nebo 'exit'): ")
    if question.lower() == "exit":
        break

    response = requests.post(API_URL, json={"question": question})
    print(response.json())
