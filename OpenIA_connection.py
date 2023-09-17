import os
import openai
from dotenv import load_dotenv
import requests
import json

class GPT_connection:
  def __init__(self):
    load_dotenv() # read local .env file
    self.apiKey = openai.api_key = os.getenv('OPENAI_API_KEY')
    

  def talk(self, prompt):
    str_key = os.getenv('OPENAI_API_KEY')
    url = "https://api.openai.com/v1/chat/completions"

    payload = json.dumps({
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "user",
          "content": prompt
        }
      ],
      "temperature": 0.7
    })
    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {str_key}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    data = response.json()
    message = data["choices"][0]["message"]["content"]

    return message
  
# gpt = GPT_connection()

# print(gpt.talk("Hola como estas?"))