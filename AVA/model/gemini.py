import google.generativeai as genai
import os
genai.configure(api_key="AIzaSyDoH4bxqpmoe0oiGvwbUlqxRQ5SlyUhCvU")
model = genai.GenerativeModel(model_name="tunedModels/mentalhealthchatbot-7mrtrsg0fib1")
response = model.generate_content("What is a panic attack?")
print(response.text)