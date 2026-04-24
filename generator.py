import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

my_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key = my_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_response(prompt):
    response = model.generate_content(prompt)
    return response.text