# config.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)