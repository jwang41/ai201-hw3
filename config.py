import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Data ---
DATA_PATH = "./data"
