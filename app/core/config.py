import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_KEY_LLM1: str = os.getenv("API_KEY_LLM1", "")
    API_KEY_LLM2: str = os.getenv("API_KEY_LLM2", "")
    API_KEY_JUDGE: str = os.getenv("API_KEY_JUDGE", "")

settings = Settings()
