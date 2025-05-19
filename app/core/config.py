import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """
    Configurações da aplicação, carregadas de variáveis de ambiente.
    Fornece um valor padrão vazio se a variável não for encontrada,
    mas as classes que usam essas chaves devem verificar se elas existem.
    """
    API_KEY_GEMINI: str = os.getenv("API_KEY_GEMINI", "")
    API_KEY_GROQ: str = os.getenv("API_KEY_GROQ", "")
    API_KEY_OPENAI: str = os.getenv("API_KEY_OPENAI", "")
    API_KEY_JUDGE: str = os.getenv("API_KEY_JUDGE", "") 

    def __init__(self):
        if not self.API_KEY_GEMINI:
            print("AVISO: API_KEY_GEMINI não definida no .env.")
        if not self.API_KEY_GROQ:
            print("AVISO: API_KEY_GROQ não definida no .env.")
        if not self.API_KEY_OPENAI:
            print("AVISO: API_KEY_OPENAI não definida no .env.")
        if not self.API_KEY_JUDGE:
            print("AVISO: API_KEY_JUDGE não definida no .env.")

settings = Settings()

