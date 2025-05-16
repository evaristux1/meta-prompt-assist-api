from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI

from core.config import settings  # <-- Importa suas settings do dotenv


class LLMProvider:
    def __init__(self, model_type: str):
        """
        Inicializa o LLM com base no tipo desejado (groq, gemini, openai).
        Usa API_KEY_LLM1 por padrão, vindo de settings.
        """
        self.llm: BaseLanguageModel = self._load_model(model_type, settings.API_KEY_LLM1)

        self.template = PromptTemplate.from_template(
            "Reformule o seguinte prompt de forma {style}: {prompt}"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.template)

    def _load_model(self, model_type: str, api_key: str) -> BaseLanguageModel:
        if model_type == "openai":
            return ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        elif model_type == "groq":
            return ChatGroq(groq_api_key=api_key, model_name="mixtral-8x7b-32768", temperature=0.7)
        elif model_type == "gemini":
            return ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro", temperature=0.7)
        else:
            raise ValueError(f"Modelo '{model_type}' não suportado.")

    def generate_reformulations(self, prompt: str):
        """
        Gera duas reformulações com estilos diferentes para o prompt informado.
        """
        reformulation_1 = self.chain.run(prompt=prompt, style="criativa")
        reformulation_2 = self.chain.run(prompt=prompt, style="clara e objetiva")
        return reformulation_1, reformulation_2
