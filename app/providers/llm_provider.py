from langchain_core.language_models import BaseLanguageModel

from app.core.config import settings 

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

class LLMProvider:
    def __init__(self, model_type: str, api_key_override: str = None):
        """
        Inicializa o LLMProvider e carrega o modelo de linguagem especificado.
        Se api_key_override for fornecida, ela será usada. Caso contrário,
        as chaves de API padrão são obtidas de 'settings'.
        """
        self.model_type = model_type # Pode ser útil para logging ou debug
        self.llm: BaseLanguageModel = self._load_model(model_type, api_key_override)
        print(f"LLMProvider: Instância LLM '{self.model_type}' carregada (modelo específico: {self.llm.model_name if hasattr(self.llm, 'model_name') else getattr(self.llm, 'model', 'N/A')}).")


    def _load_model(self, model_type: str, api_key_override: str = None) -> BaseLanguageModel:
        """
        Carrega e retorna a instância do modelo de linguagem com base no model_type.
        Usa api_key_override se fornecida, caso contrário, usa as chaves de settings.
        """
        selected_api_key = None
        model_name_to_load = None 
        temperature_to_set = 0.7 # Temperatura padrão para geração
        # Definir prefixo para mensagens de log/print dentro deste método
        log_prefix = f"LLMProvider._load_model ({model_type}):"

        if model_type == "openai":
            selected_api_key = api_key_override if api_key_override else settings.API_KEY_OPENAI
            if not selected_api_key:
                raise ValueError(f"{log_prefix} API key para OpenAI ('API_KEY_OPENAI' ou override) não encontrada.")
            
            model_name_to_load = "gpt-3.5-turbo" # Modelo padrão para geração
            temperature_to_set = 0.7
            if api_key_override and api_key_override == settings.API_KEY_JUDGE:
                model_name_to_load = "gpt-4o" # Usando o modelo mais recente e capaz para judge
                temperature_to_set = 0.2
            
            print(f"{log_prefix} Carregando OpenAI model: {model_name_to_load}, temp: {temperature_to_set}, key_override: {'Sim' if api_key_override else 'Não'}")
            return ChatOpenAI(
                openai_api_key=selected_api_key,
                model=model_name_to_load,
                temperature=temperature_to_set
            )
        elif model_type == "groq":
            selected_api_key = api_key_override if api_key_override else settings.API_KEY_GROQ
            if not selected_api_key:
                raise ValueError(f"{log_prefix} API key para Groq ('API_KEY_GROQ' ou override) não encontrada.")
            
            model_name_to_load = "mixtral-8x7b-32768" # Modelo padrão para geração
            temperature_to_set = 0.7
            if api_key_override and api_key_override == settings.API_KEY_JUDGE:
                # Para Groq, podemos manter o mesmo modelo mas ajustar a temperatura,
                # ou escolher outro se disponível e mais adequado para avaliação.
                # model_name_to_load = "outromodelo_groq_para_judge" 
                temperature_to_set = 0.2
            
            print(f"{log_prefix} Carregando Groq model: {model_name_to_load}, temp: {temperature_to_set}, key_override: {'Sim' if api_key_override else 'Não'}")
            return ChatGroq(
                groq_api_key=selected_api_key,
                model_name=model_name_to_load,
                temperature=temperature_to_set
            )
        elif model_type == "gemini":
            selected_api_key = api_key_override if api_key_override else settings.API_KEY_GEMINI
            if not selected_api_key:
                raise ValueError(f"{log_prefix} API key para Gemini ('API_KEY_GEMINI' ou override) não encontrada.")
            
            model_name_to_load = "gemini-1.5-flash-latest" # Modelo padrão para geração
            temperature_to_set = 0.7
            if api_key_override and api_key_override == settings.API_KEY_JUDGE:
                model_name_to_load = "gemini-1.5-pro-latest" # Modelo mais robusto para judge
                temperature_to_set = 0.2
            
            print(f"{log_prefix} Carregando Gemini model: {model_name_to_load}, temp: {temperature_to_set}, key_override: {'Sim' if api_key_override else 'Não'}")
            return ChatGoogleGenerativeAI(
                google_api_key=selected_api_key,
                model=model_name_to_load,
                temperature=temperature_to_set,
                # convert_system_message_to_human=True # Pode ser útil para alguns modelos Gemini se usar mensagens de sistema
            )
        else:
            raise ValueError(f"{log_prefix} Modelo '{model_type}' não suportado. Opções: openai, groq, gemini.")

    def get_llm_instance(self) -> BaseLanguageModel:
        """Retorna a instância LLM carregada."""
        if not self.llm:
            # Isso não deveria acontecer se o construtor funcionar, mas é uma verificação de segurança.
            raise RuntimeError("LLMProvider: Instância LLM não foi carregada corretamente.")
        return self.llm
