from app.providers.llm_provider import LLMProvider
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence 
from langchain_core.output_parsers import StrOutputParser

class ReformulationError(Exception):
    """Exceção customizada para erros na geração de reformulações."""
    pass


UNIFIED_REFORMULATION_TEMPLATE = """
Com base nos seguintes critérios de qualidade para prompts, reformule o prompt original fornecido.
O objetivo é criar uma versão significativamente melhorada do prompt.

Critérios de Qualidade para Prompts:
1.  **Clareza e Especificidade**: Reduza ambiguidades, aumente o detalhamento e defina claramente os objetivos.
2.  **Estrutura e Organização**: Organize as informações de maneira lógica e hierárquica.
3.  **Consistência Interna**: Remova contradições e preserve coerência e terminologia uniforme.
4.  **Contextualização**: Forneça contexto suficiente, relevante e bem balanceado.
5.  **Parâmetros de Execução**: Inclua restrições, diretivas de formato e critérios claros de sucesso.
6.  **Eficiência Linguística**: Use linguagem concisa, precisa e econômica em tokens.
7.  **Robustez**: Adapte o prompt para prevenir falhas e ser verificável em casos extremos.
8.  **Adaptabilidade**: Torne o prompt flexível, reutilizável e aplicável a diferentes modelos.
9.  **Preparação e Enquadramento**: Use exemplos neutros e enquadre cognitivamente a tarefa.
10. **Princípios Éticos**: Assegure responsabilidade, transparência e inclusão.

Prompt Original a ser Reformulado:
\"\"\"
{prompt_original_text}
\"\"\"

Sua Tarefa:
Reescreva o prompt original, aplicando os critérios de qualidade acima para melhorá-lo substancialmente.
Retorne APENAS o novo prompt reformulado. Não inclua nenhuma explicação, introdução, ou qualquer texto além do prompt reformulado.
Para gerar uma variação, você pode, por exemplo, focar em diferentes aspectos dos critérios ou explorar diferentes formas de aplicar as melhorias.
"""

def generate_reformulations(original_prompt: str, generation_model_type: str = "gemini") -> tuple[str, str]:
    """
    Gera duas reformulações para o prompt original usando a MESMA orientação
    baseada em critérios de qualidade.
    Levanta ReformulationError em caso de falha.
    """
    print(f"Iniciando geração de DUAS reformulações para o prompt com modelo: {generation_model_type} usando orientação unificada.")
    error_msg_prefix = f"generate_reformulations (modelo: {generation_model_type}):"

    try:
        # Instancia o provedor LLM.
        # LLMProvider usa a chave de API padrão do .env para o generation_model_type
        # e seleciona um modelo/temperatura apropriados para geração.
        provider = LLMProvider(model_type=generation_model_type)
        llm_instance = provider.get_llm_instance() # Obtém a instância LLM configurada

        # Cria o template e a chain para a reformulação unificada
        unified_prompt_template = PromptTemplate(
            template=UNIFIED_REFORMULATION_TEMPLATE,
            input_variables=["prompt_original_text"] 
        )
        
        unified_reformulation_chain: RunnableSequence = unified_prompt_template | llm_instance | StrOutputParser()
        
        print(f"{error_msg_prefix} Gerando a primeira reformulação...")
        reformulation_1 = unified_reformulation_chain.invoke({
            "prompt_original_text": original_prompt
        })
        print(f"{error_msg_prefix} Primeira reformulação gerada.")

        if not reformulation_1 or not reformulation_1.strip():
            raise ReformulationError("A primeira reformulação resultou em uma string vazia ou None.")

        print(f"{error_msg_prefix} Gerando a segunda reformulação...")
  
        reformulation_2 = unified_reformulation_chain.invoke({
            "prompt_original_text": original_prompt
        })
        print(f"{error_msg_prefix} Segunda reformulação gerada.")

        if not reformulation_2 or not reformulation_2.strip():
            raise ReformulationError("A segunda reformulação resultou em uma string vazia ou None.")

        print(f"{error_msg_prefix} Ambas as reformulações geradas com sucesso.")
        return reformulation_1, reformulation_2
        
    except ValueError as ve:
        error_msg = f"Erro de configuração do provedor LLM para geração ({generation_model_type}): {ve}"
        print(f"{error_msg_prefix} {error_msg}")
        raise ReformulationError(error_msg)
    except ReformulationError: # Relança a exceção já tratada (ex: string vazia)
        raise
    except Exception as e:
        error_msg = f"Erro inesperado ({type(e).__name__}) durante a geração das reformulações: {e}"
        print(f"{error_msg_prefix} {error_msg}")
        import traceback
        traceback.print_exc()
        raise ReformulationError(error_msg)

