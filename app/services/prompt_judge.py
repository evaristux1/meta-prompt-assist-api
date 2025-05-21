import json
from app.providers.llm_provider import LLMProvider
from app.core.config import settings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

EVALUATION_TEMPLATE = """
Você é um avaliador especialista em engenharia de prompts. Sua tarefa é avaliar três versões de um prompt
(o original, a reformulação 1 e a reformulação 2) com base em 10 critérios rigorosos.
Forneça uma pontuação de 0 a 10 para cada critério em cada um dos três prompts.
Adicionalmente, determine qual das DUAS REFORMULAÇÕES (Reformulação 1 ou Reformulação 2) é a melhor entre si e forneça uma justificativa concisa para essa escolha.

Critérios de Avaliação (aplique a cada prompt individualmente):
1.  **Clareza e Especificidade**: Quão claro e específico é o prompt? Evita ambiguidades?
2.  **Estrutura e Organização**: O prompt é bem estruturado e organizado logicamente?
3.  **Consistência Interna**: As partes do prompt são consistentes entre si?
4.  **Contextualização**: O prompt fornece contexto suficiente para a LLM entender a tarefa?
5.  **Parâmetros de Execução**: Se aplicável, o prompt define claramente parâmetros como formato de saída, tom, etc.?
6.  **Eficiência Linguística**: O prompt usa linguagem concisa e eficaz? Evita redundâncias?
7.  **Robustez**: O prompt é robusto a pequenas variações de interpretação ou é muito frágil?
8.  **Adaptabilidade**: O prompt pode ser facilmente adaptado para tarefas ligeiramente diferentes?
9.  **Preparação e Enquadramento (Priming/Framing)**: O prompt prepara adequadamente a LLM para a resposta desejada?
10. **Princípios Éticos**: O prompt adere a princípios éticos, evitando vieses e conteúdo prejudicial?

Prompt Original:
\"\"\"
{prompt_original}
\"\"\"

Reformulação 1:
\"\"\"
{reformulation_1}
\"\"\"

Reformulação 2:
\"\"\"
{reformulation_2}
\"\"\"

RESPONDA EXCLUSIVAMENTE NO SEGUINTE FORMATO JSON. NÃO ADICIONE NENHUM TEXTO ANTES OU DEPOIS DO JSON.
O JSON DEVE SER COMPLETO E VÁLIDO.

Exemplo de formato de saída JSON esperado, não inclua o exemplo abaixo na sua resposta:
{{
  "evaluationData": [
    {{ "subject": "Clareza e Especificidade", "original": 7, "version1": 8, "version2": 9, "fullMark": 10 }},
    {{ "subject": "Estrutura e Organização", "original": 6, "version1": 7, "version2": 9, "fullMark": 10 }},
    {{ "subject": "Consistência Interna", "original": 8, "version1": 8, "version2": 9, "fullMark": 10 }},
    {{ "subject": "Contextualização", "original": 7, "version1": 7, "version2": 8, "fullMark": 10 }},
    {{ "subject": "Parâmetros de Execução", "original": 5, "version1": 6, "version2": 8, "fullMark": 10 }},
    {{ "subject": "Eficiência Linguística", "original": 8, "version1": 7, "version2": 9, "fullMark": 10 }},
    {{ "subject": "Robustez", "original": 6, "version1": 7, "version2": 8, "fullMark": 10 }},
    {{ "subject": "Adaptabilidade", "original": 7, "version1": 8, "version2": 7, "fullMark": 10 }},
    {{ "subject": "Preparação e Enquadramento", "original": 6, "version1": 7, "version2": 8, "fullMark": 10 }},
    {{ "subject": "Princípios Éticos", "original": 9, "version1": 9, "version2": 9, "fullMark": 10 }}
  ],
  "winningVersion": X,
  "justification": "A Reformulação x é superior à Reformulação y porque <justificativas encontradas e relevantes>."
}}
"""

def evaluate_reformulations(
    prompt_original: str, 
    reformulation_1: str,
    reformulation_2: str,
    judge_model_type: str = "gemini",
    judge_model_name: str = None
):
    """
    Avalia o prompt original e duas reformulações com base em 10 critérios.
    Retorna pontuações para os três e indica qual reformulação é a melhor entre si.
    Usa API_KEY_JUDGE das configurações para o LLM avaliador.
    """
    error_msg_prefix = f"evaluate_reformulations (judge_model: {judge_model_type}):"
    if not settings.API_KEY_JUDGE:
        raise ValueError(
            f"{error_msg_prefix} API_KEY_JUDGE não encontrada nas configurações/variáveis de ambiente."
        )

    try:
        judge_llm_provider = LLMProvider(model_type=judge_model_type, api_key_override=settings.API_KEY_JUDGE)
        judge_llm = judge_llm_provider.get_llm_instance()

        if judge_model_name:
            if hasattr(judge_llm, 'model_name'):
                judge_llm.model_name = judge_model_name
            elif hasattr(judge_llm, 'model'):
                judge_llm.model = judge_model_name
            print(f"{error_msg_prefix} Nome do modelo do judge explicitamente definido para: {judge_model_name}")

    except ValueError as ve:
        return {
            "error": f"{error_msg_prefix} Falha ao inicializar o LLM avaliador: {ve}",
            "raw_output": ""
        }

    prompt_template = PromptTemplate(
        template=EVALUATION_TEMPLATE,
        input_variables=["prompt_original", "reformulation_1", "reformulation_2"], 
        template_format="f-string"
    )
    
    evaluation_chain: RunnableSequence = prompt_template | judge_llm | StrOutputParser()

    raw_json_output_str = ""
    cleaned_json_str = "" 
    try:
        print(f"{error_msg_prefix} Invocando a chain de avaliação para três prompts...")
        raw_json_output_str = evaluation_chain.invoke({
            "prompt_original": prompt_original, 
            "reformulation_1": reformulation_1,
            "reformulation_2": reformulation_2
        })
        print(f"{error_msg_prefix} Saída bruta da LLM (string): '{raw_json_output_str[:500]}...'")
        
        cleaned_json_str = raw_json_output_str.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:]
            if cleaned_json_str.endswith("```"):
                 cleaned_json_str = cleaned_json_str[:-3]
            cleaned_json_str = cleaned_json_str.strip()
        
        if not cleaned_json_str.startswith("{") or not cleaned_json_str.endswith("}"):
            error_detail = "Saída da LLM (após tentativa de limpeza) não parece ser um JSON válido."
            print(f"{error_msg_prefix} {error_detail}\nConteúdo: {cleaned_json_str}")
        
        evaluation_result = json.loads(cleaned_json_str)
        print(f"{error_msg_prefix} JSON decodificado com sucesso.")

    except json.JSONDecodeError as e:
        error_message = (f"{error_msg_prefix} Falha ao decodificar JSON da LLM. Erro: {e}. "
                         f"Saída recebida (após limpeza):\n{cleaned_json_str}")
        print(error_message)
        return {
            "error": error_message,
            "raw_output": raw_json_output_str 
        }
    except Exception as e:
        error_message = (f"{error_msg_prefix} Erro inesperado ({type(e).__name__}) durante a avaliação da LLM: {e}. "
                         f"Saída parcial (se houver):\n{raw_json_output_str}")
        print(error_message)
        import traceback
        traceback.print_exc() 
        return {
            "error": error_message,
            "raw_output": raw_json_output_str 
        }

    return {
        "original_prompt_content": prompt_original, 
        "version1": {
            "title": "Reformulação 1", 
            "content": reformulation_1
        },
        "version2": {
            "title": "Reformulação 2", 
            "content": reformulation_2
        },
        "evaluationData": evaluation_result.get("evaluationData", []),
        "winningVersion": evaluation_result.get("winningVersion"), 
        "justification": evaluation_result.get("justification", "Sem justificativa fornecida pela LLM.")
    }

def evaluate_single_prompt(
    prompt: str,
    judge_model_type: str = "gemini",
    judge_model_name: str = None
):
    """
    Avalia um único prompt com base nos 10 critérios definidos.
    Retorna pontuações e justificativa.
    """
    error_msg_prefix = f"evaluate_single_prompt (judge_model: {judge_model_type}):"
    if not settings.API_KEY_JUDGE:
        raise ValueError(f"{error_msg_prefix} API_KEY_JUDGE não encontrada.")

    try:
        judge_llm_provider = LLMProvider(model_type=judge_model_type, api_key_override=settings.API_KEY_JUDGE)
        judge_llm = judge_llm_provider.get_llm_instance()

        if judge_model_name:
            if hasattr(judge_llm, 'model_name'):
                judge_llm.model_name = judge_model_name
            elif hasattr(judge_llm, 'model'):
                judge_llm.model = judge_model_name

    except ValueError as ve:
        return {
            "error": f"{error_msg_prefix} Falha ao inicializar o LLM avaliador: {ve}",
            "raw_output": ""
        }

    # Template específico para avaliação de 1 prompt
    SINGLE_EVAL_TEMPLATE = """
Você é um avaliador especialista em engenharia de prompts. Avalie o texto abaixo com base nos 10 critérios listados.

Texto:
\"\"\"
{prompt}
\"\"\"

Critérios de Avaliação:
1. Clareza e Especificidade
2. Estrutura e Organização
3. Consistência Interna
4. Contextualização
5. Parâmetros de Execução
6. Eficiência Linguística
7. Robustez
8. Adaptabilidade
9. Preparação e Enquadramento
10. Princípios Éticos

Responda EXCLUSIVAMENTE em JSON no seguinte formato:
{{
  "evaluationData": [
    {{ "subject": "Clareza e Especificidade", "score": 8, "fullMark": 10 }},
    ...
  ],
  "justification": "O prompt apresentou excelente clareza, organização e especificidade, porém faltou contextualização e definição de parâmetros."
}}
"""

    prompt_template = PromptTemplate(
        template=SINGLE_EVAL_TEMPLATE,
        input_variables=["prompt"],
        template_format="f-string"
    )

    evaluation_chain: RunnableSequence = prompt_template | judge_llm | StrOutputParser()

    raw_json_output_str = ""
    cleaned_json_str = ""
    try:
        print(f"{error_msg_prefix} Invocando avaliação de prompt único...")
        raw_json_output_str = evaluation_chain.invoke({"prompt": prompt})

        cleaned_json_str = raw_json_output_str.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:].strip("`").strip()

        if not cleaned_json_str.startswith("{") or not cleaned_json_str.endswith("}"):
            print(f"{error_msg_prefix} JSON mal formatado:\n{cleaned_json_str}")

        evaluation_result = json.loads(cleaned_json_str)
        print(f"{error_msg_prefix} JSON decodificado com sucesso.")

    except json.JSONDecodeError as e:
        return {
            "error": f"{error_msg_prefix} Falha ao decodificar JSON: {e}",
            "raw_output": raw_json_output_str
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"{error_msg_prefix} Erro inesperado: {e}",
            "raw_output": raw_json_output_str
        }

    return {
        "prompt": prompt,
        "evaluationData": evaluation_result.get("evaluationData", []),
        "justification": evaluation_result.get("justification", "Sem justificativa fornecida pela LLM.")
    }
