from app.providers.llm_provider import LLMProvider
from core.config import settings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
EVALUATION_TEMPLATE = """
Compare as duas reformulações abaixo com base nos seguintes critérios:

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

Reformulação 1:
\"\"\"{reformulation_1}\"\"\"

Reformulação 2:
\"\"\"{reformulation_2}\"\"\"

Responda exclusivamente neste formato JSON:

{
  "evaluationData": [
    {
      "subject": "Clareza e Especificidade",
      "version1": 8,
      "version2": 9,
      "fullMark": 10
    },
    {
      "subject": "Estrutura e Organização",
      "version1": 7,
      "version2": 9,
      "fullMark": 10
    },
    ...
  ],
  "winningVersion": 2,
  "justification": "A reformulação 2 é mais clara, organizada e aderente aos critérios éticos."
}
"""

def evaluate_reformulations(reformulation_1: str, reformulation_2: str, model_type: str = "openai"):
    """
    Avalia duas reformulações com base em 10 critérios e retorna no formato do front.
    """

    provider = LLMProvider(model_type=model_type)
    provider.llm.openai_api_key = settings.API_KEY_JUDGE

    prompt = PromptTemplate.from_template(EVALUATION_TEMPLATE)
    chain = LLMChain(llm=provider.llm, prompt=prompt)

    raw_output = chain.run(
        reformulation_1=reformulation_1,
        reformulation_2=reformulation_2
    )

    try:
        evaluation_result = json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "error": "Falha ao decodificar JSON da LLM",
            "raw_output": raw_output
        }

    return {
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
        "justification": evaluation_result.get("justification", "Sem justificativa.")
    }
