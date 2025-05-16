

from app.providers.llm_provider import LLMProvider


PROMPT_EVALUATION_CRITERIA = """
Avalie e reformule o prompt a seguir com base nos seguintes critérios de qualidade:

1. **Clareza e Especificidade** – Reduza ambiguidades, aumente o detalhamento e defina claramente os objetivos.
2. **Estrutura e Organização** – Organize as informações de maneira lógica e hierárquica.
3. **Consistência Interna** – Remova contradições e preserve coerência e terminologia uniforme.
4. **Contextualização** – Forneça contexto suficiente, relevante e bem balanceado.
5. **Parâmetros de Execução** – Inclua restrições, diretivas de formato e critérios claros de sucesso.
6. **Eficiência Linguística** – Use linguagem concisa, precisa e econômica em tokens.
7. **Robustez** – Adapte o prompt para prevenir falhas e ser verificável em casos extremos.
8. **Adaptabilidade** – Torne o prompt flexível, reutilizável e aplicável a diferentes modelos.
9. **Preparação e Enquadramento** – Use exemplos neutros e enquadre cognitivamente a tarefa.
10. **Princípios Éticos** – Assegure responsabilidade, transparência e inclusão.

Prompt original:
\"\"\"{prompt}\"\"\"

Sua tarefa:
Reescreva o prompt original melhorando-o com base em todos os critérios acima. Retorne apenas o novo prompt reformulado.
"""

def generate_reformulations(prompt: str, model_type: str = "groq") -> tuple[str, str]:
    """
    Gera duas reformulações do prompt original com base nos critérios do MetaPromptAssist.
    """
    provider = LLMProvider(model_type=model_type)

    # Primeira reformulação: estilo neutro com foco em qualidade geral
    reformulation_1 = provider.chain.run(prompt=prompt)

    # Segunda reformulação: com leve alteração no estilo, por exemplo, mais técnico
    styled_prompt = PROMPT_EVALUATION_CRITERIA + "\nAdote um estilo mais técnico e direto."
    provider.chain.prompt.template = styled_prompt
    reformulation_2 = provider.chain.run(prompt=prompt)

    return reformulation_1, reformulation_2
