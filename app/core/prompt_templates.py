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
