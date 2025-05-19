from fastapi import APIRouter, HTTPException, status
from app.schemas.prompt import PromptRequest, PromptResponse, VersionInfo # Importe os schemas atualizados
from app.services.prompt_engineering import generate_reformulations, ReformulationError # Importe o serviço e a exceção
from app.services.prompt_judge import evaluate_reformulations # Seu serviço de avaliação

router = APIRouter()

@router.post("/processar-prompt",
             response_model=PromptResponse,
             summary="Processa um prompt, gera reformulações e as avalia",
             tags=["Prompt Processing"])
async def process_prompt(request: PromptRequest):
    """
    Recebe um prompt, gera duas reformulações (criativa e clara/objetiva)
    usando o `generation_model_type` especificado, e então avalia essas
    reformulações usando o `judge_model_type` especificado.

    Retorna as reformulações, os dados da avaliação, a versão vencedora e uma justificativa.
    """
    try:
        reform1_content, reform2_content = generate_reformulations(
            original_prompt=request.prompt,
            generation_model_type=request.generation_model_type
        )
    except ReformulationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha ao gerar reformulações: {str(e)}"
        )
    except Exception as e: 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro inesperado no serviço de geração: {str(e)}"
        )

    if not reform1_content or not reform2_content:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Serviço de geração retornou conteúdo vazio para reformulações."
        )

    evaluation_report = evaluate_reformulations(
        prompt_original=request.prompt,
        reformulation_1=reform1_content,
        reformulation_2=reform2_content,
        judge_model_type=request.judge_model_type
    )

    if "error" in evaluation_report:
        print(f"Erro do serviço de avaliação: {evaluation_report['error']}")
        print(f"Saída bruta do judge: {evaluation_report.get('raw_output')}")
        return PromptResponse(
            original_prompt=request.prompt,
            version1=VersionInfo(title="Reformulação 1 (Criativa)", content=reform1_content),
            version2=VersionInfo(title="Reformulação 2 (Clara e Objetiva)", content=reform2_content),
            error=f"Falha na avaliação das reformulações: {evaluation_report['error']}",
            raw_judge_output=str(evaluation_report.get('raw_output', ''))
        )

    return PromptResponse(
        original_prompt=request.prompt,
        version1=VersionInfo(
            title=evaluation_report.get("version1", {}).get("title", "Reformulação 1 (Criativa)"),
            content=evaluation_report.get("version1", {}).get("content", reform1_content)
        ),
        version2=VersionInfo(
            title=evaluation_report.get("version2", {}).get("title", "Reformulação 2 (Clara e Objetiva)"), 
            content=evaluation_report.get("version2", {}).get("content", reform2_content)
        ),
        evaluationData=evaluation_report.get("evaluationData"),
        winningVersion=evaluation_report.get("winningVersion"),
        justification=evaluation_report.get("justification")
    )

