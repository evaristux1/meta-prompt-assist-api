from fastapi import APIRouter
from app.schemas.prompt import PromptRequest, PromptResponse
from app.services.prompt_engineering import generate_reformulations
from app.services.prompt_judge import evaluate_reformulations

router = APIRouter()

@router.post("/processar-prompt", response_model=PromptResponse)
def process_prompt(request: PromptRequest):
    reform1, reform2 = generate_reformulations(request.prompt)
    report = evaluate_reformulations(reform1, reform2)
    return PromptResponse(
        optimized_prompt=report["best_version"],
        evaluation_report=report
    )
