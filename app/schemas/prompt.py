from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="O prompt original a ser processado.")
    generation_model_type: str = Field("gemini", description="Tipo de modelo para gerar reformulações (ex: 'gemini', 'openai', 'groq').")
    judge_model_type: str = Field("gemini", description="Tipo de modelo para avaliar as reformulações (ex: 'gemini', 'openai', 'groq').")

class VersionInfo(BaseModel):
    title: str
    content: str

class EvaluationDataItem(BaseModel):
    subject: str
    original: float 
    version1: float 
    version2: float 
    fullMark: float = Field(10.0)

class PromptResponse(BaseModel):
    original_prompt: str
    version1: Optional[VersionInfo] = None
    version2: Optional[VersionInfo] = None
    evaluationData: Optional[List[EvaluationDataItem]] = None
    winningVersion: Optional[int] = None
    justification: Optional[str] = None
    error: Optional[str] = None
    raw_judge_output: Optional[str] = None 



class SinglePromptRequest(BaseModel):
    prompt: str
    judge_model_type: Optional[str] = "gemini"


class EvaluationItem(BaseModel):
    subject: str
    score: int
    fullMark: int


class SinglePromptResponse(BaseModel):
    prompt: str
    evaluationData: List[EvaluationItem]
    justification: str