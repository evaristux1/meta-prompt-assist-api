from pydantic import BaseModel
from typing import List

class PromptRequest(BaseModel):
    prompt: str

class PromptRequest(BaseModel):
    prompt: str

class PromptVersion(BaseModel):
    title: str
    content: str

class EvaluationCriteria(BaseModel):
    subject: str
    version1: int
    version2: int
    fullMark: int

class PromptResponse(BaseModel):
    version1: PromptVersion
    version2: PromptVersion
    evaluationData: List[EvaluationCriteria]
    winningVersion: int
    justification: str
