def evaluate_reformulations(reformulation_1: str, reformulation_2: str):
    # Simula o juiz avaliando
    best = reformulation_1 if len(reformulation_1) < len(reformulation_2) else reformulation_2
    report = {
        "best_version": best,
        "scores": {"clarity": 9, "organization": 8, "context": 9},
        "justification": "Escolhemos a reformulação mais concisa e clara."
    }
    return report
