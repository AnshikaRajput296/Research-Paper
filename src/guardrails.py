import re

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?instructions",
    r"forget (everything|all|prior)",
    r"you are now",
    r"act as (a |an )?(?!research|assistant)",
    r"disregard (your |all )?",
    r"new persona",
    r"override (your |the )?",
    r"system prompt",
    r"jailbreak",
    r"pretend (you are|to be)",
    r"do anything now",
    r"dan mode",
]

OUT_OF_SCOPE_PHRASES = [
    "i don't have information",
    "not mentioned in",
    "cannot find",
    "no information available",
    "outside the scope",
    "not covered in the",
    "the provided context does not",
    "i cannot answer",
    "not found in the",
]


def is_prompt_injection(text: str) -> bool:
    """Return True if the text looks like a prompt injection attempt."""
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def validate_response_scope(answer: str) -> bool:
    """
    Return False if the answer indicates the information wasn't found
    in the uploaded papers (out-of-scope response).
    """
    answer_lower = answer.lower()
    for phrase in OUT_OF_SCOPE_PHRASES:
        if phrase in answer_lower:
            return False
    return True
