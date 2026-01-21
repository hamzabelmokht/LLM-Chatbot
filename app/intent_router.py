# intent_router.py

from enum import Enum
from typing import Dict

from llm import call_llm


# -----------------------------
# Intent definitions
# -----------------------------
class UserIntent(str, Enum):
    PROPERTY_EVALUATION = "property_evaluation"
    LOCATION_COMPARISON = "location_comparison"
    UNKNOWN = "unknown"


# -----------------------------
# Intent detection
# -----------------------------
def detect_intent(user_message: str) -> UserIntent:
    """
    Uses the LLM to classify the user's intent.
    Returns a UserIntent enum.
    """

    prompt = f"""
You are an intent classification system.

Classify the user's intent into ONE of the following categories:

1. property_evaluation
   - The user describes a specific property (beds, baths, size, city, etc.)
   - They want to know if it is a good rental or investment

2. location_comparison
   - The user asks where it is best to buy
   - They want to compare cities or regions
   - No specific property is fully described yet

3. unknown
   - The intent is unclear or conversational

Return ONLY one of:
- property_evaluation
- location_comparison
- unknown

User message:
"{user_message}"
"""

    response = call_llm(prompt).strip().lower()

    if "property_evaluation" in response:
        return UserIntent.PROPERTY_EVALUATION
    if "location_comparison" in response:
        return UserIntent.LOCATION_COMPARISON

    return UserIntent.UNKNOWN


# -----------------------------
# Routing decision
# -----------------------------
def route_request(user_message: str) -> Dict[str, str]:
    """
    High-level routing decision used by the UI or controller layer.
    """

    intent = detect_intent(user_message)

    return {
        "intent": intent.value
    }
