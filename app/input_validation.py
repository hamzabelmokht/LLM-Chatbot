# input_validation.py

from typing import Dict, List, Optional
import json

from llm import call_llm


# -----------------------------
# Required property fields
# -----------------------------
REQUIRED_FIELDS = [
    "city",
    "property_type",
    "beds",
    "baths",
    "sqft",
    "price"
]


# -----------------------------
# Default empty schema
# -----------------------------
def empty_property_state() -> Dict[str, Optional[object]]:
    return {
        "city": None,
        "property_type": None,
        "beds": None,
        "baths": None,
        "sqft": None,
        "price": None
    }


# -----------------------------
# LLM-based extraction
# -----------------------------
def extract_property_fields(
    user_message: str,
    current_state: Dict[str, Optional[object]]
) -> Dict[str, Optional[object]]:
    """
    Uses the LLM to extract structured property information
    from the user's natural-language message.
    """

    prompt = f"""
You are an information extraction system.

Extract property-related details from the user message below.
Return ONLY valid JSON.
Use null if a field is not mentioned.

Fields:
- city (string)
- property_type (string)
- beds (integer)
- baths (integer)
- sqft (integer)
- price (number)

Existing known values (do not overwrite unless user corrects them):
{json.dumps(current_state)}

User message:
"{user_message}"
"""

    raw_response = call_llm(prompt)

    try:
        extracted = json.loads(raw_response)
    except json.JSONDecodeError:
        # If LLM fails, return state unchanged
        return current_state

    # Merge extracted values into state
    updated_state = current_state.copy()
    for key in updated_state:
        if key in extracted and extracted[key] is not None:
            updated_state[key] = extracted[key]

    return updated_state


# -----------------------------
# Validation
# -----------------------------
def missing_required_fields(
    state: Dict[str, Optional[object]]
) -> List[str]:
    """
    Returns a list of required fields that are still missing.
    """
    return [field for field in REQUIRED_FIELDS if state.get(field) is None]


# -----------------------------
# Follow-up question generation
# -----------------------------
def generate_followup_question(missing_fields: List[str]) -> str:
    """
    Uses the LLM to generate a natural follow-up question
    for the missing fields.
    """

    prompt = f"""
You are a conversational assistant collecting missing information.

The following required property details are missing:
{missing_fields}

Ask ONE concise, natural question to the user
that helps collect these details.
Do not explain why you are asking.
"""

    return call_llm(prompt).strip()


# -----------------------------
# Orchestrator
# -----------------------------
def process_user_input(
    user_message: str,
    current_state: Dict[str, Optional[object]]
) -> Dict[str, object]:
    """
    Main entry point for input validation.

    Returns a dictionary with:
    - updated_state
    - ready_for_analysis (bool)
    - followup_question (optional)
    """

    updated_state = extract_property_fields(
        user_message=user_message,
        current_state=current_state
    )

    missing = missing_required_fields(updated_state)

    if missing:
        return {
            "updated_state": updated_state,
            "ready_for_analysis": False,
            "followup_question": generate_followup_question(missing)
        }

    return {
        "updated_state": updated_state,
        "ready_for_analysis": True,
        "followup_question": None
    }
