# controller.py

from typing import Dict

from intent_router import route_request
from input_validation import process_user_input, empty_property_state


# -----------------------------
# Main conversation controller
# -----------------------------
def handle_user_message(
    user_message: str,
    session_state: Dict
) -> Dict[str, object]:
    """
    Central orchestration point for user messages.
    """

    # Initialize property state if needed
    if "property_state" not in session_state:
        session_state["property_state"] = empty_property_state()

    # Detect intent
    route = route_request(user_message)

    # -----------------------------
    # Property evaluation flow
    # -----------------------------
    if route["intent"] == "property_evaluation":
        result = process_user_input(
            user_message=user_message,
            current_state=session_state["property_state"]
        )

        session_state["property_state"] = result["updated_state"]

        if not result["ready_for_analysis"]:
            return {
                "type": "followup",
                "message": result["followup_question"]
            }

        return {
            "type": "ready_for_analysis",
            "property_state": session_state["property_state"]
        }

    # -----------------------------
    # Location comparison flow
    # -----------------------------
    if route["intent"] == "location_comparison":
        return {
            "type": "clarify_location_intent",
            "message": "Are you planning to buy this as a rental or for personal use, and what’s your approximate budget?"
        }

    # -----------------------------
    # Unknown / fallback
    # -----------------------------
    return {
        "type": "clarification",
        "message": "Can you clarify what you’re looking to evaluate?"
    }
