# features.py

from typing import Dict, List
from datetime import datetime


# -----------------------------
# Feature contract (STABLE)
# -----------------------------
FEATURE_ORDER: List[str] = [
    "beds",
    "baths",
    "sqft",
    "property_age",
    "price",
    "city_index",
    "inflation",
    "interest_rate",
    "population_growth"
]


# -----------------------------
# City encoder (example)
# -----------------------------
# In production this would be:
# - learned during training, or
# - loaded from a saved encoder artifact
CITY_ENCODER = {
    "toronto": 0,
    "vancouver": 1,
    "calgary": 2,
    "edmonton": 3,
    "winnipeg": 4,
    "halifax": 5
}


# -----------------------------
# Validation helpers
# -----------------------------
def _require(value, field_name: str):
    if value is None:
        raise ValueError(f"Missing required feature: {field_name}")
    return value


def _positive(value, field_name: str):
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


# -----------------------------
# Feature engineering
# -----------------------------
def compute_property_age(year_built: int | None) -> int:
    """
    Converts year_built into property age.
    If year_built is unknown, returns a conservative default.
    """
    current_year = datetime.now().year

    if year_built is None:
        # Conservative assumption
        return 30

    return max(0, current_year - year_built)


# -----------------------------
# Feature normalization
# -----------------------------
def build_feature_vector(
    property_state: Dict,
    context_state: Dict
) -> List[float]:
    """
    Builds a normalized, ordered feature vector
    compatible with the trained ML model.
    """

    # -----------------------------
    # Required property features
    # -----------------------------
    beds = _positive(
        _require(property_state.get("beds"), "beds"),
        "beds"
    )

    baths = _positive(
        _require(property_state.get("baths"), "baths"),
        "baths"
    )

    sqft = _positive(
        _require(property_state.get("sqft"), "sqft"),
        "sqft"
    )

    price = _positive(
        _require(property_state.get("price"), "price"),
        "price"
    )

    # -----------------------------
    # Derived features
    # -----------------------------
    property_age = compute_property_age(
        property_state.get("year_built")
    )

    # -----------------------------
    # Categorical encoding
    # -----------------------------
    city = _require(property_state.get("city"), "city").lower()

    if city not in CITY_ENCODER:
        raise ValueError(f"Unsupported city: {city}")

    city_index = CITY_ENCODER[city]

    # -----------------------------
    # Context features (from APIs)
    # -----------------------------
    inflation = _require(
        context_state.get("inflation"),
        "inflation"
    )

    interest_rate = _require(
        context_state.get("interest_rate"),
        "interest_rate"
    )

    population_growth = _require(
        context_state.get("population_growth"),
        "population_growth"
    )

    # -----------------------------
    # Assemble vector (ORDER MATTERS)
    # -----------------------------
    feature_vector = [
        beds,
        baths,
        sqft,
        property_age,
        price,
        city_index,
        inflation,
        interest_rate,
        population_growth
    ]

    return feature_vector


# -----------------------------
# Debug / inspection helper
# -----------------------------
def describe_features() -> Dict[str, str]:
    """
    Returns a human-readable description
    of each feature (useful for debugging or UI display).
    """
    return {
        "beds": "Number of bedrooms",
        "baths": "Number of bathrooms",
        "sqft": "Interior living area in square feet",
        "property_age": "Estimated age of the property",
        "price": "Purchase price of the property",
        "city_index": "Encoded city identifier",
        "inflation": "Current inflation rate",
        "interest_rate": "Current interest rate",
        "population_growth": "Annual population growth rate"
    }
