# --- Data Schema and Constants ---

TARGET_NAME = "MEDV"

# Define ranges and initial values based on boston.csv analysis
# All ranges are slightly widened from the min/max found in the data for robustness.
FEATURE_INFO = {
    "CRIM": {"label": "Crime Rate per Capita", "min": 0.0, "max": 90.0, "value": 0.1, "step": 0.1},
    "ZN": {"label": "Residential Land Zoned (>25k sq.ft) (%)", "min": 0.0, "max": 100.0, "value": 12.0, "step": 1.0},
    "INDUS": {"label": "Non-Retail Business Acres (%)", "min": 0.0, "max": 30.0, "value": 10.0, "step": 0.1},
    "CHAS_yes": {"label": "Bounds Charles River (1=Yes, 0=No)", "min": 0, "max": 1, "value": 0, "is_categorical": True},
    "NX": {
        "label": "Nitric Oxides Concentration (parts per 10 million)",
        "min": 0.3,
        "max": 0.9,
        "value": 0.55,
        "step": 0.01,
    },
    "RM": {"label": "Average Rooms per Dwelling", "min": 3.5, "max": 8.7, "value": 6.3, "step": 0.01},
    "AGE": {
        "label": "Owner-Occupied Units Built Prior to 1940 (%)",
        "min": 2.0,
        "max": 100.0,
        "value": 68.5,
        "step": 1.0,
    },
    "DIS": {"label": "Weighted Distances to 5 Employment Centers", "min": 1.1, "max": 12.2, "value": 3.8, "step": 0.01},
    "RAD": {"label": "Accessibility Index to Radial Highways", "min": 1, "max": 24, "value": 5, "step": 1},
    "TAX": {"label": "Property-Tax Rate ($10k)", "min": 180, "max": 720, "value": 400, "step": 1},
    "PTRATIO": {"label": "Pupil-Teacher Ratio", "min": 12.6, "max": 22.0, "value": 18.5, "step": 0.1},
    "LSTAT": {"label": "% Lower Status of the Population", "min": 1.7, "max": 37.0, "value": 12.7, "step": 0.1},
    # Target variable info (for display purposes)
    TARGET_NAME: {"label": "Median Home Value", "unit": "$1000s"},
}

# --- Model Paths ---
MODEL_DIR = "models"
MODEL_FILENAME = "boston-1.0.1.joblib"
METADATA_FILENAME = "boston-1.0.1.json"
