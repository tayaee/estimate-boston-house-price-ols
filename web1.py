import os

import joblib
import pandas as pd
import streamlit as st

from const import FEATURE_INFO, METADATA_FILENAME, MODEL_DIR, MODEL_FILENAME, TARGET_NAME
from ut_model import load_model_info_from_json

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILENAME)


@st.cache_resource
def load_model_assets():
    """Load model and metadata."""
    try:
        model = joblib.load(MODEL_PATH)
        model_info = load_model_info_from_json(METADATA_PATH)
        return model, model_info
    except FileNotFoundError:
        st.error("Model or metadata not found. Run `train_model.bat` first.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def main():
    st.set_page_config(page_title="Streamlit OLS Boston Predictor", layout="wide")
    st.title("OLS Boston Housing Price Predictor")
    st.markdown("Model uses only selected features from the Boston dataset. (v4)")

    model, model_info = load_model_assets()

    if model and model_info:
        # Use features selected by the final model
        features = model_info.data_schema.features
        st.sidebar.header("Input Features")

        input_data = {}
        st.sidebar.markdown("---")

        # Create input UI using constants for features included in the model
        for feature in features:
            info = FEATURE_INFO.get(feature)

            if info is None:
                # Fallback for unexpected features, though unlikely if train_model is correct
                info = {"label": feature, "min": 0.0, "max": 10.0, "value": 1.0, "step": 0.1}
                st.sidebar.warning(f"Using default range for missing feature: {feature}")

            if info.get("is_categorical"):
                # Handle CHAS_yes as a Selectbox
                input_data[feature] = st.sidebar.selectbox(
                    f"**{feature.split('_')[0]}** ({info['label']})",
                    options=[0, 1],
                    index=info["value"],
                )
            else:
                # Handle continuous features as Sliders
                input_data[feature] = st.sidebar.slider(
                    f"**{feature}** ({info['label']})",
                    info["min"],
                    info["max"],
                    info["value"],
                    step=info["step"],
                )

        input_df = pd.DataFrame([input_data])

        st.subheader("1. Input Data")
        st.dataframe(input_df, use_container_width=True)

        # Model Info Section (Simplified)
        st.subheader("2. Model Info")
        st.info(f"**Final Equation**: {model_info.equation}")
        st.markdown(f"**Features Used**: {', '.join(features)}")

        try:
            prediction = model.predict(input_df[features])
            predicted_price = prediction[0]

            st.subheader("3. Prediction Result")
            st.metric(label=f"Predicted Median Home Value ({TARGET_NAME})", value=f"${predicted_price:,.2f} K")
        except Exception as e:
            st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
