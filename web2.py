import os

import gradio as gr
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from const import FEATURE_INFO, METADATA_FILENAME, MODEL_DIR, MODEL_FILENAME
from ut_model import load_model_info_from_json

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILENAME)

try:
    model: Pipeline = joblib.load(MODEL_PATH)
    model_info = load_model_info_from_json(METADATA_PATH)
    FEATURES = model_info.data_schema.features
    MODEL_EQUATION = model_info.equation
    IS_MODEL_LOADED = True
except Exception as e:
    raise ValueError("Model loading failed. Please run `train_model.bat` first: " + str(e))


def predict_price(*args):
    processed_args = [
        1 if isinstance(arg, bool) and arg else 0 if isinstance(arg, bool) and not arg else arg for arg in args
    ]

    if not IS_MODEL_LOADED:
        return "Error: Model not loaded. Run `train_model.bat` first.", pd.DataFrame()

    input_data = dict(zip(FEATURES, processed_args))
    input_df = pd.DataFrame([input_data])

    input_df_final = input_df[FEATURES]

    try:
        prediction = model.predict(input_df_final)
        predicted_price = prediction[0]

        result_text = f"Predicted Median Home Value: **${predicted_price:,.2f} K**"

        return result_text, input_df_final
    except Exception as e:
        return f"Prediction Error: {e}", pd.DataFrame()


inputs = []
if IS_MODEL_LOADED:
    for feature in FEATURES:
        info = FEATURE_INFO.get(feature)

        if info is None:
            continue

        if info.get("is_categorical"):
            inputs.append(gr.Checkbox(value=info["value"], label=f"{feature} ({info['label']})"))
        else:
            inputs.append(
                gr.Slider(
                    minimum=info["min"],
                    maximum=info["max"],
                    value=info["value"],
                    step=info["step"],
                    label=f"{feature} ({info['label']})",
                )
            )
else:
    inputs = [gr.Textbox(label="Model Status", value="Model loading failed. Check console.", interactive=False)]

with gr.Blocks(title="Gradio OLS Boston Predictor") as demo:
    gr.Markdown("# OLS Boston Housing Price Predictor (Gradio Demo)")
    gr.Markdown("Model uses only selected features from the Boston dataset.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Input Features")
            for comp in inputs:
                comp.render()

        with gr.Column(scale=2):
            gr.Markdown("## 2. Input Data Summary")
            output_df = gr.Dataframe(label="Input Data")

            gr.Markdown("## 3. Model Info")
            gr.Code(f"Final Equation:\n{MODEL_EQUATION}")

            gr.Markdown("## 4. Prediction Result")
            output_text = gr.Markdown(
                label="Prediction Result", value="Adjust inputs on the left to see the result instantly."
            )

    if IS_MODEL_LOADED:
        for comp in inputs:
            comp.change(
                fn=predict_price,
                inputs=inputs,
                outputs=[output_text, output_df],
            )

        demo.load(fn=predict_price, inputs=inputs, outputs=[output_text, output_df])
    else:
        gr.Warning("Model failed to load. Please run `train_model.py` first.")

if __name__ == "__main__":
    demo.launch(inbrowser=True)
