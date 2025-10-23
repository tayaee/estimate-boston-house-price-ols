@echo off
call make-venv-uv.bat
if .%VIRTUAL_ENV%. == .. (
    if not exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate
    )
)
python train_model.py --input=data/boston.csv --output-model=models/boston-1.0.1.joblib
