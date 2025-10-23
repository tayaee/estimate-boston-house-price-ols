@echo off
call make-venv-uv.bat
if .%VIRTUAL_ENV%. == .. (
    if not exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate
    )
)
python web2.py
