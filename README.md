# House Price Prediction ML Project

This project showcases a complete workflow for training a **Machine Learning (ML)** model using **scikit-learn** and deploying it with two popular Python web frameworks: **Streamlit** and **Gradio**.

___

## Model Objective

The primary goal of the model is to predict **MEDV** (the median house price) using key independent variables related to marketing efforts and product characteristics.

___

## Deployment Options

### Option 1: Using GitHub Codespaces (for repository owners/contributors)
* **Fork** the repository.
* Use **GitHub Codespaces** to edit and run the project directly in your browser.
* The Streamlit application will be accessible at a dynamic URL, such as `https://curly-broccoli-qv445qp5w6h45jv-8501.app.github.dev` (example URL).

### Option 2: Using Streamlit Community Cloud (for repository owners/contributors)
* **Sign up** for the **Streamlit Community Cloud**.
* Create a new application, referencing the GitHub repository.
* The Streamlit application will run at a URL like `https://estimate-boston-house-price-ols.streamlit.app/` (example URL).

### Option 3: Using Local Python Environment
* Ensure **Python 3.11+** is installed.
* Install the dependency manager **`uv`**.
* Run `make-venv-uv.bat` to set up the virtual environment.
* Run `streamlit run web1.py` to start the Streamlit application.
* The Streamlit application will be accessible locally at **http://localhost:8501**.

### Option 4: Using Local Python with Gradio
* Ensure **Python 3.11+** is installed.
* Install the dependency manager **`uv`**.
* Run `make-venv-uv.bat` to set up the virtual environment.
* Run `python web2.py` to start the Gradio application.
* The Gradio application will be accessible locally at **http://localhost:8501**.

### Option 5: Using Local Python + ngrok for Public Access
* Follow **Option 3** steps first.
* **Sign up** on **ngrok.com** to get an authentication token and download **ngrok.exe**.
* Set your authentication token: `ngrok config add-autotoken %NGROK_TOKEN%`
* Run ngrok to expose your local port: `ngrok http 8501`
* The Streamlit application will be available locally at **http://localhost:8501** and publicly at a temporary **ngrok** URL, such as `https://9119e5dafa3e.ngrok-free.app` (example URL).
