# SmartCare AI – Appointment No-Show Predictor

Modern Streamlit prototype for CCS3440 Option A.

## Files
- `streamlit_app.py` – Streamlit application
- `appointment_no_show_best_model.joblib` – trained Random Forest pipeline
- `model_metadata.json` – evaluation metrics and feature importance
- `requirements.txt` – Python dependencies
- `run_app.bat` – Windows launcher

## Run locally

1. Open a terminal in this folder.
2. Install dependencies:

   pip install -r requirements.txt

3. Run the application:

   streamlit run streamlit_app.py

Then open the local URL shown by Streamlit.

## Model inputs
The application uses the same 12 predictors and feature engineering used in the supplied Jupyter notebook.

## Important
This is an academic decision-support prototype using a synthetic coursework dataset.
It is not clinically validated and should not be used to deny or restrict patient care.
# SmartCare-AI
