from pathlib import Path
from datetime import date
import json

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "appointment_no_show_best_model.joblib"
META_PATH = APP_DIR / "model_metadata.json"

st.set_page_config(
    page_title="SmartCare AI | No-Show Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- UI styling ----------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 15% 5%, rgba(51, 153, 255, 0.12), transparent 28%),
            radial-gradient(circle at 85% 12%, rgba(42, 207, 165, 0.10), transparent 26%),
            #f7f9fc;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f33 0%, #102c46 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f6f8fb;
    }

    .hero {
        padding: 1.5rem 1.6rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0b1f33 0%, #1261a0 58%, #13a88f 100%);
        box-shadow: 0 16px 38px rgba(11, 31, 51, 0.18);
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
        margin: 0 0 .35rem 0;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
    }

    .hero p {
        color: rgba(255,255,255,.86);
        margin: 0;
        font-size: 1rem;
    }

    .eyebrow {
        color: #86e8d2;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    .soft-card {
        background: rgba(255,255,255,.94);
        border: 1px solid rgba(11,31,51,.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(20,42,65,.06);
        margin-bottom: .7rem;
    }

    .result-card {
        border-radius: 22px;
        padding: 1.4rem 1.5rem;
        margin-top: .8rem;
        box-shadow: 0 12px 30px rgba(20,42,65,.10);
    }

    .result-low {
        background: linear-gradient(135deg, #eafaf5 0%, #f7fffc 100%);
        border: 1px solid #bdebdc;
    }

    .result-medium {
        background: linear-gradient(135deg, #fff8e6 0%, #fffdf5 100%);
        border: 1px solid #f1d995;
    }

    .result-high {
        background: linear-gradient(135deg, #fff0f0 0%, #fff9f9 100%);
        border: 1px solid #f0b8b8;
    }

    .risk-number {
        font-size: 2.7rem;
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.04em;
        color: #0b1f33;
        margin: .35rem 0 .55rem 0;
    }

    .risk-label {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0b1f33;
        margin-bottom: .25rem;
    }

    .small-muted {
        color: #5e6b78;
        font-size: .88rem;
    }

    .section-title {
        color: #0b1f33;
        font-size: 1.25rem;
        font-weight: 800;
        margin: .3rem 0 .8rem 0;
    }

    .footer-note {
        color: #687887;
        font-size: .82rem;
        margin-top: 1.25rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 3rem;
        font-weight: 800;
        border: 0;
        background: linear-gradient(90deg, #1261a0 0%, #13a88f 100%);
        color: white;
        box-shadow: 0 8px 18px rgba(18,97,160,.18);
    }

    div.stButton > button:hover {
        color: white;
        border: 0;
        filter: brightness(1.04);
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(11,31,51,.07);
        padding: .85rem 1rem;
        border-radius: 16px;
        box-shadow: 0 7px 18px rgba(20,42,65,.05);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error("Required model files are missing. Keep the .joblib and metadata files beside streamlit_app.py.")
    st.stop()

model = load_model()
metadata = load_metadata()


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 🏥 SmartCare AI")
    st.caption("Appointment No-show Decision Support")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["Predict"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    # st.markdown("**Model**")
    # st.write(metadata.get("model_name", "Random Forest"))
    # st.caption("Option A • Binary Classification")

    st.markdown("---")
    # st.caption("Academic prototype — not a clinical decision system.")


# ---------------- Header ----------------
st.markdown("""
<div class="hero">
    <h1>Appointment No-show Predictor</h1>
</div>
""", unsafe_allow_html=True)


# ---------------- Prediction page ----------------
if page == "Predict":
    st.markdown('<div class="section-title">Patient & appointment details</div>', unsafe_allow_html=True)

    with st.form("prediction_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### Patient profile")
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                help="Patient age in years."
            )
            gender = st.selectbox("Gender", ["Female", "Male"])
            blood_group = st.selectbox(
                "Blood group",
                ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
            )

        with c2:
            st.markdown("#### Clinical context")
            department = st.selectbox(
                "Department",
                [
                    "Cardiology",
                    "General Medicine",
                    "Laboratory Services",
                    "Neurology",
                    "Orthopedics",
                    "Pediatrics",
                    "Radiology",
                ]
            )
            diagnosis = st.selectbox(
                "Diagnosis",
                [
                    "Asthma",
                    "Back Pain",
                    "Chest Pain",
                    "Diabetes",
                    "Fever",
                    "Fracture",
                    "Hypertension",
                    "Kidney Infection",
                    "Migraine",
                    "Pneumonia",
                ]
            )
            appointment_date = st.date_input(
                "Appointment date",
                value=date.today()
            )

        with c3:
            st.markdown("#### Appointment history")
            waiting_days = st.number_input(
                "Waiting days",
                min_value=0,
                max_value=365,
                value=10,
                step=1,
                help="Number of days between booking and the appointment."
            )
            previous_appointments = st.number_input(
                "Previous appointments",
                min_value=0,
                max_value=100,
                value=2,
                step=1,
            )
            missed_previous_appointments = st.number_input(
                "Previously missed appointments",
                min_value=0,
                max_value=100,
                value=0,
                step=1,
            )

        st.markdown("")
        submitted = st.form_submit_button("✨ Predict no-show risk")

    if submitted:
        if missed_previous_appointments > previous_appointments:
            st.error("Previously missed appointments cannot be greater than previous appointments.")
        else:
            appointment_month = appointment_date.month
            appointment_dayofweek = appointment_date.weekday()
            is_weekend = int(appointment_dayofweek in [5, 6])

            # Keep exactly the same formula used in the training notebook
            missed_appointment_rate = min(
                missed_previous_appointments / (previous_appointments + 1),
                1.0
            )

            patient = pd.DataFrame([{
                "age": age,
                "gender": gender,
                "blood_group": blood_group,
                "department": department,
                "diagnosis": diagnosis,
                "waiting_days": waiting_days,
                "previous_appointments": previous_appointments,
                "missed_previous_appointments": missed_previous_appointments,
                "appointment_month": appointment_month,
                "appointment_dayofweek": appointment_dayofweek,
                "is_weekend": is_weekend,
                "missed_appointment_rate": missed_appointment_rate,
            }])

            prediction = int(model.predict(patient)[0])
            probability = float(model.predict_proba(patient)[0, 1])

            if probability < 0.35:
                risk_label = "Low no-show risk"
                risk_class = "result-low"
                icon = "✅"
                action = "Standard appointment reminder is likely sufficient."
            elif probability < 0.60:
                risk_label = "Moderate no-show risk"
                risk_class = "result-medium"
                icon = "⚠️"
                action = "Consider an additional reminder or confirmation message."
            else:
                risk_label = "High no-show risk"
                risk_class = "result-high"
                icon = "🔔"
                action = "Consider proactive confirmation or scheduling support."

            st.markdown(
                f"""
                <div class="result-card {risk_class}">
                    <div class="risk-label">{icon} {risk_label}</div>
                    <div class="risk-number">{probability:.1%}</div>
                    <div class="small-muted">Predicted probability of missing the scheduled appointment</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(min(max(probability, 0.0), 1.0))

            r1, r2, r3 = st.columns(3)
            r1.metric("Model output", "No Show" if prediction == 1 else "Attended")
            r2.metric("No-show probability", f"{probability:.1%}")
            r3.metric("Decision threshold", "50%")

            # st.info(f"**Suggested operational action:** {action}")

            # with st.expander("See engineered model inputs"):
            #     display_patient = patient.copy()
            #     display_patient["appointment_day_name"] = appointment_date.strftime("%A")
            #     display_patient["appointment_date"] = appointment_date.isoformat()
            #     st.dataframe(display_patient, use_container_width=True)

            # st.caption(
            #     "This prototype is for hospital operational decision support. "
            #     "It should not be used to deny, delay, or restrict patient care."
            # )

    else:
        st.markdown("""
        <div class="soft-card" style="display:none">
            <b>How to use:</b> Enter the patient and appointment information above, then select
            <b>Predict no-show risk</b>. The application applies the same preprocessing and feature
            engineering pipeline used when the model was trained.
        </div>
        """, unsafe_allow_html=True)


# ---------------- Model insights page ----------------
elif page == "Model Insights":
    st.markdown('<div class="section-title">Model evaluation</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metadata['accuracy']:.1%}")
    m2.metric("Precision", f"{metadata['precision']:.1%}")
    m3.metric("Recall", f"{metadata['recall']:.1%}")
    m4.metric("F1 Score", f"{metadata['f1_score']:.1%}")
    m5.metric("ROC-AUC", f"{metadata['roc_auc']:.3f}")

    st.markdown("")
    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("#### Confusion matrix")
        cm = metadata["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Attended", "Actual: No Show"],
            columns=["Predicted: Attended", "Predicted: No Show"],
        )
        st.dataframe(cm_df, use_container_width=True)

        st.caption(
            f"Evaluation used {metadata['test_records']} held-out records from a "
            f"{metadata['dataset_records']}-record SmartCare dataset."
        )

    with right:
        st.markdown("#### Top feature importance")
        imp = pd.DataFrame(metadata["feature_importance"])
        imp = imp.sort_values("importance", ascending=False).head(8).set_index("feature")
        st.bar_chart(imp["importance"])

    st.markdown("""
    <div class="soft-card">
        <b>Interpretation:</b> The model is a Random Forest classifier. Feature importance indicates
        which inputs affect predictive performance most strongly, but importance does not prove
        causation. Accuracy alone should not be used to judge a healthcare-support model; recall,
        precision, F1 and ROC-AUC should be considered together.
    </div>
    """, unsafe_allow_html=True)

    st.warning(
        "This coursework dataset contains synthetic records. The model is suitable for an academic "
        "prototype, not for real clinical deployment without external validation, governance, "
        "fairness testing, privacy controls, and ongoing monitoring."
    )


# ---------------- About page ----------------
else:
    st.markdown('<div class="section-title">About this prototype</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="soft-card">
        <h3 style="margin-top:0;">Option A — Appointment Prediction</h3>
        <p>
        This application predicts whether a patient is likely to miss a scheduled hospital appointment.
        The target variable is <code>no_show</code>, making this a binary-classification problem.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("#### Model inputs")
        st.markdown("""
        - Age, gender and blood group
        - Department and diagnosis
        - Waiting days
        - Previous appointments
        - Previously missed appointments
        - Appointment month and weekday
        - Weekend indicator
        - Engineered missed-appointment rate
        """)

    with a2:
        st.markdown("#### Design decisions")
        st.markdown("""
        - `appointment_status` is excluded to prevent target leakage.
        - IDs are excluded because they are identifiers.
        - Post-appointment billing/admission variables are excluded from prediction.
        - Preprocessing is stored inside the Scikit-Learn pipeline.
        - Unknown categorical values are safely ignored by one-hot encoding.
        """)

    st.markdown("#### Responsible use")
    st.write(
        "The prediction is intended to help with reminder and scheduling workflows. "
        "A no-show prediction must not be treated as a clinical judgment or used to refuse care."
    )

# st.markdown(
#     '<div class="footer-note">SmartCare AI • CCS3440 Artificial Intelligence Coursework • Streamlit Prototype</div>',
#     unsafe_allow_html=True
# )
