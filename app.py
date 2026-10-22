# EduMetrics: College Placement & Salary Analytics
# Beginner-friendly Streamlit project for Data Science coursework.
# The bundled CSV contains synthetic demo records, not real placement statistics.

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------- Page and design -------------------------------
st.set_page_config(page_title="EduMetrics | Placement Analytics", page_icon="🎓", layout="wide")
DATA_FILE = Path(__file__).parent / "data" / "student_placement_sample.csv"
FEATURES = ["cgpa", "internships", "projects", "communication_score", "aptitude_score", "technical_score"]
REQUIRED = FEATURES + ["placed"]

st.markdown(
    """
    <style>
    /* Force a light, readable theme even when the user's Streamlit preference is dark. */
    .stApp {background:#f5f7fb; color:#1e293b;}
    [data-testid="stMain"] {color:#1e293b;}
    [data-testid="stSidebar"] {background:#101b35;}
    [data-testid="stSidebar"] * {color:#f4f7ff;}

    /* Main-page text was inheriting white from dark mode; set explicit readable colors. */
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] li,
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {
        color:#1e293b !important;
    }

    /* Preserve the hero's white text over its dark gradient. */
    .hero {padding:25px 28px; border-radius:20px; margin-bottom:20px; color:white !important;
           background:linear-gradient(115deg,#172554,#4f46e5 60%,#0f766e);
           box-shadow:0 12px 28px rgba(30,41,59,.13);}
    .hero h1, .hero * {color:white !important;}
    .hero p {color:#e0e7ff !important; margin:0; font-size:15px;}

    /* Keep tabs readable in both browser themes. */
    [data-testid="stTabs"] button,
    [data-testid="stTabs"] button p,
    [data-baseweb="tab"] {color:#334155 !important;}
    [data-testid="stTabs"] button[aria-selected="true"],
    [data-testid="stTabs"] button[aria-selected="true"] p {color:#e11d48 !important;}

    div[data-testid="stMetric"] {background:white; padding:16px; border-radius:14px;
           border:1px solid #e5eaf3; box-shadow:0 4px 14px rgba(15,23,42,.04);}
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {color:#64748b !important;}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {color:#172554 !important;}
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- Data --------------------------------------
def load_data(uploaded_file):
    """Load the included sample CSV or a user-uploaded CSV."""
    data = pd.read_csv(uploaded_file) if uploaded_file else pd.read_csv(DATA_FILE)
    data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    missing = [column for column in REQUIRED if column not in data.columns]
    if missing:
        st.error("CSV is missing required columns: " + ", ".join(missing))
        st.caption("See README.md or use the included sample CSV as a template.")
        st.stop()

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "salary_lpa" not in data.columns:
        data["salary_lpa"] = np.nan
    data["salary_lpa"] = pd.to_numeric(data["salary_lpa"], errors="coerce")

    # Convert common placement labels to a consistent Yes/No format.
    labels = {
        "yes": "Yes", "placed": "Yes", "1": "Yes", "true": "Yes",
        "no": "No", "not placed": "No", "not_placed": "No", "0": "No", "false": "No",
    }
    data["placed"] = data["placed"].astype(str).str.strip().str.lower().map(labels)
    if "branch" not in data.columns:
        data["branch"] = "Not specified"
    data["branch"] = data["branch"].fillna("Not specified").astype(str)
    if "student_id" not in data.columns:
        data["student_id"] = [f"STU{i + 1:04d}" for i in range(len(data))]
    return data.dropna(subset=FEATURES, how="all").copy()


def train_models(data):
    """Train logistic regression for placement and linear regression for salary."""
    result = {"placement_model": None, "placement_accuracy": None,
              "salary_model": None, "salary_mae": None, "salary_r2": None}

    # Placement model: Yes/No is a classification problem.
    labelled = data.dropna(subset=["placed"])
    if len(labelled) >= 10 and labelled["placed"].nunique() == 2 and labelled["placed"].value_counts().min() >= 2:
        x_train, x_test, y_train, y_test = train_test_split(
            labelled[FEATURES], labelled["placed"], test_size=0.2,
            random_state=42, stratify=labelled["placed"]
        )
        placement_model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)
        )
        placement_model.fit(x_train, y_train)
        result["placement_model"] = placement_model
        result["placement_accuracy"] = accuracy_score(y_test, placement_model.predict(x_test))

    # Salary model: use only placed students who have a positive salary recorded.
    salary_data = data[(data["placed"] == "Yes") & (data["salary_lpa"] > 0)].dropna(subset=["salary_lpa"])
    if len(salary_data) >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            salary_data[FEATURES], salary_data["salary_lpa"], test_size=0.2, random_state=42
        )
        salary_model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
        )
        salary_model.fit(x_train, y_train)
        predictions = salary_model.predict(x_test)
        result["salary_model"] = salary_model
        result["salary_mae"] = mean_absolute_error(y_test, predictions)
        result["salary_r2"] = r2_score(y_test, predictions) if len(y_test) > 1 else None
    return result


# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.markdown("# 🎓 EduMetrics")
    st.caption("COLLEGE PLACEMENT ANALYTICS")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a student CSV", type=["csv"])
    threshold = st.slider("Placement probability threshold", 0.30, 0.80, 0.50, 0.05)
    st.markdown("---")
    st.caption("Data Science for Engineers · Educational project")


data = load_data(uploaded_file)
models = train_models(data)
known = data.dropna(subset=["placed"])
placed = data[data["placed"] == "Yes"]
salaries = placed.loc[placed["salary_lpa"] > 0, "salary_lpa"].dropna()
placement_rate = known["placed"].eq("Yes").mean() * 100 if not known.empty else 0
average_cgpa = data["cgpa"].mean()
average_salary = salaries.mean() if not salaries.empty else None

st.markdown(
    "<div class='hero'><h1>Placement & Salary Analytics</h1>"
    "<p>Explore student outcomes, understand placement trends, and try simple ML predictions.</p></div>",
    unsafe_allow_html=True,
)
if uploaded_file is None:
    st.info("Demo mode: the included dataset is synthetic and does not represent actual college placements.")

# The UI has three sections; the separate Viva/Methodology screen has been removed.
overview_tab, data_tab, predict_tab = st.tabs(["📊 Overview", "🔎 Explore Data", "🎯 Predict"])


# -------------------------------- Overview ---------------------------------
with overview_tab:
    st.subheader("Placement overview")
    a, b, c, d = st.columns(4)
    a.metric("Students", f"{len(data):,}")
    b.metric("Placement rate", f"{placement_rate:.1f}%")
    c.metric("Average CGPA", f"{average_cgpa:.2f}" if pd.notna(average_cgpa) else "—")
    d.metric("Average salary", f"₹{average_salary:.2f} LPA" if average_salary is not None else "—")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Placement by branch")
        branch_chart = known.groupby(["branch", "placed"]).size().reset_index(name="Students")
        branch_chart["Outcome"] = branch_chart["placed"].map({"Yes": "Placed", "No": "Not placed"})
        if not branch_chart.empty:
            fig = px.bar(branch_chart, x="branch", y="Students", color="Outcome", barmode="group",
                         color_discrete_map={"Placed": "#0f9f8f", "Not placed": "#818cf8"})
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No placement outcomes are available in this dataset.")

    with right:
        st.markdown("#### CGPA vs technical score")
        scatter = data.dropna(subset=["cgpa", "technical_score"]).copy()
        scatter["Outcome"] = scatter["placed"].map({"Yes": "Placed", "No": "Not placed"}).fillna("Unknown")
        fig = px.scatter(scatter, x="cgpa", y="technical_score", color="Outcome",
                         color_discrete_map={"Placed": "#0f9f8f", "Not placed": "#818cf8", "Unknown": "#94a3b8"},
                         labels={"cgpa": "CGPA", "technical_score": "Technical score"})
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Salary distribution")
        if not salaries.empty:
            fig = px.histogram(pd.DataFrame({"salary_lpa": salaries}), x="salary_lpa", nbins=18,
                               labels={"salary_lpa": "Salary (LPA)"}, color_discrete_sequence=["#6366f1"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Salary chart appears when valid salary values are available.")

    with right:
        st.markdown("#### Model performance")
        m1, m2 = st.columns(2)
        m1.metric("Placement accuracy", f"{models['placement_accuracy'] * 100:.1f}%" if models["placement_accuracy"] is not None else "—")
        m2.metric("Salary MAE", f"₹{models['salary_mae']:.2f} LPA" if models["salary_mae"] is not None else "—")
        st.caption("Metrics come from a held-out test set. They depend on data quality and size.")


# ------------------------------ Explore data -------------------------------
with data_tab:
    st.subheader("Explore student records")
    first, second = st.columns(2)
    branches = sorted(data["branch"].dropna().astype(str).unique())
    with first:
        selected_branches = st.multiselect("Branch", branches, default=branches)
    with second:
        outcome = st.selectbox("Placement outcome", ["All", "Placed", "Not placed", "Unknown"])

    filtered = data[data["branch"].astype(str).isin(selected_branches)].copy()
    if outcome == "Placed":
        filtered = filtered[filtered["placed"] == "Yes"]
    elif outcome == "Not placed":
        filtered = filtered[filtered["placed"] == "No"]
    elif outcome == "Unknown":
        filtered = filtered[filtered["placed"].isna()]

    display = filtered.copy()
    display["placed"] = display["placed"].map({"Yes": "Placed", "No": "Not placed"})
    st.caption(f"Showing {len(display):,} of {len(data):,} students")
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="edumetrics_students.csv", mime="text/csv")

    st.markdown("#### Feature correlation")
    fig = px.imshow(data[FEATURES + ["salary_lpa"]].corr(numeric_only=True), text_auto=".2f",
                    color_continuous_scale="Blues", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------- Prediction -------------------------------
with predict_tab:
    st.subheader("Try a student profile")
    st.caption("Change the inputs to see what the trained models estimate.")
    left, right = st.columns(2)
    with left:
        cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
        internships = st.number_input("Internships", min_value=0, max_value=10, value=1)
        projects = st.number_input("Projects", min_value=0, max_value=10, value=3)
    with right:
        communication = st.slider("Communication score", 0, 100, 75)
        aptitude = st.slider("Aptitude score", 0, 100, 70)
        technical = st.slider("Technical score", 0, 100, 75)

    profile = pd.DataFrame([{
        "cgpa": cgpa, "internships": internships, "projects": projects,
        "communication_score": communication, "aptitude_score": aptitude,
        "technical_score": technical,
    }])
    st.markdown("---")
    first, second = st.columns(2)
    with first:
        st.markdown("#### Placement estimate")
        model = models["placement_model"]
        if model is not None:
            probabilities = model.predict_proba(profile)[0]
            classes = list(model.named_steps["logisticregression"].classes_)
            probability = float(probabilities[classes.index("Yes")])
            st.metric("Estimated placement probability", f"{probability * 100:.1f}%")
            st.progress(probability)
            if probability >= threshold:
                st.success("Prediction: Likely placed")
            else:
                st.warning("Prediction: Not likely placed")
        else:
            st.warning("Add more labelled records containing both Yes and No outcomes to train this model.")

    with second:
        st.markdown("#### Salary estimate")
        model = models["salary_model"]
        if model is not None:
            estimate = max(0, float(model.predict(profile)[0]))
            st.metric("Estimated package", f"₹{estimate:.2f} LPA")
            st.caption("The salary model learns only from placed students with recorded salaries.")
        else:
            st.warning("Add at least 10 placed students with valid salary values to train this model.")

    st.caption("Predictions are educational estimates, not guarantees of employment or salary.")

st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:12px;padding:24px 0 8px'>EduMetrics · College Placement & Salary Analytics</div>",
    unsafe_allow_html=True,
)
