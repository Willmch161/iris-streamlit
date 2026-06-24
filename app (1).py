import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ── Load dataset & train model ────────────────────────────────────────────────
@st.cache_resource
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, iris.target_names

model, target_names = train_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌸 Iris Flower Classification")
st.write("Adjust the sliders below to input flower measurements, then click **Predict** to identify the species.")

st.sidebar.header("Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)",  2.0, 4.5, 3.4, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.7, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)",  0.1, 2.5, 1.3, 0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Predict", use_container_width=True):
    prediction    = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    species       = target_names[prediction]

    st.markdown("---")
    st.subheader("Prediction Result")

    color_map = {
        "setosa":     "🟣",
        "versicolor": "🔵",
        "virginica":  "🟢",
    }
    emoji = color_map.get(species, "🌸")
    st.success(f"{emoji}  Predicted species: **Iris {species.capitalize()}**")

    st.subheader("Prediction Probability")
    prob_df = pd.DataFrame({
        "Species":     [f"Iris {n.capitalize()}" for n in target_names],
        "Probability": [f"{p*100:.1f}%" for p in probabilities],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.bar_chart(
        pd.DataFrame({"Probability": probabilities}, index=[f"Iris {n.capitalize()}" for n in target_names])
    )

# ── Input summary ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Your Input")
st.table(pd.DataFrame({
    "Feature":       ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    "Value (cm)":    [sepal_length, sepal_width, petal_length, petal_width],
}))

# ── About ─────────────────────────────────────────────────────────────────────
with st.expander("About this app"):
    st.write("""
    This app uses a **Random Forest Classifier** trained on the classic Iris dataset
    (Fisher, 1936) to predict the species of an iris flower based on four measurements.

    **Species:**
    - 🟣 *Iris setosa*
    - 🔵 *Iris versicolor*
    - 🟢 *Iris virginica*

    **Model:** Random Forest (100 estimators)  
    **Dataset:** sklearn built-in Iris dataset
    """)
