import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Titanic modellen", layout="centered")

st.title("Titanic Model Accuracy Dashboard")

path = os.path.join("bestanden", "Titanic_Model_Combination_Accuracies.csv")
df = pd.read_csv(path)

st.write("Naast een np.where is ook getest met verschillende modellen van machine learning. Deze modellen hebben ieder een andere manier hoe ze een gok maken hoe en wat gecorreleerd is met elkaar en op welke manier. Elk model is met alle mogelijke combinaties van de kolommen getrained. De kolommen zijn aangepast om alle NaN waardes eruit te halen:")
st.write({"Male": 0, "Female": 1, "S": 0, "Q": 1, "C": 2, "Age": "mediaan (28)", "Embarked": "meest voorkomend (S)", "Fare": "mediaan (14.4542)"})
st.write("Hieronder is een barplot weergegeven met de keuze om elk model wel of niet mee te nemen:")

st.subheader("Modellen & kolommen om wel/niet mee te nemen")

col1, col2 = st.columns([1,1])

with col1:
    all_models = sorted(df["Model"].unique())
    selected_models = st.multiselect(
        "Modellen om wel/niet toe te voegen:",
        options=all_models,
        default=all_models,
        key="models"
    )

with col2:
    st.markdown("Kolommen om wel/niet toe te voegen:")
    all_features = sorted(set(",".join(df["Features"].values).replace(" ", "").split(",")))
    include_features = []
    for feature in all_features:
        checked = True if feature == "Sex" else False
        if st.checkbox(feature, value=checked, key=f"feature_{feature}"):
            include_features.append(feature)

filtered_df = df[df["Model"].isin(selected_models)]
if include_features:
    filtered_df = filtered_df[
        filtered_df["Features"].apply(lambda f: set(f.replace(" ", "").split(",")) == set(include_features))
    ]
if not include_features:
    st.warning("Geen kolommen geselecteerd, kies minimaal 1 kolom.")
    st.stop()
if filtered_df.empty:
    st.warning("Geen modellen geselecteerd, kies minimaal 1 model.")
    st.stop()

overall_best = df.groupby("Model")["Accuracy"].max().reset_index(name="Overall Best Accuracy")
filtered_best = filtered_df.groupby("Model")["Accuracy"].mean().reset_index(name="Filtered Best Accuracy")
summary = pd.merge(overall_best, filtered_best, on="Model", how="left").fillna(0)
summary = summary.sort_values(by="Overall Best Accuracy", ascending=False)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=summary["Model"],
    y=summary["Overall Best Accuracy"],
    text=summary["Hoogst mogelijke nauwkeurigheid"].apply(lambda x: f"{x:.3f}"),
    textposition="outside",
    marker=dict(color="lightgray"),
    name="Overall Best Accuracy",
    opacity=0.6
))

fig.add_trace(go.Bar(
    x=summary["Model"],
    y=summary["Filtered Best Accuracy"],
    text=summary["Nauwkeurigheid geselecteerde kolommen"].apply(lambda x: f"{x:.3f}"),
    textposition="outside",
    marker=dict(color="royalblue"),
    name="Best Accuracy"
))

fig.update_layout(
    title="Vergelijking model nauwkeurigheid",
    xaxis_title="Model",
    yaxis_title="Nauwkeurigheid",
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    barmode="overlay",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("Alle geselecteerde modellen en kolommen zijn hieronder ook in een tabel weergegeven:")

st.dataframe(filtered_df.sort_values(by="Accuracy", ascending=False), use_container_width=True)
