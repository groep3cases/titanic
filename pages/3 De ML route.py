import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Titanic Model Accuracy Dashboard", layout="centered")

st.title("Titanic Model Accuracy Dashboard")

path = os.path.join("bestanden", "Titanic_Model_Combination_Accuracies.csv")
df = pd.read_csv(path)

st.subheader("Models & columns to include/exclude")

col1, col2 = st.columns([1,1])

with col1:
    all_models = sorted(df["Model"].unique())
    selected_models = st.multiselect(
        "Select models to include or exclude:",
        options=all_models,
        default=all_models,
        key="models"
    )

with col2:
    st.markdown("Select columns to include or exclude:")
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
    st.warning("No columns selected, please select columns.")
    st.stop()
if filtered_df.empty:
    st.warning("No columns selected, please select columns.")
    st.stop()

overall_best = df.groupby("Model")["Accuracy"].max().reset_index(name="Overall Best Accuracy")
filtered_best = filtered_df.groupby("Model")["Accuracy"].mean().reset_index(name="Filtered Best Accuracy")
summary = pd.merge(overall_best, filtered_best, on="Model", how="left").fillna(0)
summary = summary.sort_values(by="Overall Best Accuracy", ascending=False)

fig = go.Figure()

fig.add_trace(go.Bar(
    x=summary["Model"],
    y=summary["Overall Best Accuracy"],
    text=summary["Overall Best Accuracy"].apply(lambda x: f"{x:.3f}"),
    textposition="outside",
    marker=dict(color="lightgray"),
    name="Overall Best Accuracy",
    opacity=0.6
))

fig.add_trace(go.Bar(
    x=summary["Model"],
    y=summary["Filtered Best Accuracy"],
    text=summary["Filtered Best Accuracy"].apply(lambda x: f"{x:.3f}"),
    textposition="outside",
    marker=dict(color="royalblue"),
    name="Best Accuracy"
))

fig.update_layout(
    title=" Model Accuracy Comparison (Overall vs Selected Features)",
    xaxis_title="Model",
    yaxis_title="Accuracy",
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    barmode="overlay",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("Here, all selected models and selected columns, sorted by accuracy, are shown:")
st.dataframe(filtered_df.sort_values(by="Accuracy", ascending=False), use_container_width=True)