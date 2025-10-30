import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import os

st.title("De Titanic case")

st.write(
    "Welkom op deze Streamlit-pagina! Hier nemen we een kijkje naar de beroemde Titanic-dataset, "
    "bekend van de Kaggle-competitie: "
    "[Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/overview). "
    "In deze opdracht is de analyse tweemaal uitgevoerd: een eerste versie en een verbeterde, "
    "uitgebreidere versie waarin de resultaten verder zijn geoptimaliseerd."
)
st.markdown("---")

col1, col2 = st.columns([1, 1.2])
with col2:
    st.image("bestanden/titanic.png", use_container_width=True)
with col1:
    st.markdown("""
    <div style="
        background-color: var(--background-color-secondary);
        border-left: 6px solid #4CAF50;
        padding: 1em;
        border-radius: 6px;
        margin-top: 0.5em;
    ">
    <b>Over de Titanic</b><br>
    De RMS <i>Titanic</i> was een luxueus passagiersschip dat in april 1912 zonk tijdens haar eerste reis van Southampton naar New York. 
    Van de ruim 2200 passagiers en bemanningsleden kwamen meer dan 1500 mensen om het leven. 
    De ramp werd wereldberoemd door het tekort aan reddingsboten en de sterke sociale verschillen aan boord, 
    wat het een klassiek voorbeeld maakt voor data-analyse over overleving en menselijke factoren.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.write(
    "De dataset bevat informatie over de passagiers van de Titanic en of zij de ramp hebben overleefd. "
    "Hieronder is een kaart te zien die toont in welke havens de Titanic haar passagiers aan boord nam."
)

path = os.path.join("bestanden", "train.csv")
df = pd.read_csv(path)

df = df[df["Embarked"].notna()]

port_coords = {
    "S": {"lat": 50.903, "lon": -1.404, "name": "Southampton"},
    "C": {"lat": 49.633, "lon": -1.616, "name": "Cherbourg"},
    "Q": {"lat": 51.85, "lon": -8.30, "name": "Queenstown"}
}

df["lat"] = df["Embarked"].map(lambda x: port_coords[x]["lat"])
df["lon"] = df["Embarked"].map(lambda x: port_coords[x]["lon"])
df["port name"] = df["Embarked"].map(lambda x: port_coords[x]["name"])

agg_df = df.groupby("Embarked").agg(
    most_frequent_class=("Pclass", lambda x: x.mode()[0]),
    most_frequent_sex=("Sex", lambda x: x.mode()[0]),
    avg_age=("Age", "mean"),
    avg_fare=("Fare", "mean"),
    survival_rate=("Survived", "mean"),
    passenger_count=("PassengerId", "count")
).reset_index()

agg_df["lat"] = agg_df["Embarked"].map(lambda x: port_coords[x]["lat"])
agg_df["lon"] = agg_df["Embarked"].map(lambda x: port_coords[x]["lon"])
agg_df["port name"] = agg_df["Embarked"].map(lambda x: port_coords[x]["name"])

m = folium.Map(location=[50.3755, -4.1427], zoom_start=6, tiles="OpenStreetMap") # Plymouth

for _, row in agg_df.iterrows():
    popup_html = f"""
    <b>{row['port name']}</b><br>
    Passenger Count: {row['passenger_count']}<br>
    Most Frequent Class: {row['most_frequent_class']}<br>
    Most Frequent Sex: {row['most_frequent_sex']}<br>
    Average Age: {row['avg_age']:.1f}<br>
    Average Fare: ${row['avg_fare']:.2f}<br>
    Survival Rate: {row['survival_rate']*100:.1f}%
    """
    folium.Marker(
        location=[row["lat"], row["lon"]],
        icon=folium.Icon(color="blue", icon="info-sign"),
        popup=folium.Popup(popup_html, max_width=300)
    ).add_to(m)

st_folium(m, width=700, height=500)

st.markdown("---")






