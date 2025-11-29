import streamlit as st

st.set_page_config(
    page_title="Titanic — Data Analyzer",
    page_icon="🚢",
    layout="wide",
)

st.markdown("""
<div style='padding: 30px;'>

<h1 style='font-size: 42px; font-weight: 800; color:#3A405A; margin-bottom: 10px;'>
    Titanic — Data Analyzer
</h1>

<p style='font-size: 18px; color:#555; max-width: 800px; line-height: 1.6;'>
    An Interactive Dashboard for Titanic Dataset Exploration, Visualization and Survival Prediction.
</p>

<hr style='margin: 25px 0; border: none; border-top: 1px solid #DDD;'>

<h2 style='font-size: 28px; color:#6A5ACD; font-weight:700; margin-top: 20px;'>
    Explore the Sections:
</h2>

<ul style='font-size: 18px; color:#444; line-height:1.8;'>
    <li><b>Summary:</b> Dataset overview, missing values, stats.</li>
    <li><b>Graphs:</b> Age, Fare, Class, Gender, Survival patterns.</li>
    <li><b>Prediction:</b> Machine Learning model survival prediction.</li>
</ul>

</div>
""", unsafe_allow_html=True)
