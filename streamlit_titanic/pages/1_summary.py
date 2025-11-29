import streamlit as st
import pandas as pd

st.title("📄 Dataset Summary")

@st.cache_data
def load_data(path: str = "titanic.csv"):
    df = pd.read_csv(path)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'titanic.csv' file not found. Please keep titanic.csv in the root folder.")
    st.stop()
except Exception as e:
    st.error(f"Error while reading titanic.csv: {e}")
    st.stop()

st.subheader("🔹 First 10 Rows")
st.dataframe(df.head(10))

st.subheader("🔹 Shape of Dataset")
st.write(df.shape)

st.subheader("🔹 Column Info (dtypes)")
st.write(df.dtypes)

st.subheader("🔹 Summary Statistics (numeric)")
st.write(df.describe())

st.subheader("🔹 Summary Statistics (all columns)")
st.write(df.describe(include='all'))
