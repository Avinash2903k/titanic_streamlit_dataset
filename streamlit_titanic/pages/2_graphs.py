# pages/2_graphs.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Visual Analysis", layout="wide", page_icon="📊")

# ---------- Load & normalize dataset ----------
@st.cache_data
def load_data(path="titanic.csv"):
    df = pd.read_csv(path)
    # Normalize column names: strip, lower
    df.columns = df.columns.str.strip().str.lower()
    # Try to rename common columns to standard names used below
    rename_map = {}
    # typical variations -> standard lower names
    if 'age' not in df.columns and 'age ' in df.columns:
        rename_map['age '] = 'age'
    # map common variants:
    for cand in df.columns:
        if cand.lower() in ['pclass','p_class','p class','class']:
            rename_map[cand] = 'pclass'
        if cand.lower() in ['sex','gender']:
            rename_map[cand] = 'sex'
        if cand.lower() in ['survived','survival','survive']:
            rename_map[cand] = 'survived'
        if cand.lower() in ['fare','ticket_fare','fare_amount']:
            rename_map[cand] = 'fare'
        if cand.lower() in ['sibsp','sib_sp','siblingsspouses']:
            rename_map[cand] = 'sibsp'
        if cand.lower() in ['parch','parentschildren','par_ch']:
            rename_map[cand] = 'parch'
    df = df.rename(columns=rename_map)
    # ensure required columns exist (safe defaults)
    for col in ['age','sex','pclass','survived','fare','sibsp','parch']:
        if col not in df.columns:
            df[col] = np.nan
    # basic cleanup
    # convert sex to lowercase strings
    df['sex'] = df['sex'].astype(str).str.lower().replace({'nan': np.nan})
    # ensure numeric columns are numeric
    for num in ['age','fare','pclass','survived','sibsp','parch']:
        if num in df.columns:
            df[num] = pd.to_numeric(df[num], errors='coerce')
    return df

try:
    df = load_data("titanic.csv")
except FileNotFoundError:
    st.error("titanic.csv not found in project folder. Please place titanic.csv in the project root.")
    st.stop()

st.title("📊 Titanic — Visual Analysis (All visuals in one page)")
st.markdown("Use the sidebar to filter data. Scroll down to see interactive plots.")

# Show detected columns (helpful)
with st.expander("Detected columns (for debugging)"):
    st.write(list(df.columns))

# ---------- Sidebar filters ----------
st.sidebar.header("Filters & Controls")

# handle if Age column is all NaN
age_min = int(df['age'].min(skipna=True)) if df['age'].notna().any() else 0
age_max = int(df['age'].max(skipna=True)) if df['age'].notna().any() else 100

pclass_options = sorted([int(x) for x in df['pclass'].dropna().unique() if not np.isnan(x)]) if df['pclass'].notna().any() else [1,2,3]
sex_options = sorted([s for s in df['sex'].dropna().unique()]) if df['sex'].notna().any() else ['male','female']

pclass_filter = st.sidebar.multiselect("Passenger Class", options=pclass_options, default=pclass_options)
gender_filter = st.sidebar.multiselect("Gender", options=sex_options, default=sex_options)
age_range = st.sidebar.slider("Age range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
fare_max = int(df['fare'].max(skipna=True)) if df['fare'].notna().any() else 1000
fare_range = st.sidebar.slider("Fare range", 0, fare_max, (0, fare_max))

# optional toggles
show_correlation = st.sidebar.checkbox("Show correlation heatmap", value=True)
show_sankey = st.sidebar.checkbox("Show Sankey diagram", value=True)

# apply filters safely
filtered = df.copy()
if 'pclass' in df.columns:
    filtered = filtered[filtered['pclass'].isin(pclass_filter)]
if 'sex' in df.columns:
    filtered = filtered[filtered['sex'].isin(gender_filter)]
if 'age' in df.columns and filtered['age'].notna().any():
    filtered = filtered[(filtered['age'] >= age_range[0]) & (filtered['age'] <= age_range[1])]
if 'fare' in df.columns and filtered['fare'].notna().any():
    filtered = filtered[(filtered['fare'] >= fare_range[0]) & (filtered['fare'] <= fare_range[1])]

# quick info
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric("Rows (filtered)", int(len(filtered)))
with col2:
    st.metric("Unique passengers", int(filtered.shape[0]))
with col3:
    alive = int(filtered['survived'].sum()) if filtered['survived'].notna().any() else 0
    st.metric("Survived (count)", alive)

# ---------- Grid layout for many visuals ----------
# We'll create multiple rows/columns for neat layout
st.markdown("---")
st.markdown("## Overview visuals")

# Row 1: Age distribution & Gender count
r1c1, r1c2 = st.columns(2)
with r1c1:
    st.subheader("🎂 Age Distribution")
    if filtered['age'].notna().any():
        fig = px.histogram(filtered, x="age", nbins=30, title="Age distribution", color_discrete_sequence=["#A3D2CA"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Age data available to plot histogram.")

with r1c2:
    st.subheader("👩‍🦰 Gender Counts")
    if filtered['sex'].notna().any():
        counts = filtered['sex'].value_counts().reset_index()
        counts.columns = ['gender','count']
        fig = px.bar(counts, x='gender', y='count', color='gender', color_discrete_sequence=["#FFB3C6","#A3D2CA"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Gender data available.")

# Row 2: Survival by gender & by class
r2c1, r2c2 = st.columns(2)
with r2c1:
    st.subheader("🚢 Survival by Gender")
    if filtered['sex'].notna().any() and filtered['survived'].notna().any():
        fig = px.histogram(filtered, x="sex", color="survived", barmode="group",
                           color_discrete_sequence=["#FFB3C6","#A3D2CA"],
                           labels={"sex":"Gender","survived":"Survived"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need both Gender and Survived data to show this plot.")

with r2c2:
    st.subheader("🏷 Survival by Class")
    if filtered['pclass'].notna().any() and filtered['survived'].notna().any():
        fig = px.histogram(filtered, x="pclass", color="survived", barmode="group",
                           labels={"pclass":"Passenger Class"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need Pclass & Survived columns to show this plot.")

# Row 3: Scatter Age vs Fare and Correlation heatmap
r3c1, r3c2 = st.columns([1.2, 0.8])
with r3c1:
    st.subheader("🎯 Age vs Fare (bubble)")
    if filtered['age'].notna().any() and filtered['fare'].notna().any():
        fig = px.scatter(filtered, x="age", y="fare", color="survived", size="fare",
                         hover_data=["sex","pclass"], color_continuous_scale="Mint")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need Age & Fare data for scatter plot.")

with r3c2:
    if show_correlation:
        st.subheader("🔥 Correlation Heatmap (numeric cols)")
        num_cols = [c for c in ['age','fare','sibsp','parch','pclass','survived'] if c in filtered.columns]
        if len(num_cols) > 1 and filtered[num_cols].dropna().shape[0] > 0:
            corr = filtered[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric data for correlation heatmap.")

# Row 4: Pie charts - class & survival
r4c1, r4c2 = st.columns(2)
with r4c1:
    st.subheader("🥧 Class Distribution")
    if filtered['pclass'].notna().any():
        fig = px.pie(filtered, names='pclass', title="Class split")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Pclass data.")

with r4c2:
    st.subheader("❤️ Survival Rate")
    if filtered['survived'].notna().any():
        fig = px.pie(filtered, names='survived', title="Survival split")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Survived data.")

# Row 5: Boxplots
st.markdown("---")
st.subheader("Boxplots & Distribution by group")
b1, b2 = st.columns(2)
with b1:
    st.markdown("### 📦 Fare by Class")
    if filtered['fare'].notna().any() and filtered['pclass'].notna().any():
        fig = px.box(filtered, x='pclass', y='fare', color='pclass', title="Fare distribution by class")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Fare/Pclass data for boxplot.")
with b2:
    st.markdown("### 🧊 Age by Gender")
    if filtered['age'].notna().any() and filtered['sex'].notna().any():
        fig = px.box(filtered, x='sex', y='age', color='sex', title="Age by gender")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Age/Gender data for boxplot.")

# Row 6: Violin + Fare boxplot (larger)
st.markdown("---")
st.subheader("Advanced distributions")
v1, v2 = st.columns([1,1])
with v1:
    st.markdown("### 🎻 Age Violin by Survival")
    if filtered['age'].notna().any() and filtered['survived'].notna().any():
        fig = px.violin(filtered, x='survived', y='age', color='survived', box=True, points='all')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for violin plot.")
with v2:
    st.markdown("### 📦 Fare distribution (box + points)")
    if filtered['fare'].notna().any():
        fig = px.histogram(filtered, x='fare', nbins=40, title="Fare histogram")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Fare data.")

# Row 7: Sankey diagram (Class -> Sex -> Survived)
st.markdown("---")
if show_sankey:
    st.subheader("🔗 Sankey: Class → Gender → Survived")
    try:
        df_s = filtered.copy()
        # Prepare nodes and links
        df_s['surv_label'] = df_s['survived'].map({1: 'Survived', 0: 'Died'}).fillna('Unknown')
        classes = sorted(df_s['pclass'].dropna().unique().tolist())
        genders = sorted(df_s['sex'].dropna().unique().tolist())
        survs = sorted(df_s['surv_label'].dropna().unique().tolist())

        labels = [f"Class {int(c)}" for c in classes] + genders + survs
        source = []
        target = []
        value = []

        # class -> gender
        for i, c in enumerate(classes):
            for j, g in enumerate(genders):
                cnt = int(len(df_s[(df_s['pclass'] == c) & (df_s['sex'] == g)]))
                source.append(i)
                target.append(len(classes) + j)
                value.append(cnt)
        # gender -> survived
        for j, g in enumerate(genders):
            for k, s in enumerate(survs):
                cnt = int(len(df_s[(df_s['sex'] == g) & (df_s['surv_label'] == s)]))
                source.append(len(classes) + j)
                target.append(len(classes) + len(genders) + k)
                value.append(cnt)

        if sum(value) == 0:
            st.info("Sankey: not enough data to draw flows.")
        else:
            fig = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15, thickness=20),
                                            link=dict(source=source, target=target, value=value))])
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Sankey error: {e}")

# Row 8: Survival probability curve by Age (simple logistic)
st.markdown("---")
st.subheader("📈 Survival probability (Age vs probability) — logistic fit")
try:
    df_ml = filtered.dropna(subset=['age','survived'])
    if df_ml.shape[0] >= 30:  # need enough rows to fit
        X = df_ml[['age']].values.reshape(-1,1)
        y = df_ml['survived'].values
        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y)
        ages = np.linspace(df_ml['age'].min(), df_ml['age'].max(), 200).reshape(-1,1)
        probs = lr.predict_proba(ages)[:,1]
        fig = px.line(x=ages.flatten(), y=probs, labels={'x':'Age','y':'Survival probability'},
                      title="Survival probability vs Age")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough rows to fit logistic model for survival probability (need ~30 rows).")
except Exception as e:
    st.error(f"Probability curve error: {e}")

st.markdown("---")
st.info("If any plot shows 'No data' or looks empty, please replace titanic.csv with the full Titanic dataset (Kaggle/datasciencedojo). Filters may reduce rows to very few values.")
st.caption("Need more plots or different groupings? Tell me which plot and I will add it.")