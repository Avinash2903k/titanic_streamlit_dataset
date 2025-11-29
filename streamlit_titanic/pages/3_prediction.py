import streamlit as st
import pickle
import numpy as np

st.title("🤖 Titanic Survival Prediction")

# =========================
# Load trained model.pkl
# =========================
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ 'model.pkl' not found.\n\n👉 Please run `python train_model.py` once to create the model.")
        st.stop()
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.stop()

model = load_model()

st.markdown("Fill passenger details below and click **Predict**:")

# Inputs
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    age = st.slider("Age", 1, 100, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)

with col2:
    sex = st.radio("Gender", ["male", "female"])
    parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
    fare = st.slider("Fare Paid", 0.0, 600.0, 50.0)

# Gender encoding: same as train_model.py (male=0, female=1)
sex_val = 1 if sex == "female" else 0

if st.button("🔍 Predict Survival"):
    # Feature order MUST match training: [pclass, sex, age, sibsp, parch, fare]
    data = np.array([[pclass, sex_val, age, sibsp, parch, fare]])

    try:
        # Option 1: class prediction
        pred_class = model.predict(data)[0]

        # Option 2: probability
        try:
            prob_survive = model.predict_proba(data)[0][1]  # probability of class 1
        except AttributeError:
            # if model doesn't support predict_proba
            prob_survive = None

        # Show results
        if pred_class == 1:
            st.success("🌟 Prediction: Passenger **WOULD SURVIVE**")
        else:
            st.error("💀 Prediction: Passenger **WOULD NOT SURVIVE**")

        if prob_survive is not None:
            st.info(f"📊 Estimated survival probability: **{prob_survive * 100:.2f}%**")
    except Exception as e:
        st.error(f"Error while making prediction: {e}")
