import streamlit as st
import pandas as pd
import pickle

st.title("Titanic Passenger Survival Prediction")
st.write("Enter the passenger details to predict survival.")

# Load the trained model
with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input fields
pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Female", "Male"])
sibling = st.selectbox("Number of Siblings", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox(
    "Port of Embarkation",
    ["Southampton", "Cherbourg", "Queenstown"]
)

# Convert inputs using the notebook's label-encoding order
pclass_map = {"First": 0, "Second": 1, "Third": 2}
gender_map = {"Female": 0, "Male": 1}
sibling_map = {"One": 0, "Three": 1, "Two": 2, "Zero": 3}
embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}

if st.button("Predict Survival"):
    input_data = pd.DataFrame([[
        pclass_map[pclass],
        gender_map[gender],
        sibling_map[sibling],
        embarked_map[embarked]
    ]], columns=[
        "PClass", "Gender", "Sibling", "Embarked"
    ])

    prediction = model.predict(input_data)[0]

    # The notebook encodes survived labels: No = 0, Yes = 1
    if prediction == 1:
        st.success("Prediction: The passenger is predicted to survive.")
    else:
        st.warning("Prediction: The passenger is predicted not to survive.")

    st.caption("Educational machine-learning project; predictions are not guarantees.")