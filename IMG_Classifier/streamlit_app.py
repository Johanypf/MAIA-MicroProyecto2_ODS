#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Image Classifier", page_icon="❄️"
)

### My vars

ctrl = ModelController()

### My UI starting here

with st.form(key="my_form"):
    user_input = st.text_input(
        label="Enter your text"
    )

    submit_button = st.form_submit_button(label="Submit")

if submit_button and user_input:
    input_text, is_valid = ctrl.load_input_data(user_input)
    st.session_state["input_text"] = input_text if is_valid else None

input_text = st.session_state.get("input_text")

if input_text is not None:

    st.caption("✅ This is your input text")
    st.write(input_text)

    # Run prediction directly on the text
    X, y_pred = ctrl.predict(input_text)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.caption("🧠 Your Prediction")

        if y_pred is not None:
            st.success(f"Predicted class: {y_pred}")
        else:
            st.error("Prediction failed")
