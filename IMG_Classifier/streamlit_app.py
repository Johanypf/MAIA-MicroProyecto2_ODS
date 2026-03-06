#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

import plotly.express as px

from src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Prediccion de ODS", page_icon=":earth_asia:"
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
    X, y_pred, y_scores, classes = ctrl.predict(input_text)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.caption("🧠 Your Prediction")

        if y_pred is not None:
            st.success(f"Predicted class: {y_pred}")
            y_scores = np.array(y_scores)
            if y_scores.ndim == 1:
                y_scores = y_scores.reshape(1, -1)

            exp_scores = np.exp(y_scores)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            st.caption("📊 Class Probabilities")
            
            data = pd.DataFrame({
                "ODS": classes,
                "valor": probs[0].flatten()
            })

        else:
            st.error("Prediction failed")
    colores = [
    "#4E79A7",  "#F28E2B",  "#E15759",  "#76B7B2",  
    "#59A14F", "#EDC948",  "#B07AA1",  "#FF9DA7",  "#9C755F", 
    "#BAB0AC",  "#2F4B7C", "#A05195", "#D45087",  "#F95D6A",  
    "#FF7C43", "#FFA600"  ]

    fig = px.bar(
                data,
                x="ODS",
                y="valor",
                title="Probabilidades de Predicción por Categoría",
                text=data["valor"].apply(lambda x: f"{x*100:.2f}%"))
    
    fig.update_traces(marker_color=colores[:len(data)])
    fig.update_traces(textposition="outside", textfont_size=12)

    fig.update_layout(showlegend=True)
    fig.update_xaxes(tickmode="linear")
    st.plotly_chart(fig)
