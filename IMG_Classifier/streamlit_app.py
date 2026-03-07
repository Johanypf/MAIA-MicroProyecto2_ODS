#  We ensure proper path handling in Python
import Definitions
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go

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

    

    if y_pred is not None:

        st.success(f"Predicted class: {y_pred}")

        y_scores_flat = np.array(y_scores).flatten()
        y_scores_stable = y_scores_flat - np.max(y_scores_flat)
        exp_scores = np.exp(y_scores_stable)
        probs = exp_scores / np.sum(exp_scores)

        # Convertimos a listas puras de Python (evita errores de tipos en Cloud)
        eje_x = [f"ODS {int(c)}" for c in classes]
        eje_y = [float(p) for p in probs]
        textos = [f"{p*100:.1f}%" for p in probs]

        # 2. Construcción manual de la gráfica (graph_objects)
        fig = go.Figure(data=[
            go.Bar(
                x=eje_x,
                y=eje_y,
                text=textos,
                textposition='outside',
                marker_color=[
                    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", 
                    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC", 
                    "#2F4B7C", "#A05195", "#D45087", "#F95D6A", "#FF7C43", "#FFA600"
                ][:len(eje_x)]
            )
        ])

        # 3. Ajustes de diseño obligatorios para el servidor
        fig.update_layout(
            title="Probabilidades de Predicción",
            xaxis=dict(
                type='category',      # Evita que intente hacer una línea numérica
                tickangle=-45
            ),
            yaxis=dict(
                type='linear',        # FUERZA la escala matemática
                range=[0, max(eje_y) * 1.2], # Ajusta el techo al valor más alto + 20%
                tickformat='.0%',
                showgrid=True
            ),
            margin=dict(l=20, r=20, t=40, b=100),
            height=500,
            showlegend=False
        )

        # 4. Renderizado con limpieza de configuración
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


        

    else:
        st.error("Prediction failed")
        
    