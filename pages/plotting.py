import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

import source

def run_plotting():

    st.set_page_config(
        page_title="Визуализация полученных результатов",
        page_icon="🖥")
    
    st.sidebar.success(r"Данный раздел предназначен для виулизации полученных в разделе \app результатов.")
    
    st.write("### 1. Загрузка сохраненных прогнозных данных")

    st.write("Загрузите входные данные в формате `.csv` или `.xlsx`.")

    uploaded_prediction = st.file_uploader("Prediction", 
                                     type=['csv', 'xlsx'], 
                                     label_visibility='collapsed')
    data_pred = source.load_file_to_st(uploaded_prediction)

    st.write("Что предсказано?")

    what_to_predict = st.selectbox("Предсказано: ", 
                                    ("TC_par", 
                                    "TC_per",
                                    "VHC",
                                    "Anisotropy"))
    
    st.info(f"Конфигурация прогноза: {what_to_predict}")

    if what_to_predict == 'TC_par':
        text = r'TC_par, W/m/K'
    elif what_to_predict == 'TC_per':
        text = r'TC_per, W/m/K'
    elif what_to_predict == 'VHC':
        text = r'Cp MJ/m^3/K'
    elif what_to_predict == 'Anisotropy':
        text = r'K, c.u.'

    # !!!
    import plotly.graph_objects as go

    depth = data_pred.iloc[:, 0]
    val = data_pred.iloc[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val, y=depth, mode='lines'))

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        yaxis_title='Глубина, м',
        xaxis_title=text,
        font_size=10,
        width=500,
        height=1000
    )

    st.plotly_chart(fig)

if __name__ == "__main__":
    run_plotting()