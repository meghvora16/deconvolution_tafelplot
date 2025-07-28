import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Overlay: Complete Tafel Plots from Multiple Files")

st.info("""
Upload multiple CSV/XLSX files. Each file must have columns:
**'Potential applied (V)'** and **'WE(1).Current (A)'**.
The app will plot log10(|i|) vs. E for ALL cleaned data in each file.
""")

uploaded_files = st.file_uploader(
    "Upload one or more CSV/XLSX files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

area_cm2 = st.number_input('Sample surface area (cm², for current density)', min_value=1e-6, value=1.0, format="%.4f")

if uploaded_files:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.get_cmap('tab10', max(10, len(uploaded_files)))
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            pot_col = 'Potential applied (V)'
            cur_col = 'WE(1).Current (A)'
            if not all(x in df.columns for x in [pot_col, cur_col]):
                st.warning(f"{uploaded_file.name}: missing required columns, skipping.")
                continue
            E = df[pot_col].astype(float).values
            I = df[cur_col].astype(float).values
            mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
            E = E[mask]
            I = I[mask]
            i = I / area_cm2
            if np.sum(mask) < 3:
                st.warning(f"{uploaded_file.name}: too few valid data points for plot, skipping.")
                continue
            ax.plot(E, np.log10(np.abs(i)), '-', label=uploaded_file.name, color=colors(idx))
        except Exception as err:
            st.warning(f"{uploaded_file.name}: {err}")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("log10(|i|) (A/cm²)")
    ax.set_title("Complete Tafel Plots Overlay")
    ax.legend(fontsize=8)
    ax.grid(True)
    st.pyplot(fig)
