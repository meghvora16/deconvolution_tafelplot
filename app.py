import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Overlay Tafel Plots from Multiple Files")

st.info("""
Upload multiple CSV/XLSX electrochemical data files.
Each must have columns: **'Potential applied (V)'** and **'WE(1).Current (A)'**.
Choose your region for the Tafel plot (potential window).
""")

uploaded_files = st.file_uploader(
    "Upload one or more CSV/XLSX files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

area_cm2 = st.number_input('Sample surface area (cm², for current density)', min_value=1e-6, value=1.0, format="%.4f")
tafel_start = st.number_input('Tafel region START (V)', value=-0.90)
tafel_end   = st.number_input('Tafel region END (V)', value=-0.82)

if uploaded_files:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.get_cmap('tab10', len(uploaded_files))
    added = 0
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
            tmask = (E >= tafel_start) & (E <= tafel_end)
            if np.sum(tmask) < 3:
                st.warning(f"{uploaded_file.name}: too few points in the Tafel region for plot.")
                continue
            ax.plot(
                E[tmask], np.log10(np.abs(i[tmask])),
                '-', label=uploaded_file.name, color=colors(idx)
            )
            added += 1
        except Exception as err:
            st.warning(f"{uploaded_file.name}: {err}")
    if added == 0:
        st.error("No valid Tafel curves to display.")
    else:
        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("log10(|i|) (A/cm²)")
        ax.set_title("Overlay of Tafel Regions")
        ax.legend(fontsize=8)
        ax.grid(True)
        st.pyplot(fig)
