import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

st.title("Compare Tafel Plots from Multiple Files")

st.info("Place all your CSV/XLSX data files in a single directory (same structure: 'Potential applied (V)' and 'WE(1).Current (A)')")

# Choose folder with multiple files
folder = st.text_input("Enter the path to the directory with your CSV/XLSX files", value=".")

if st.button("Plot all Tafel curves"):

    files = [f for f in os.listdir(folder) if f.lower().endswith(('.csv','.xlsx'))]

    if not files:
        st.warning("No CSV/XLSX files found in that folder.")
    else:
        area_cm2 = st.number_input('Sample surface area (cm², applied to all files)', min_value=1e-6, value=1.0, format="%.4f")
        tafel_start = st.number_input('Tafel fit region START (V)', value=-0.90)
        tafel_end   = st.number_input('Tafel fit region END (V)', value=-0.82)

        fig, ax = plt.subplots(figsize=(7, 5))
        for fname in files:
            try:
                if fname.lower().endswith('.csv'):
                    df = pd.read_csv(os.path.join(folder, fname))
                else:
                    df = pd.read_excel(os.path.join(folder, fname))
                pot_col = 'Potential applied (V)'
                cur_col = 'WE(1).Current (A)'
                if not all(x in df.columns for x in [pot_col, cur_col]):
                    st.warning(f"File {fname}: missing required columns, skipping.")
                    continue

                E = df[pot_col].astype(float).values
                I = df[cur_col].astype(float).values
                mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
                E = E[mask]
                I = I[mask]
                i = I / area_cm2

                # Mask for requested Tafel window
                tmask = (E >= tafel_start) & (E <= tafel_end)
                if np.sum(tmask) < 3:
                    st.warning(f"File {fname}: too few points in fit window for plot.")
                    continue

                ax.plot(E[tmask], np.log10(np.abs(i[tmask])), '-', label=fname)
            except Exception as err:
                st.warning(f"File {fname}: {err}")

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("log10(|i|) (A/cm²)")
        ax.set_title("Tafel Plots Overlay")
        ax.legend(fontsize=8)
        ax.grid(True)
        st.pyplot(fig)
