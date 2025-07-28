import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

st.title("Visualize $E_{corr}$ Shift from Two Fitting Regions")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential, Current)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_ecorr_shift'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format."); st.stop()

    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    E = df[pot_col].values
    I = df[cur_col].values

    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]
    N = len(E_clean)

    st.markdown("#### Fitting region set **A** (Tafel & plateau):")
    tafel_min_A = st.number_input('Tafel A min (V)', value=float(np.percentile(E_clean, 5)), key='taAmin')
    tafel_max_A = st.number_input('Tafel A max (V)', value=float(np.percentile(E_clean, 25)), key='taAmax')
    plateau_min_A = st.number_input('Plateau A min (V)', value=float(np.percentile(E_clean, 75)), key='paAmin')
    plateau_max_A = st.number_input('Plateau A max (V)', value=float(np.percentile(E_clean, 98)), key='paAmax')

    st.markdown("#### Fitting region set **B** (Tafel & plateau):")
    tafel_min_B = st.number_input('Tafel B min (V)', value=float(np.percentile(E_clean, 10)), key='taBmin')
    tafel_max_B = st.number_input('Tafel B max (V)', value=float(np.percentile(E_clean, 30)), key='taBmax')
    plateau_min_B = st.number_input('Plateau B min (V)', value=float(np.percentile(E_clean, 80)), key='paBmin')
    plateau_max_B = st.number_input('Plateau B max (V)', value=float(np.percentile(E_clean, 99)), key='paBmax')

    # ---- Helper for any region
    def fit_tafel_and_plateau(E, I, tmin, tmax, pmin, pmax):
        tafel_mask = (E >= tmin) & (E <= tmax)
        plateau_mask = (E >= pmin) & (E <= pmax)
        E_tafel = E[tafel_mask]
        I_tafel = I[tafel_mask]
        logI_tafel = np.log10(np.abs(I_tafel)) if len(I_tafel) > 0 else None
        E_plateau = E[plateau_mask]
        I_plateau = np.abs(I[plateau_mask])
        # Linear fit
        slope, intercept, r, p, std = linregress(E_tafel, logI_tafel) if len(E_tafel) > 1 else (np.nan, np.nan, 0, 0, 0)
        tafel_slope = 1/slope if slope else np.nan
        I_lim = np.median(I_plateau) if len(I_plateau) > 0 else np.nan
        log_Il = np.log10(I_lim) if I_lim > 0 else np.nan
        E_corr = (log_Il - intercept) / slope if (slope and not np.isnan(log_Il)) else np.nan
        return (E_tafel, slope, intercept, tafel_slope, r**2 if slope else 0), (E_plateau, I_lim), E_corr

    # -- Fit set A
    (E_tafel_A, slope_A, intercept_A, tafs_A, r2A), (E_plt_A, I_lim_A), Ecorr_A = fit_tafel_and_plateau(E_clean, I_clean, tafel_min_A, tafel_max_A, plateau_min_A, plateau_max_A)
    # -- Fit set B
    (E_tafel_B, slope_B, intercept_B, tafs_B, r2B), (E_plt_B, I_lim_B), Ecorr_B = fit_tafel_and_plateau(E_clean, I_clean, tafel_min_B, tafel_max_B, plateau_min_B, plateau_max_B)

    # --- Overlay plot
    fig, ax = plt.subplots()
    ax.plot(E_clean, np.abs(I_clean), '-', c="blue", label="|I| observed")
    # A
    if len(E_tafel_A) > 1:
        ax.plot(E_tafel_A, 10**(slope_A*E_tafel_A + intercept_A), 'r--', label=f"Tafel A (slope={tafs_A*1000:.1f})")
    if len(E_plt_A) > 1:
        ax.axhline(I_lim_A, color='green', linestyle='--', label='Plateau A')
    if Ecorr_A and not np.isnan(Ecorr_A):
        ax.axvline(Ecorr_A, color='purple', linestyle='-', label=f'$E_{{corr}}$ A: {Ecorr_A:.3f} V')
    # B
    if len(E_tafel_B) > 1:
        ax.plot(E_tafel_B, 10**(slope_B*E_tafel_B + intercept_B), 'm:', label=f"Tafel B (slope={tafs_B*1000:.1f})")
    if len(E_plt_B) > 1:
        ax.axhline(I_lim_B, color='lime', linestyle=':', label='Plateau B')
    if Ecorr_B and not np.isnan(Ecorr_B):
        ax.axvline(Ecorr_B, color='orange', linestyle='-.', label=f'$E_{{corr}}$ B: {Ecorr_B:.3f} V')
    ax.set_yscale('log')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|I| (A/m²)")
    ax.set_title("Two Region Fits: $E_{corr}$ shift")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"""---
    #### Results **A**
    - Tafel region: {tafel_min_A:.3f} – {tafel_max_A:.3f} V (slope = {tafs_A*1000:.1f} mV/dec, R²={r2A:.3f})  
    - Plateau region: {plateau_min_A:.3f} – {plateau_max_A:.3f} V (|I_lim|={I_lim_A:.2e} A)  
    - $E_{{corr}}$ from fit A: **{Ecorr_A:.4f} V**

    ---
    #### Results **B**
    - Tafel region: {tafel_min_B:.3f} – {tafel_max_B:.3f} V (slope = {tafs_B*1000:.1f} mV/dec, R²={r2B:.3f})  
    - Plateau region: {plateau_min_B:.3f} – {plateau_max_B:.3f} V (|I_lim|={I_lim_B:.2e} A)  
    - $E_{{corr}}$ from fit B: **{Ecorr_B:.4f} V**

    ---
    - **Visualize how different reasonable region choices shift $E_{{corr}}$.**
    - If you want to see both overlays, plot with two colors and two lines.
    """)
