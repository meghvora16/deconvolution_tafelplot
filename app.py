import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

st.title("Automatic Extraction: $E_{corr}$ from Activation & Diffusion Regions")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential, Current)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_auto_regions'
os.makedirs(plot_output_folder, exist_ok=True)

def find_best_linear_window(E, logI, wsize=15):
    """Find the window of size wsize where logI vs E is most linear (highest R^2)."""
    best_R2 = -np.inf
    best_slope, best_intercept, best_start = 0, 0, 0
    for i in range(len(E) - wsize + 1):
        thisE = E[i:i+wsize]
        thisLogI = logI[i:i+wsize]
        slope, intercept, r, p, std = linregress(thisE, thisLogI)
        if r**2 > best_R2:
            best_R2 = r**2
            best_slope = slope
            best_intercept = intercept
            best_start = i
    return best_start, best_start+wsize, best_slope, best_intercept, best_R2

def find_best_flat_window(E, I, wsize=15):
    """Find the window of size wsize with the lowest std(|I|), i.e., flattest region."""
    best_std = np.inf
    best_start = 0
    for i in range(len(E) - wsize + 1):
        thisI = I[i:i+wsize]
        stdv = np.std(thisI)
        if stdv < best_std:
            best_std = stdv
            best_start = i
    return best_start, best_start+wsize, best_std

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
    # always sort by potential for windowing
    sorter = np.argsort(E)
    E_clean = E[mask][sorter]
    I_clean = I[mask][sorter]

    logI_clean = np.log10(np.abs(I_clean))

    # ---- Auto-detect best Tafel region (max linear, size 15 points or 12.5% of data) ----
    win_size = max(10, len(E_clean)//8)
    t_start, t_end, slope, intercept, r2_tafel = find_best_linear_window(E_clean, logI_clean, wsize=win_size)
    E_tafel = E_clean[t_start:t_end]
    logI_tafel = logI_clean[t_start:t_end]

    tafel_slope = 1/slope if slope != 0 else np.inf  # V/dec

    # ---- Auto-detect best plateau region (min std, size 15 points or 12.5% of data) ----
    p_start, p_end, std_plateau = find_best_flat_window(E_clean, np.abs(I_clean), wsize=win_size)
    E_plateau = E_clean[p_start:p_end]
    I_plateau = np.abs(I_clean[p_start:p_end])
    I_lim = np.median(I_plateau)

    # -- Ecorr as intersection (Tafel line with log10(I_lim))
    log_Il = np.log10(I_lim)
    E_corr_auto = (log_Il - intercept) / slope if slope != 0 else np.nan

    # --- Outputs
    st.success(f"Auto Tafel region: {E_tafel[0]:.3f}–{E_tafel[-1]:.3f} V, slope={tafel_slope*1000:.2f} mV/dec, R²={r2_tafel:.3f}")
    st.success(f"Auto Plateau region: {E_plateau[0]:.3f}–{E_plateau[-1]:.3f} V, median I_lim={I_lim:.3e} A")
    st.success(f"Automated $E_{{corr}}$: {E_corr_auto:.4f} V")

    # --- Display overlay plot
    fig, ax = plt.subplots()
    ax.plot(E_clean, np.abs(I_clean), '-', c="blue", label="|I| observed")
    ax.plot(E_tafel, 10**(slope*E_tafel + intercept), 'r--', label=f"Tafel fit (auto)")
    ax.axhline(I_lim, color='green', linestyle='--', label='Auto plateau (I_lim)')
    if not np.isnan(E_corr_auto):
        ax.axvline(E_corr_auto, color='purple', linestyle='-.', label='$E_{corr}$ (auto)')
    ax.set_yscale('log')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|I| (A/m²)")
    ax.set_title("Auto region selection: Tafel, Plateau, $E_{corr}$")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"Best Tafel region values: {E_tafel[0]:.3f} – {E_tafel[-1]:.3f} V, slope {tafel_slope*1000:.2f} mV/dec, R² {r2_tafel:.3f}")
    st.markdown(f"Best plateau (diffusion) region: {E_plateau[0]:.3f} – {E_plateau[-1]:.3f} V, median |I_lim| = {I_lim:.3e} A")
    st.markdown(f"Computed intersection $E_{{corr}}$ = {E_corr_auto:.4f} V")

    st.markdown("""
    ---
    **This app auto-detects and fits the regions in your polarization curve most likely to represent the activation (Tafel) and diffusion (plateau) processes.
    It overlays both regimes and shows $E_{corr}$ from their intersection, with no manual selection.**
    """)
