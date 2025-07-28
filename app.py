import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

st.title("Automatic Highlighting: Activation (Tafel) vs. Diffusion Regions")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential, Current)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_region_highlight'
os.makedirs(plot_output_folder, exist_ok=True)

def find_best_linear_window(E, logI, wsize=15):
    """Find window (size wsize) of maximum linearity logI vs E (max R^2)."""
    best_R2 = -np.inf
    best_start = 0
    for i in range(len(E) - wsize + 1):
        winE, winlogI = E[i:i+wsize], logI[i:i+wsize]
        slope, intercept, r, *_ = linregress(winE, winlogI)
        if r**2 > best_R2:
            best_R2 = r**2
            best_start = i
    return best_start, best_start + wsize

def find_best_flat_window(E, I, wsize=15):
    """Find window (size wsize) where |I| is flattest (min std)."""
    best_std = np.inf
    best_start = 0
    for i in range(len(E) - wsize + 1):
        winI = I[i:i+wsize]
        stdv = np.std(winI)
        if stdv < best_std:
            best_std = stdv
            best_start = i
    return best_start, best_start + wsize

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
    sorter = np.argsort(E)
    E_clean = E[mask][sorter]
    I_clean = I[mask][sorter]
    logI_clean = np.log10(np.abs(I_clean))

    # Window size: adjustable or default
    wsize = st.number_input('Window size (points to consider per region)', min_value=5, max_value=min(80, len(E_clean)//2), value=max(15, len(E_clean)//12), step=1)

    # Activation (Tafel) region: most linear window in logI vs E
    t_start, t_end = find_best_linear_window(E_clean, logI_clean, wsize=wsize)
    E_tafel, logI_tafel = E_clean[t_start:t_end], logI_clean[t_start:t_end]
    slope, intercept, r, *_ = linregress(E_tafel, logI_tafel)

    # Plateau (diffusion) region: flattest window in |I|
    p_start, p_end = find_best_flat_window(E_clean, np.abs(I_clean), wsize=wsize)
    E_plateau, I_plateau = E_clean[p_start:p_end], np.abs(I_clean)[p_start:p_end]

    # -- Plot: colored regions
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(E_clean, np.abs(I_clean), '-', c="0.75", label="|I| data (all)")
    # Highlight activation region
    ax.plot(E_tafel, 10**(slope*E_tafel + intercept), 'r-', lw=2, label="Tafel linear fit (activation region)")
    ax.fill_between(E_tafel, np.min(np.abs(I_clean)), np.max(np.abs(I_clean)), color='red', alpha=0.15, label="Activation (Tafel) region")
    # Highlight plateau region
    ax.hlines(np.median(I_plateau), np.min(E_plateau), np.max(E_plateau), color='green', linestyle='--', label="Plateau (lim. current) region")
    ax.fill_between(E_plateau, np.min(I_plateau), np.max(I_plateau), color='green', alpha=0.10, label="Diffusion region")
    # Cosmetic
    ax.set_yscale('log')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|I| (A/m²)")
    ax.set_title("Activation (Tafel, RED) vs Diffusion/Plateau (GREEN) regions, auto-found")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"**Activation (Tafel) region:** {E_tafel[0]:.3f} V – {E_tafel[-1]:.3f} V (slope {1/slope*1000:.2f} mV/dec, R²={r**2:.3f})")
    st.markdown(f"**Diffusion/plateau region:** {E_plateau[0]:.3f} V – {E_plateau[-1]:.3f} V (median |I_lim|={np.median(I_plateau):.3e} A)")

    st.markdown("""
    ---
    **RED = activation (Tafel/linear) region; GREEN = plateau (diffusion-limited) region.  
    Regions are detected automatically (just like in van Ede & Angst 2022, POLFIT, and modern analysis).**
    """)
    
