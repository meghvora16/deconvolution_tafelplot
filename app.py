import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

st.title("Visualizing $E_{corr}$: Tafel (activation) vs. Plateau (diffusion) Fitting")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential, Current)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_Ecorr_regions'
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

    st.write("Choose regions with sliders below (guided by log plot):")
    fig, ax = plt.subplots()
    ax.plot(E_clean, np.abs(I_clean), '-', c="blue", label="|I|")
    ax.set_yscale('log')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|I| [A/m²]")
    ax.set_title("Log(|I|) vs E")
    st.pyplot(fig)

    st.write("### Tafel region (log-linear activation region) for fit:")
    tafel_min = st.number_input('Tafel region: Minimum potential', value=float(np.percentile(E_clean, 5)))
    tafel_max = st.number_input('Tafel region: Maximum potential', value=float(np.percentile(E_clean, 25)))

    st.write("### Plateau (diffusion) region for fit:")
    plateau_min = st.number_input('Plateau region: Minimum potential', value=float(np.percentile(E_clean, 75)))
    plateau_max = st.number_input('Plateau region: Maximum potential', value=float(np.percentile(E_clean, 98)))

    # --- Fit Tafel region (linear, log(|I|))
    tafel_mask = (E_clean >= tafel_min) & (E_clean <= tafel_max)
    E_tafel = E_clean[tafel_mask]
    I_tafel = I_clean[tafel_mask]
    logI_tafel = np.log10(np.abs(I_tafel))

    show_fit = False
    if len(E_tafel) >= 2:
        # Linear fit in Tafel region
        slope, intercept, r, p, std = linregress(E_tafel, logI_tafel)
        tafel_slope = 1/slope  # V/dec
        show_fit = True

    # --- Plateau region fit (mean/median)
    plateau_mask = (E_clean >= plateau_min) & (E_clean <= plateau_max)
    E_plateau = E_clean[plateau_mask]
    I_plateau = I_clean[plateau_mask]
    I_lim = np.median(np.abs(I_plateau))

    # -- Find Ecorr as intersection of linear (logI= slope*E+intercept) and plateau (logIl)
    if show_fit:
        log_Il = np.log10(I_lim)
        # Solve for E: slope*E + intercept = log_Il  =>  E = (log_Il - intercept) / slope
        E_corr_tafel_vs_plateau = (log_Il - intercept) / slope
        st.success(f"Intersection $E_{{corr}}$ (Tafel/Plateau): {E_corr_tafel_vs_plateau:.4f} V")
        st.write(f"Tafel slope: {tafel_slope*1000:.2f} mV/dec")
        st.write(f"R²(Tafel fit): {r**2:.3f}")
        st.write(f"Limiting current (plateau): {I_lim:.3e} A")

    # -- Plot all
    fig2, ax2 = plt.subplots()
    ax2.plot(E_clean, np.abs(I_clean), '-', c="blue", label="|I| observed")
    if show_fit:
        E_line = np.linspace(tafel_min, tafel_max, 100)
        ax2.plot(E_line, 10**(slope*E_line + intercept), 'r--', label="Tafel fit (activation)")
        ax2.axhline(I_lim, color='green', linestyle='--', label='Plateau (diffusion) median')
        ax2.axvline(E_corr_tafel_vs_plateau, color='purple', linestyle='-.', label='$E_{corr}$ (Tafel/Plateau intersection)')
    ax2.set_yscale('log')
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("|I| (A/m²)")
    ax2.set_title("Activation vs Diffusion Region: $E_{corr}$ visualization")
    ax2.legend()
    st.pyplot(fig2)
    plt.savefig(os.path.join(plot_output_folder, "activation_vs_diffusion_Ecorr.png"))

    st.markdown("""
    ---
    **You can now see the $E_{corr}$ as found either by the full Tafel/activation region fit,
    by the intersection with the plateau region, and how different segments of the curve give
    different $E_{corr}$ values.**
    """)
