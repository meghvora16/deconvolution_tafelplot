import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import os

st.title("Tafel Slope and Limiting Current: Region-Selective Piecewise Fitting")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential, Current cols)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_piecewise_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'

    E = df[pot_col].values
    I = df[cur_col].values

    st.write(df[[pot_col, cur_col]].head(10))

    # Clean zeros/NaNs
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    # Let user CHOOSE ranges (by dragging slider or set defaults)
    st.write("**Pick potential window for Tafel (linear) region fit:**")
    tafel_min = st.number_input('Tafel region: Minimum potential', value=float(np.percentile(E_clean, 5)))
    tafel_max = st.number_input('Tafel region: Maximum potential', value=float(np.percentile(E_clean, 25)))

    st.write("**Pick potential window for diffusion/plateau region (flat current):**")
    plateau_min = st.number_input('Plateau region: Minimum potential', value=float(np.percentile(E_clean, 75)))
    plateau_max = st.number_input('Plateau region: Maximum potential', value=float(np.percentile(E_clean, 98)))

    # Tafel fit (linear)
    tafel_mask = (E_clean >= tafel_min) & (E_clean <= tafel_max)
    E_tafel = E_clean[tafel_mask]
    logI_tafel = np.log10(np.abs(I_clean[tafel_mask]))
    slope, intercept, r_value, p_value, std_err = linregress(E_tafel, logI_tafel)
    tafel_slope = 1/slope # V/dec

    st.markdown(f"### Tafel Region Fit (log(|I|) vs. E):")
    st.write(f"- Fitted region: {tafel_min:.3f} V to {tafel_max:.3f} V")
    st.write(f"- Tafel slope: {tafel_slope*1000:.2f} mV/dec")
    st.write(f"- Intercept: {intercept:.3f}")
    st.write(f"- R²: {r_value**2:.4f}")

    # Plateau (limiting current) fit (average)
    plateau_mask = (E_clean >= plateau_min) & (E_clean <= plateau_max)
    I_plateau = np.abs(I_clean[plateau_mask])
    I_lim_mean = np.mean(I_plateau)
    I_lim_median = np.median(I_plateau)
    st.markdown("### Limiting Current (plateau region):")
    st.write(f"- Fitted region: {plateau_min:.3f} V to {plateau_max:.3f} V")
    st.write(f"- Mean |I_lim|: {I_lim_mean:.3e} A")
    st.write(f"- Median |I_lim|: {I_lim_median:.3e} A")
    st.write(f"- Std Dev: {np.std(I_plateau):.3e} A")

    # Plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(E_clean, np.abs(I_clean), label="|I| observed", color="C0")
    plt.plot(E_tafel, 10**(slope*E_tafel + intercept), label="Tafel fit", color="C1")
    plt.axhline(I_lim_median, color="C2", linestyle="--", label="Plateau median")
    plt.yscale('log')
    plt.xlabel("E [V vs Ref]")
    plt.ylabel("|I| [A/m²]")
    plt.legend()
    plt.title("Piecewise Fit: Tafel region + Plateau")
    plt.tight_layout()
    outpath = os.path.join(plot_output_folder, "piecewise_fit.png")
    plt.savefig(outpath)
    st.image(outpath, caption="Piecewise fit plot (see regions)")
