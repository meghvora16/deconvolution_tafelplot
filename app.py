import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import os

st.title("POLFIT-Style Separate Region Fitting: Tafel & Diffusion Control")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential & Current)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_piecewise_fit'
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

    st.write("First few points:", df[[pot_col, cur_col]].head())

    # Remove zeros/NaNs
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    # Show plots to guide region choice
    st.write("Log plot (select fitting regions below):")
    fig, ax = plt.subplots()
    ax.plot(E_clean, np.abs(I_clean), '-', c="blue", label="|I|")
    ax.set_yscale('log')
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|I| (A/m²)")
    ax.set_title("Log(|I|) vs E")
    st.pyplot(fig)

    # --- User-select or default region for Tafel (linear) region
    st.write("### Tafel region selection (for log-linear activation branch fit)")
    tafel_min, tafel_max = st.select_slider(
        "Select potential region for Tafel fit",
        options=np.round(E_clean, 5),
        value=(round(E_clean[10], 5), round(E_clean[len(E_clean)//3], 5))
    )

    tafel_mask = (E_clean >= tafel_min) & (E_clean <= tafel_max)
    E_tafel = E_clean[tafel_mask]
    logI_tafel = np.log10(np.abs(I_clean[tafel_mask]))

    if len(E_tafel) < 2:
        st.warning("Not enough points selected for Tafel fit.")
    else:
        slope, intercept, r, p, std = linregress(E_tafel, logI_tafel)
        tafel_slope = 1/slope
        tafel_label = f'Tafel fit slope: {tafel_slope*1000:.2f} mV/dec'
        st.write(f"**Tafel region:** from {tafel_min:.3f} V to {tafel_max:.3f} V")
        st.write(f"**Tafel slope:** {tafel_slope*1000:.2f} mV/dec")
        st.write(f"**R²:** {r**2:.3f}")

    # --- User-select region for plateau/limiting current region
    st.write("### Plateau (diffusion) region selection")
    plateau_min, plateau_max = st.select_slider(
        "Select potential region for plateau fit",
        options=np.round(E_clean, 5),
        value=(round(E_clean[-len(E_clean)//6], 5), round(E_clean[-5], 5))
    )
    plateau_mask = (E_clean >= plateau_min) & (E_clean <= plateau_max)
    I_plateau = np.abs(I_clean[plateau_mask])

    if len(I_plateau) < 2:
        st.warning("Not enough points selected for plateau/limiting current fit.")
    else:
        ilim_mean = np.mean(I_plateau)
        ilim_median = np.median(I_plateau)
        st.write(f"**Plateau region:** from {plateau_min:.3f} V to {plateau_max:.3f} V")
        st.write(f"**Limiting current (mean):** {ilim_mean:.3e} A")
        st.write(f"**Limiting current (median):** {ilim_median:.3e} A")
    
    # --- Show results overlayed on plot
    fig2, ax2 = plt.subplots()
    ax2.plot(E_clean, np.abs(I_clean), '-', c="blue", label="|I| observed")
    if len(E_tafel) >= 2:
        ax2.plot(E_tafel, 10**(slope*E_tafel + intercept), 'r--', label=tafel_label)
    if len(I_plateau) >= 2:
        ax2.axhline(ilim_median, color='green', linestyle='--', label='Limiting current median')
    ax2.set_yscale('log')
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("|I| (A/m²)")
    ax2.set_title("Piecewise POLFIT-region fitting")
    ax2.legend()
    st.pyplot(fig2)

    # Save output
    outpath = os.path.join(plot_output_folder, "piecewise_polfit.png")
    fig2.savefig(outpath)
    st.markdown(f"Image saved: {outpath}")

    st.markdown("""
    ---
    **This app mimics POLFIT and recommended literature practice: you select (by inspection or algorithm) the physical regions corresponding to the Tafel (activation) and limiting current (diffusion) regimes.
    Each region is fit using the appropriate model (linear for log(|I|) in Tafel region, mean in plateau region).
    This approach ensures the values you use/report are physically meaningful - which is not generally true of global fits for real data!**
    """)
