import streamlit as st
import pandas as pd
import numpy as np
from polcurvefit import tafel
import os
import matplotlib.pyplot as plt
import shutil

st.title("Tafel Fitting with polcurvefit (Ule Angst Method)")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# Prepare plot output folder
plot_output_folder = 'tafel_fit_visualization'
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # Read uploaded file
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # Reset pointer
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Uploaded file must be .csv or .xlsx.")
        st.stop()

    st.write("**Data Preview:**")
    st.dataframe(df.head())

    # Get columns
    columns = df.columns.tolist()

    # Try to automatically use the correct columns for WE(1) data (customize here if needed)
    pot_candidates = [c for c in columns if "potential" in c.lower()]
    curr_candidates = [c for c in columns if "current" in c.lower()]

    default_pot = pot_candidates[0] if pot_candidates else columns[0]
    default_curr = curr_candidates[0] if curr_candidates else columns[1]

    # Let user choose which columns just in case (preselected to best guess)
    st.write("**Select columns to use for Tafel analysis:**")
    pot_col = st.selectbox("Potential column", columns, index=columns.index(default_pot))
    curr_col = st.selectbox("Current column", columns, index=columns.index(default_curr))

    E = df[pot_col].astype(float).values
    I = df[curr_col].astype(float).values

    # Input for sample area if you want current density (otherwise leave as 1)
    area = st.number_input("Sample surface area for current density (cm²)", min_value=1e-8, value=1e-4, format="%.0e")
    use_current_density = st.checkbox("Convert to current density (A/cm²)", value=True)
    if use_current_density:
        i = I / area
        y_label = "Potential (V)"
        x_label = "log10(Current density / A/cm²)"
    else:
        i = I
        y_label = "Potential (V)"
        x_label = "log10(Current / A)"

    # Allow user to select a fitting region in Potential
    st.write("**Select potential window for Tafel fit**")
    emin, emax = float(np.min(E)), float(np.max(E))
    step = np.round((emax - emin) / 50, 5)
    tafel_window = st.slider(
        "Fitting range (V)",
        min_value=emin, max_value=emax, value=(emin, emax), step=step
    )
    emask = (E >= tafel_window[0]) & (E <= tafel_window[1])
    if np.sum(emask) < 4:
        st.warning("Not enough points in this window for a fit.")
        st.stop()

    E_fit = E[emask]
    i_fit = i[emask]

    # Tafel fit via polcurvefit
    result = tafel.fit(i=i_fit, E=E_fit, verbose=False)

    st.markdown("## Fitted Tafel Parameters")
    st.json({k: f"{v:.5g} ± {result.errors.get(k,0):.2g}" for k, v in result.params.items()})
    
    # Make and save plot
    plt.figure()
    plt.plot(np.log10(i_fit), E_fit, 'o', label='Data for fitting')
    xfit = np.linspace(np.log10(i_fit.min()*0.9), np.log10(i_fit.max()*1.1), 200)
    fit_line = result.params['a'] + result.params['b'] * xfit
    plt.plot(xfit, fit_line, 'r-', label='Tafel fit')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plotfile = os.path.join(plot_output_folder, "tafel_fit.png")
    plt.savefig(plotfile, bbox_inches='tight', dpi=150)
    plt.close()

    st.image(plotfile, caption="Tafel fit: log(current) vs. potential")

    st.success("Analysis complete. You can change window/columns and re-run.")

# Requirements: streamlit, pandas, numpy, matplotlib, polcurvefit
# Run: streamlit run app.py
