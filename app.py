import streamlit as st
import pandas as pd
import os
from polcurvefit import polcurvefit

st.title("Electrochemical Data Analysis (Mixed Control)")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_activation_control_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(df)

    # You can hardcode these for your file, or keep as is for general use:
    potential_column = 'Potential applied (V)'
    current_column = 'WE(1).Current (A)'

    E = df[potential_column].values
    I = df[current_column].values

    # Initialize polcurvefit with file data
    Polcurve = polcurvefit(E, I, sample_surface=1E-4)

    # --- Here's the key line --- #
    popt, E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r_square = Polcurve.active_diff_pol_fit()

    # Show results
    st.write("Fitted Parameters (Mixed Control):")
    st.write(f"Fitted parameters: {popt}")
    st.write(f"E_corr: {E_corr}")
    st.write(f"I_corr: {I_corr}")
    st.write(f"Anodic slope: {anodic_slope}")
    st.write(f"Cathodic slope: {cathodic_slope}")
    st.write(f"Limiting current: {lim_current}")
    st.write(f"R²: {r_square}")

    # Save and show plots
    Polcurve.plotting(output_folder=plot_output_folder)
    st.write("Plots:")
    files = os.listdir(plot_output_folder)
    for plot_file in files:
        if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            plot_path = os.path.join(plot_output_folder, plot_file)
            st.image(plot_path, caption=plot_file)
