import streamlit as st
import pandas as pd
from polcurvefit import polcurvefit
import os

# Set up the Streamlit app title
st.title("Electrochemical Data Analysis")

# File uploader widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Directory where plots are saved
plot_output_folder = 'Visualization_activation_control_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    try:
        # Read the uploaded CSV or Excel file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Extract potential and current columns
        potential_column = df.columns[0]
        current_column = df.columns[2]

        E = df[potential_column].values
        I = df[current_column].values

        # Initialize polcurvefit with the loaded data
        Polcurve = polcurvefit(E, I, sample_surface=1E-4)
        
        # Perform the active polarization curve fit
        popt, E_corr, I_corr, anodic_slope, cathodic_slope, r_square = Polcurve.active_pol_fit(window=[-0.07, 0.07])

        # Display results
        st.write("Fitted Parameters:")
        st.write(f"Fitted parameters: {popt}")
        st.write(f"E_corr: {E_corr}")
        st.write(f"I_corr: {I_corr}")
        st.write(f"Anodic slope: {anodic_slope}")
        st.write(f"Cathodic slope: {cathodic_slope}")
        st.write(f"R²: {r_square}")

        # Save plots
        Polcurve.plotting(output_folder=plot_output_folder)

        # Display each plot in the visualization directory
        st.write("Plots:")
        files = os.listdir(plot_output_folder)

        for plot_file in files:
            # Check if the file is an image
            if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                plot_path = os.path.join(plot_output_folder, plot_file)
                st.image(plot_path, caption=plot_file)
    except Exception as e:
        st.error(f"An error occurred: {e}")
