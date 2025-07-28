import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Dummy function for the fitting model - adjust based on specific model you are using
def activation_diffusion(x, a1, b1, a2, b2):
    return a1 * np.exp(b1 * x) - a2 * np.exp(b2 * x)

def main():
    st.title("Tafel Plot Deconvolution App")

    # Upload data
    st.sidebar.title("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        # Read data depending on file type
        if uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file)

        # Use specific column names from the provided data structure
        potential = data['Potential applied (V)'].values
        current_density = data['WE(1).Current (A)'].values
        
        # Logarithm of current density for fitting: use only positive values
        log_current_density = np.log10(np.abs(current_density))

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(potential, log_current_density, label='Experimental Data', marker='o', linestyle='')

        # Perform curve fitting using scipy's curve_fit
        try:
            fit_params, _ = curve_fit(activation_diffusion, potential, current_density)
            
            # Create a fit curve from the fitted parameters
            fit_curve = activation_diffusion(potential, *fit_params)
            ax.plot(potential, np.log10(np.abs(fit_curve)), label='Curve Fit Model', color='purple')

            # Display fit parameters
            st.write(f"Curve Fit Parameters: a1 = {fit_params[0]:.3e}, b1 = {fit_params[1]:.3e}, a2 = {fit_params[2]:.3e}, b2 = {fit_params[3]:.3e}")
        
        except Exception as e:
            st.error(f"Error in curve fitting: {e}")

        # Set labels and legend
        ax.set_xlabel('Potential applied (V)')
        ax.set_ylabel('log(Current Density) (log(A))')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    main()
