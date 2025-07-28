import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import polcurvefit as pcf

def main():
    st.title("Tafel Plot Deconvolution App with Excel Support")

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
        
        # Logarithm of current density for fitting
        log_current_density = np.log10(np.abs(current_density))

        # Plot the data
        fig, ax = plt.subplots()
        ax.plot(potential, log_current_density, label='Experimental Data', marker='o', linestyle='')
        
        # Fit using polcurvefit (adjust based on polcurvefit capabilities and API)
        try:
            # Perform the fitting using polcurvefit
            fit_params = pcf.fit(curve=current_density, x=potential, method='tafel')  # Adjust method/args per API
            
            # Use these fit_params to construct a curve (dummy implementation; update based on lib specifics)
            fit_curve = fit_params['a'] * np.exp(fit_params['b'] * potential) - fit_params['c'] * np.exp(fit_params['d'] * potential)
            ax.plot(potential, np.log10(np.abs(fit_curve)), label='Polcurvefit Model', color='purple')

            # Display fit parameters
            st.write(f"Polcurvefit Parameters: {fit_params}")
        
        except Exception as e:
            st.error(f"Error in polcurvefit: {e}")

        # Set labels and legend
        ax.set_xlabel('Potential applied (V)')
        ax.set_ylabel('log(Current Density) (log(A))')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

if __name__ == "__main__":
    main()
