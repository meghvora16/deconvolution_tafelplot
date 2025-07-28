import streamlit as st
import pandas as pd
import os
from polcurvefit import polcurvefit

st.title("Mixed Control Tafel Deconvolution (van Ede & Angst, 2022, using polcurvefit)")

uploaded_file = st.file_uploader(
    "Upload your polarization data file (CSV or Excel) with columns: 'Potential applied (V)', 'WE(1).Current (A)'",
    type=["csv", "xlsx"]
)

plot_output_folder = 'Visualization_activation_diffusion_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    try:
        # Load the file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.write("Sample of Uploaded Data:")
        st.dataframe(df.head(5))

        # --- Hard-coded column names for your file structure ---
        required_pot_col = 'Potential applied (V)'
        required_cur_col = 'WE(1).Current (A)'

        # Check if columns exist
        if required_pot_col not in df.columns or required_cur_col not in df.columns:
            st.error(
                f"Could not find columns '{required_pot_col}' and/or '{required_cur_col}' in your file! "
                "Please check your file and upload again."
            )
            st.stop()

        E = df[required_pot_col].values
        I = df[required_cur_col].values

        # --- Set/Ask for surface area (default to 1 cm2) ---
        surface_area = st.number_input(
            'Sample surface area in cm² (default: 1, i.e. 1 cm²)',
            value=1.0,
            min_value=0.0001,
            step=0.1,
            format="%.4f"
        )
        st.write(f"Assuming sample area of **{surface_area} cm²** "
                 f"(polcurvefit expects m², so internally using {surface_area*1e-4} m²)")

        # --- Fitting using polcurvefit (the paper's core) ---
        Polcurve = polcurvefit(E, I, sample_surface=surface_area * 1e-4)
        st.info("Running mixed activation/diffusion control Tafel fit (as in the paper)...")

        fit_result = Polcurve.active_diff_pol_fit()
        (popt, E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r_square) = fit_result

        st.markdown("### Fitting Results (Mixed Activation-Diffusion):")
        st.write(f"- **E_corr:** {E_corr:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.4e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope:.2f} V/dec (={anodic_slope*1000:.1f} mV/dec)")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope:.2f} V/dec (={cathodic_slope*1000:.1f} mV/dec)")
        st.write(f"- **Limiting current:** {lim_current:.4e} A")
        st.write(f"- **R² (fit quality):** {r_square:.4f}")

        # Save and show plots
        Polcurve.plotting(output_folder=plot_output_folder)
        st.markdown("### Fit Plots:")
        files = os.listdir(plot_output_folder)
        images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for plot_file in images:
            plot_path = os.path.join(plot_output_folder, plot_file)
            st.image(plot_path, caption=plot_file)
            # Offer download
            with open(plot_path, 'rb') as imgf:
                st.download_button(
                    label=f"Download {plot_file}",
                    data=imgf,
                    file_name=plot_file,
                    mime="image/png"
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")
