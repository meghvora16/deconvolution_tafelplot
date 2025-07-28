import streamlit as st
import pandas as pd
import os
from polcurvefit import polcurvefit

st.title("Mixed Control Tafel Deconvolution (van Ede & Angst 2022, polcurvefit)")

uploaded_file = st.file_uploader("Upload CSV or Excel (first col = potential, current in 3rd/choose below)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_mixed_control_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        st.write("Sample of Uploaded Data:")
        st.dataframe(df.head(10))

        # Let user select columns (potential and current)
        potential_col = st.selectbox("Select the column for Potential (V):", df.columns, index=0)
        current_col = st.selectbox("Select the column for Current (A):", df.columns, index=2 if len(df.columns) > 2 else 1)

        E = df[potential_col].values
        I = df[current_col].values

        # Let user select sample surface area
        surface_area_cm2 = st.number_input("Sample Surface Area (cm²)", value=1.0, min_value=0.0001)
        surface_area_m2 = surface_area_cm2 * 1e-4

        # Do the mixed-control fit (objective, robust deconvolution)
        Polcurve = polcurvefit(E, I, sample_surface=surface_area_m2)
        st.info("Fitting mass-transfer and kinetic parameters over the full curve...")

        result = Polcurve.active_diff_pol_fit()
        popt, E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2 = result

        st.success("Fitting done!")

        st.markdown("### Extracted Parameters (Mixed Control Global Fit):")
        st.write(f"Corrosion potential \($E_{{corr}}$\): **{E_corr:.4f} V**")
        st.write(f"Corrosion current \($I_{{corr}}$\): **{I_corr:.3e} A**")
        st.write(f"Anodic Tafel slope: **{anodic_slope*1000:.2f} mV/dec**")
        st.write(f"Cathodic Tafel slope: **{cathodic_slope*1000:.2f} mV/dec**")
        st.write(f"Limiting current (diffusion): **{lim_current:.3e} A**")
        st.write(f"Fit $R^2$: **{r2:.4f}**")

        # Save polcurvefit plots & display
        Polcurve.plotting(output_folder=plot_output_folder)
        st.markdown("### Visualization:")

        for plot_file in sorted(os.listdir(plot_output_folder)):
            if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                plot_path = os.path.join(plot_output_folder, plot_file)
                st.image(plot_path, caption=plot_file)
                # Download button for each plot
                with open(plot_path, "rb") as imgf:
                    st.download_button(
                        label=f"Download {plot_file}",
                        data=imgf,
                        file_name=plot_file,
                        mime="image/png"
                    )

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("**How it works:** This app fits the entire polarization curve to a model combining activation (Tafel) and diffusion (limiting) control, using a physically-based, objective, mathematically robust method (van Ede & Angst, 2022; [polcurvefit docs](https://polcurvefit.readthedocs.io/en/latest/)). This avoids subjective region selection and gives true kinetic and transport parameters. Contact your corrosion scientist for further interpretation!")
