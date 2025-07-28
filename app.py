import streamlit as st
import pandas as pd
import os
from polcurvefit import polcurvefit

# Diagnostic: version/location & available fit methods
import polcurvefit as pcfit
st.write("polcurvefit version:", pcfit.__version__)
st.write("polcurvefit location:", pcfit.__file__)
st.write("Available fit methods:", [m for m in dir(pcfit.polcurvefit) if "fit" in m])

st.title("Global Mixed-Control Tafel Fit (Activation + Diffusion Regions)")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
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

        st.write("Sample Data:")
        st.dataframe(df.head())

        # Let user select which columns to use
        potential_col = st.selectbox("Select Potential column:", df.columns, index=0)
        current_col = st.selectbox("Select Current column:", df.columns, index=2 if len(df.columns) > 2 else 1)

        E = df[potential_col].values
        I = df[current_col].values

        # Optional area input
        st.markdown("**Surface area will affect Icorr/ilim output. For current in A and area in cm², default is 1 cm².**")
        area_cm2 = st.number_input('Sample surface area (cm²)', min_value=0.0001, value=1.0)
        area_m2 = area_cm2 * 1e-4

        # Setup polcurvefit
        Polcurve = polcurvefit(E, I, sample_surface=area_m2)
        st.info("Fitting entire curve (activation + diffusion regions) to physically-based model...")

        # ------- USE THE GLOBAL MIXED-CONTROL FIT ----------

        if hasattr(Polcurve, "active_diff_pol_fit"):
            # Newer polcurvefit
            fit_result = Polcurve.active_diff_pol_fit()
            popt, E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2 = fit_result
        elif hasattr(Polcurve, "mixed_pol_fit"):
            # Fallback for older codebase/classic method name
            # Use the entire curve window (relative to E_corr)
            e_corr_loc = Polcurve._find_Ecorr()
            # Window: all points relative to corrosion potential
            window = [E.min() - e_corr_loc, E.max() - e_corr_loc]
            fit_result = Polcurve.mixed_pol_fit(window)
            [_, _], E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2, _ = fit_result
        else:
            st.error("No suitable mixed-control fit method found in your polcurvefit install.")
            st.stop()

        st.success("Fit completed using global mixed-control model!")

        st.markdown("### Results (from entire curve fit):")
        st.write(f"- **E_corr:** {E_corr:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim):** {lim_current:.3e} A")
        st.write(f"- **Goodness of fit (R²):** {r2:.4f}")

        # Save and display plots
        Polcurve.plotting(output_folder=plot_output_folder)
        st.markdown("### Plots:")
        for plot_file in sorted(os.listdir(plot_output_folder)):
            if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                plot_path = os.path.join(plot_output_folder, plot_file)
                st.image(plot_path, caption=plot_file)
                with open(plot_path, "rb") as imgf:
                    st.download_button(
                        label=f"Download {plot_file}",
                        data=imgf,
                        file_name=plot_file,
                        mime="image/png"
                    )
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("""
---
**This app fits the *entire* polarization curve using a model that mathematically separates activation (Tafel) and diffusion effects, following van Ede & Angst (2022) and implemented via [polcurvefit](https://polcurvefit.readthedocs.io/). This provides physically meaningful Tafel slopes, corrosion current, and diffusion-limited current, free from manual/subjective selection. If you have trouble, copy and share the version/diagnostic output above for help!
""")
