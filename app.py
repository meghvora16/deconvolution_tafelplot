import streamlit as st
import pandas as pd
import os

try:
    # Try import from package or local file
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please make sure 'polcurvefit.py' is in your app folder or the package is installed.")
    st.stop()

st.title("Mixed-Control Tafel Fit (Activation + Diffusion)")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel (potential in one column, current in another)", 
    type=["csv", "xlsx"]
)
plot_output_folder = 'Visualization_mixed_control_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    try:
        # Load file
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
        st.markdown("**Surface area affects Icorr and Ilim. In cm². Default is 1 cm².**")
        area_cm2 = st.number_input('Sample surface area (cm²)', min_value=0.0001, value=1.0)
        area_m2 = area_cm2 * 1e-4

        # Instantiate fitting object
        Polcurve = polcurvefit(E, I, sample_surface=area_m2)
        st.info("Fitting whole curve (activation + diffusion regions) to physics-based model...")

        # Try both method names
        fit_result = None
        if hasattr(Polcurve, "active_diff_pol_fit"):
            fit_result = Polcurve.active_diff_pol_fit()
            popt, E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2 = fit_result
        elif hasattr(Polcurve, "mixed_pol_fit"):
            # Use entire experimental window (relative to Ecorr)
            e_corr_loc = Polcurve._find_Ecorr()
            wmin, wmax = min(E) - e_corr_loc, max(E) - e_corr_loc
            fit_result = Polcurve.mixed_pol_fit([wmin, wmax])
            [_, _], E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result
        else:
            st.error("No global mixed-control fitting method found in your polcurvefit install.")
            st.stop()

        st.success("Fit completed on entire curve!")

        st.markdown("### Results (from global fit):")
        st.write(f"- **E_corr:** {E_corr:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim):** {lim_current:.3e} A")

        # ------ NEW: Standard R² calculation on log(|I|) ------
        import numpy as np
        E_fit = np.array(Polcurve.fit_results[1])
        I_fit = np.array(Polcurve.fit_results[0])
        I_pred = np.interp(E, E_fit, I_fit)
        I_obs = I
        mask = (np.abs(I_obs) > 0) & (np.abs(I_pred) > 0)
        if np.sum(mask) < 3:
            st.warning("Not enough nonzero data for meaningful R² calculation.")
            r2_log = np.nan
        else:
            logI_obs = np.log10(np.abs(I_obs[mask]))
            logI_pred = np.log10(np.abs(I_pred[mask]))
            r2_log = 1 - np.sum((logI_obs - logI_pred) ** 2) / np.sum((logI_obs - np.mean(logI_obs)) ** 2)
        st.write(f"- **Goodness of fit (R², log-I):** {r2_log:.4f}")
        # ------------------------------------------------------

        # Save and display plots
        try:
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
        except Exception as plotexc:
            st.warning(f"Plotting failed: {plotexc}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
