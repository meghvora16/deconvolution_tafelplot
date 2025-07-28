import streamlit as st
import numpy as np
import pandas as pd
import os

try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please make sure 'polcurvefit.py' is in your app folder or the package is installed.")
    st.stop()

st.title("Robust Automated Mixed-Control Tafel Fit (Safe initial guesses)")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel (potential in one column, current in another)", 
    type=["csv", "xlsx"]
)
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

        potential_col = st.selectbox("Select Potential column:", df.columns, index=0)
        current_col = st.selectbox("Select Current column:", df.columns, index=2 if len(df.columns) > 2 else 1)
        E = df[potential_col].values
        I = df[current_col].values

        # Surface area input
        st.markdown("**Surface area affects Icorr and Ilim. Default is 1 cm².**")
        area_cm2 = st.number_input('Sample surface area (cm²)', min_value=0.0001, value=1.0)
        area_m2 = area_cm2 * 1e-4

        # Clean input (remove zeros/NaNs)
        mask = (np.abs(I) > 0) & (~np.isnan(I)) & (~np.isnan(E))
        E_clean = E[mask]
        I_clean = I[mask]

        # Check for enough points and calculate robust initial guesses
        if len(I_clean) < 5:
            st.error("Not enough valid (nonzero) points for fitting. Check your data window and file.")
            st.stop()

        abs_Iclean = np.abs(I_clean)
        i_min = np.min(abs_Iclean)
        i_max = np.max(abs_Iclean)
        if np.isclose(i_min, i_max, rtol=1e-6, atol=1e-12):
            i_corr_guess = i_min * 1.1
            i_L_guess = i_min * 1.2
            if i_corr_guess == i_L_guess:
                i_L_guess = i_corr_guess * 1.1
        else:
            delta = (i_max - i_min)
            i_corr_guess = i_min + delta * 0.2
            i_L_guess   = i_min + delta * 0.8
            if not (i_min < i_corr_guess < i_max):
                i_corr_guess = (i_min*0.9 + i_max*1.1)/2
            if not (i_min < i_L_guess < i_max):
                i_L_guess = (i_min*1.1 + i_max*0.9)/2

        # Log all diagnostics
        st.info(f"Initial guess: i_corr={i_corr_guess:.3e}, i_L={i_L_guess:.3e}, min={i_min:.3e}, max={i_max:.3e}")

        # Initialize polcurvefit instance and auto window
        Polcurve = polcurvefit(E_clean, I_clean, sample_surface=area_m2)
        e_corr = Polcurve._find_Ecorr()
        wwin = 0.6
        window = [-wwin, +wwin]
        st.info(f"Fitting Ecorr ±{wwin} V window. Weighted fit near Ecorr.")

        fit_result = Polcurve.mixed_pol_fit(
            window,
            i_corr_guess=i_corr_guess,
            i_L_guess=i_L_guess,
            apply_weight_distribution=True,
            w_ac=0.04,   # width around Ecorr with extra weight (V)
            W=80         # percent weight in window
        )
        [_, _], E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Windowed, weighted fit completed!")

        st.markdown("### Results (from improved fit):")
        st.write(f"- **E_corr:** {E_corr:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim):** {lim_current:.3e} A")

        # Goodness-of-fit on log-current scale
        E_fit = np.array(Polcurve.fit_results[1])
        I_fit = np.array(Polcurve.fit_results[0])
        I_pred = np.interp(E_clean, E_fit, I_fit)
        I_obs = I_clean
        mask2 = (np.abs(I_obs) > 0) & (np.abs(I_pred) > 0)
        if np.sum(mask2) < 3:
            st.warning("Not enough data after cleaning for R² calculation.")
            r2_log = np.nan
        else:
            logI_obs = np.log10(np.abs(I_obs[mask2]))
            logI_pred = np.log10(np.abs(I_pred[mask2]))
            r2_log = 1 - np.sum((logI_obs - logI_pred) ** 2) / np.sum((logI_obs - np.mean(logI_obs)) ** 2)
        st.write(f"- **Goodness of fit (R², log-I):** {r2_log:.4f}")

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

st.markdown("""
---
**This app fits your polarization curve using a window and weighting scheme that focuses on the main Tafel/diffusion region, using bulletproof initial guesses (never out-of-bounds). If fit is still non-optimal, or if you see a fitting error, check your data: all currents may be equal, or your selected columns may not contain valid data.**
""")
