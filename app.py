import streamlit as st
import pandas as pd
import numpy as np
import os

try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please make sure 'polcurvefit.py' is in your app folder or the package is installed.")
    st.stop()

st.title("Global Mixed-Control Tafel Fit (Fits Linear + Diffusion Region)")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel (first col Potential, one col Current)", 
    type=["csv", "xlsx"]
)
plot_output_folder = 'Visualization_mixed_control_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # --- HARDCODED COLUMN NAMES for your data ---
    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    E = df[pot_col].values
    I = df[cur_col].values

    st.write(df[[pot_col, cur_col]].head(20))

    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=0.0001, value=1.0)
    area_m2 = area_cm2 * 1e-4

    # Remove NaNs, zeros if present
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    Polcurve = polcurvefit(E_clean, I_clean, sample_surface=area_m2)
    e_corr = Polcurve._find_Ecorr()

    # -- *** FIT WINDOW: ENTIRE POTENTIAL RANGE *** --
    window = [np.min(E_clean) - e_corr, np.max(E_clean) - e_corr]
    st.info(f"Fitting entire window: [{window[0]:.3f}, {window[1]:.3f}] V around Ecorr.")

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window,
            apply_weight_distribution=True,
            w_ac=0.06,
            W=80
        )
        [_, _], E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Fit completed!")
        st.write(f"- **E_corr:** {E_corr:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim):** {lim_current:.3e} A")

        # Log-scale R² calculation
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
        except Exception as plotexc:
            st.warning(f"Plotting failed: {plotexc}")

    except Exception as fit_exc:
        st.error(f"Fit failed: {fit_exc}")

st.markdown("""
---
**This app fits your entire polarization curve using the full model (activation+diffusion/plateau regions) in one step. If the plateau is visible in your data, you will see it fitted. For advanced multi-process fitting, consult a corrosion analyst or use data containing both Tafel and limiting branches.**
""")
