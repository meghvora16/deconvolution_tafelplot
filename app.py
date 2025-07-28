import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil

# Try to import polcurvefit, else show error
try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please make sure 'polcurvefit.py' is in your app folder or the package is installed. Install: pip install polcurvefit")
    st.stop()

st.title("Global Mixed-Control Tafel Fit (Activation + Diffusion)")

# File uploader
uploaded_file = st.file_uploader(
    "Upload CSV or Excel (columns: your potential and current)", 
    type=["csv", "xlsx"]
)

plot_output_folder = 'Visualization_mixed_control_fit'

# Wipe output folder on new run for clean plot listing
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # ---- Column names for YOUR data, with check and user select fallback ----
    expected_cols = ['Potential applied (V)', 'WE(1).Current (A)']
    for col in expected_cols:
        if col not in df.columns:
            st.error(f"Expected column '{col}' not found! Columns found: {df.columns.tolist()}")
            st.stop()

    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    st.write("Using columns:", pot_col, "and", cur_col)

    E = df[pot_col].astype(float).values
    I = df[cur_col].astype(float).values

    st.write(df[[pot_col, cur_col]].head(20))

    # Get surface area (default 1 cm^2 for current density, user can edit)
    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=1e-8, value=1.0, format="%.4f")
    
    # Remove NaNs, zeros if present (to avoid fit crash)
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    # Initialize polcurvefit (area in cm^2 is correct argument)
    Polcurve = polcurvefit(E_clean, I_clean, sample_surface=area_cm2)

    # Find Ecorr for potential windowing (ensures window centered on Ecorr)
    e_corr = Polcurve._find_Ecorr()

    # Window covering whole range, centered around Ecorr
    window = [np.min(E_clean) - e_corr, np.max(E_clean) - e_corr]
    st.info(f"Fitting entire window: [{window[0]:.3f}, {window[1]:.3f}] V (centered at Ecorr)")

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window,
            apply_weight_distribution=True,
            w_ac=0.04,
            W=80
        )
        [_, _], E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Fit completed!")
        st.write(f"- **E_corr (V):** {E_corr:.4f}")
        st.write(f"- **I_corr (A):** {I_corr:.3e}")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim, A):** {lim_current:.3e}")

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
This app fits your polarization curve using activation (Tafel) and diffusion (plateau) mixed control 
according to Ule Angst's method (polcurvefit). 

1. Upload a polarization curve CSV or Excel file (columns: Potential, Current)
2. Confirm/adjust your working electrode area (cm²)
3. The software fits the _whole measured curve_ using a global model.
4. You see the extracted corrosion parameters, slopes, limiting current and R² for log(current).
5. Plots of the fit are generated for QC.

*For column names different from the defaults, edit the code section that picks columns.*
""")
