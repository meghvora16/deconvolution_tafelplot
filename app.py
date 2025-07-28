import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil

try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please make sure 'polcurvefit.py' is in your app folder or the package is installed. Install: pip install polcurvefit")
    st.stop()

st.title("Global Mixed-Control Tafel Fit (Tune your window and weighting!)")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel (columns: your potential and current)", 
    type=["csv", "xlsx"]
)

plot_output_folder = 'Visualization_mixed_control_fit'
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    # ---- Column names for YOUR data
    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    for col in [pot_col, cur_col]:
        if col not in df.columns:
            st.error(f"Expected column '{col}' not found! Columns found: {df.columns.tolist()}")
            st.stop()
    E = df[pot_col].astype(float).values
    I = df[cur_col].astype(float).values

    st.write(df[[pot_col, cur_col]].head(20))
    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=1e-8, value=1.0, format="%.4f")

    # Clean
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    Polcurve = polcurvefit(E_clean, I_clean, sample_surface=area_cm2)
    e_corr = Polcurve._find_Ecorr()

    # --- USER SELECTS FITTING WINDOW ---
    vmin,vmax = float(np.min(E_clean)), float(np.max(E_clean))
    st.write("**Select the region (vs Ecorr) to fit (default: usually -0.1 to +0.1 V)**")
    suggested_min = max(vmin, e_corr-0.12)
    suggested_max = min(vmax, e_corr+0.12)
    window = st.slider(
        "Fitting window (V vs Ecorr, left = more cathodic, right = more anodic)", 
        min_value=vmin, max_value=vmax, 
        value=(suggested_min, suggested_max), step=0.005,
        format="%.3f"
    )
    window_vs_Ecorr = [window[0] - e_corr, window[1] - e_corr]

    # --- USER TUNES WEIGHTING ---
    st.write("**Tune weighting to change how much priority is given to active (kinetic) vs diffusion (plateau):**")
    w_ac = st.slider("Relative weight for ACTIVE region (w_ac)", 0.01, 0.2, value=0.05, step=0.01, format="%.2f")
    W = st.slider("Weight for DIFFUSION/plateau (W)", 1, 200, value=40, step=5)

    st.info(f"Fitting window: [{window_vs_Ecorr[0]:.3f}, {window_vs_Ecorr[1]:.3f}] V (vs. Ecorr: {e_corr:.3f} V)")

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window_vs_Ecorr,
            apply_weight_distribution=True,
            w_ac=w_ac,
            W=W
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
**TIP:**  
- Narrow the fitting window to the Tafel region (the regime *after* the "bend" but before the plateau/limiting region).
- Increase `w_ac` to give more emphasis to the kinetics; decrease `W` to reduce the influence of the plateau.
- Try changing the window and weights and see the fit visually update!
""")
