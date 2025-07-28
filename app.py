import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil

try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please install with 'pip install polcurvefit'.")
    st.stop()

st.title("Global Mixed-Control (and Tafel) Polarization Fit")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel (cols: 'Potential applied (V)','WE(1).Current (A)')", 
    type=["csv", "xlsx"]
)

plot_output_folder = 'Visualization_mixed_control_fit'
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    if not all(x in df.columns for x in [pot_col, cur_col]):
        st.error(f"Columns missing! Found: {df.columns.to_list()}")
        st.stop()

    st.write(df[[pot_col, cur_col]].head(20))
    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=0.0001, value=1.0, format="%.4f")

    # Remove NaNs and zero currents
    E_all = df[pot_col].values.astype(float)
    I_all = df[cur_col].values.astype(float)
    mask = (~np.isnan(E_all)) & (~np.isnan(I_all)) & (np.abs(I_all) > 0)
    E = E_all[mask]
    I = I_all[mask]

    if len(I) < 6:
        st.error("Not enough valid points after cleaning. Check your data!")
        st.stop()

    # --- Optional: Let user fit a window, but give robust defaults:
    st.markdown("#### Set Fitting Window (absolute potential, not vs Ecorr):")
    emin, emax = float(np.min(E)), float(np.max(E))
    default_vmin = np.percentile(E, 10)
    default_vmax = np.percentile(E, 90)
    fit_win = st.slider(
        "Choose region [V] for fit (use the region where your data shows a clear Tafel/mixed transition):",
        min_value=emin, max_value=emax,
        value=(default_vmin, default_vmax), step=0.002, format="%.3f"
    )
    wmin, wmax = fit_win

    # Show selected window
    st.info(f"Fit window: {wmin:.3f} V  to  {wmax:.3f} V  (you may edit above; use a narrower region around Tafel bending if the global fit is poor)")

    region_mask = (E >= wmin) & (E <= wmax)
    E_win = E[region_mask]
    I_win = I[region_mask]

    # Show OCP (data, closest to zero current)
    idx_OCP = np.argmin(np.abs(I_win))
    OCP = E_win[idx_OCP]
    st.write(f"Open Circuit Potential (OCP, data): {OCP:.4f} V (in window)")

    # Fit (area in cm² as required by polcurvefit)
    Polcurve = polcurvefit(E_win, I_win, sample_surface=area_cm2)
    e_corr = Polcurve._find_Ecorr()

    # Now define window **relative to Ecorr** for the fit, as required by polcurvefit:
    window_vs_Ecorr = [wmin - e_corr, wmax - e_corr]

    # Allow tuning weighting
    st.markdown("#### Tuning weights (for advanced):\n- Lower W = less plateau, higher w_ac = more Tafel slope influence.")
    w_ac = st.slider("Weight for ACTIVE/Tafel region", 0.01, 0.2, value=0.08, step=0.01)
    W = st.slider("Weight for diffusion/plateau", 5, 120, value=15, step=5)

    st.info(f"Fitting window (vs. Ecorr): [{window_vs_Ecorr[0]:.3f}, {window_vs_Ecorr[1]:.3f}] V  |  w_ac={w_ac}, W={W}")

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window=window_vs_Ecorr,
            apply_weight_distribution=True,
            w_ac=w_ac,
            W=W
        )
        [_, _], E_corr, I_corr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Fit completed!")
        st.write(f"- **E_corr (fit):** {E_corr:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope * 1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope * 1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim):** {lim_current:.3e} A")
        st.write(f"- **Fit R² (log|I|):** {r2:.4f}")
        st.write(f"- **Δ(OCP - Ecorr):** {(OCP - E_corr):.4f} V")

        if abs(OCP - E_corr) > 0.025:
            st.warning("OCP and Ecorr differ by >25 mV. Check data or fit window.")

        # Save and display plots
        try:
            Polcurve.plotting(output_folder=plot_output_folder)
            st.markdown("### Plots:")
            for plot_file in sorted(os.listdir(plot_output_folder)):
                if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    st.image(os.path.join(plot_output_folder, plot_file), caption=plot_file)
        except Exception as plotexc:
            st.warning(f"Plotting failed: {plotexc}")

    except Exception as fit_exc:
        st.error(f"Fit failed: {fit_exc}")

st.markdown("""
---
**Instructions:**  
- Upload your polarization curve in CSV or Excel with columns 'Potential applied (V)' and 'WE(1).Current (A)'.
- Choose a fit window covering the true Tafel/mixed region (where your data is most "straight" on log(I) plot and there's curvature before reaching the plateau).
- Tune weights for slope or plateau influence if needed.
- If the fit is poor, try narrowing the fit window to just the well-behaved region.
- Results show Ecorr, Icorr, slopes, limiting current, and high-quality fit plots for publication or reporting.
""")
