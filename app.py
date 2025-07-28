import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit is not installed. Please run 'pip install polcurvefit'")
    st.stop()

st.title("Smart Mixed-Control Polarization Curve Fit (Tafel Overlay — Interactive Region and Weight Tuning)")

st.markdown("""
Upload your polarization curve data (CSV or Excel, columns: 'Potential applied (V)', 'WE(1).Current (A)').
You can interactively select a fitting window and weights for a physically meaningful, high-quality fit!
The plot shows **log10(|i|) vs E** (true Tafel plot) so you can overlay observed and fit curves clearly.
""")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file (columns: 'Potential applied (V)', 'WE(1).Current (A)')",
    type=["csv", "xlsx"]
)

plot_output_folder = 'MixedControlFitPlots'
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # --- Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    for col in [pot_col, cur_col]:
        if col not in df.columns:
            st.error(f"Missing column '{col}'! Found: {df.columns.tolist()}")
            st.stop()

    st.write("**Data preview (first 20 rows):**")
    st.dataframe(df[[pot_col, cur_col]].head(20))

    area_cm2 = st.number_input('Sample surface area (cm², for current density)', min_value=1e-6, value=1.0, format="%.4f")

    # --- Clean data
    E = df[pot_col].astype(float).values
    I = df[cur_col].astype(float).values
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E = E[mask]
    I = I[mask]
    i = I / area_cm2

    if len(E) < 7:
        st.error("Too few valid data points after cleaning. Check your file!")
        st.stop()

    # --- OCP from data
    idx_OCP = np.argmin(np.abs(I))
    OCP = E[idx_OCP]
    st.info(f"Open Circuit Potential (OCP, from data): {OCP:.4f} V (where |I| minimal)")

    # --- Initial fit to get Ecorr for slider reference
    Polcurve_preview = polcurvefit(E, I, sample_surface=area_cm2)
    Ecorr_preview = Polcurve_preview._find_Ecorr()

    st.markdown("#### Select potential window for fit ⟶")
    vmin, vmax = E.min(), E.max()
    # Suggest a window ±0.15V around Ecorr for Tafel/mixed
    win_default = (max(vmin, Ecorr_preview-0.15), min(vmax, Ecorr_preview+0.15))
    fit_window = st.slider(
        "Potential window for fit (V) [absolute, not vs Ecorr]",
        min_value=float(vmin), max_value=float(vmax),
        value=(float(win_default[0]), float(win_default[1])),
        step=0.002, format="%.3f"
    )
    wmin, wmax = fit_window
    region_mask = (E >= wmin) & (E <= wmax)
    E_fit = E[region_mask]
    I_fit = I[region_mask]
    i_fit = i[region_mask]

    st.info(f"Fit window: {wmin:.3f} V to {wmax:.3f} V (absolute, shown in red on plot)")

    # --- Fit weights (user-adjustable)
    w_ac = st.slider("Weight for active (Tafel) region (w_ac)", 0.01, 0.20, 0.08, step=0.01, format="%.2f")
    W = st.slider("Weight for diffusion (plateau) region (W)", 5, 120, 20, step=1)
    st.info(f"Fitting weights: w_ac={w_ac}, W={W}")

    # --- Fit
    if len(E_fit) < 7:
        st.error("Too few points in selected window for fit. Adjust the window.")
        st.stop()

    Polcurve = polcurvefit(E_fit, I_fit, sample_surface=area_cm2)
    Ecorr = Polcurve._find_Ecorr()
    window = [np.min(E_fit) - Ecorr, np.max(E_fit) - Ecorr]
    try:
        result = Polcurve.mixed_pol_fit(window=window, apply_weight_distribution=True, w_ac=w_ac, W=W)
        [_, _], Ecorr, Icorr, anodic_slope, cathodic_slope, lim_current, r2, *_ = result

        # --- R² calculation in log10(|i|) domain (best practice)
        try:
            fit_results = Polcurve.fit_results
            I_model, E_model = np.array(fit_results[0]), np.array(fit_results[1])
            from scipy.interpolate import interp1d
            interp_log_model = interp1d(E_model, np.log10(np.abs(I_model/area_cm2)), bounds_error=False, fill_value="extrapolate")
            logI_obs = np.log10(np.abs(i_fit))
            logI_pred = interp_log_model(E_fit)
            valid = np.isfinite(logI_obs) & np.isfinite(logI_pred)
            logI_obs = logI_obs[valid]
            logI_pred = logI_pred[valid]
            if len(logI_obs) >= 2:
                r2_log = 1 - np.sum((logI_obs - logI_pred) ** 2) / np.sum((logI_obs - np.mean(logI_obs)) ** 2)
            else:
                r2_log = np.nan
        except Exception as ex:
            r2_log = np.nan

        st.success("Fit completed! Tafel log plot below.")
        st.markdown(f"""
- **Ecorr (corrosion potential, fit):** {Ecorr:.4f} V
- **Icorr (corrosion current, fit):** {Icorr:.3e} A
- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec
- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec
- **Limiting current (fit):** {lim_current:.3e} A
- **R² (log10|i|, fit quality):** {r2_log:.4f}
- **OCP (from data):** {OCP:.4f} V
- **Δ(OCP - Ecorr):** {OCP - Ecorr:+.4f} V
""")
        if abs(OCP - Ecorr) > 0.025:
            st.warning("OCP and Ecorr differ by >25 mV. Check data quality or window.")

        # -- Plot: log10(|i|) vs E with overlay in fit region
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(E, np.log10(np.abs(i)), 'o', alpha=0.3, label='All data')
        ax.plot(E_fit, np.log10(np.abs(i_fit)), 'ro', mfc='none', label='Fit region')
        try:
            ax.plot(E_model, np.log10(np.abs(I_model/area_cm2)), 'orange', linewidth=2, label='Fit (model)')
        except Exception:
            st.warning("Could not plot fit overlay directly.")
        ax.set_xlabel("E [V vs Ref]")
        ax.set_ylabel("log10(|i|) (A/cm²)")
        ax.set_title("Tafel Plot: log10(|i|) vs E (fit overlay)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    except Exception as fit_exc:
        st.error(f"Fit failed: {fit_exc}")

st.markdown("""
---
**How to get a good fit:**  
- Select the window covering only the region where your plot is (locally) straight in log10(|i|) — typically ±0.10 to ±0.25 V around Ecorr, avoiding side/reverse/plateau regions.  
- Tune the weights for your chemistry:  
    - Lower W = less plateau influence  
    - Higher w_ac = more Tafel slope influence  
- Use the log plot overlay to judge quality — the model (orange) should follow the red fit-region points.
- Adjust until your fit describes the most important region of your science.  

All fit numbers, OCP, Ecorr, and log-plot overlays are provided for your record and publication.
""")
