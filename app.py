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

st.title("Auto-center Mixed-Control Polarization Curve Fit (Ecorr-centered Tafel Window)")

st.markdown("""
Upload your polarization curve data (CSV or Excel, columns: 'Potential applied (V)', 'WE(1).Current (A)').
The fit window will be **automatically centered on the fitted Ecorr** (±0.15 V by default, or as much as your data allows).
You can still tune weights for a physically meaningful, high-quality fit!
Plot is **log10(|i|) vs E** and key values are shown below.
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

    # --- Preliminary fit to get Ecorr
    Polcurve_preview = polcurvefit(E, I, sample_surface=area_cm2)
    Ecorr = Polcurve_preview._find_Ecorr()

    # --- Set fit window centered on Ecorr
    tafel_width = st.number_input('Tafel window width (V, ± around Ecorr)', min_value=0.05, max_value=0.40, value=0.15, step=0.01, format="%.2f")
    wmin = max(E.min(), Ecorr - tafel_width)
    wmax = min(E.max(), Ecorr + tafel_width)
    st.info(f"Auto-fit window: {wmin:.3f} V to {wmax:.3f} V (centered on Ecorr = {Ecorr:.4f} V)")

    region_mask = (E >= wmin) & (E <= wmax)
    E_fit = E[region_mask]
    I_fit = I[region_mask]
    i_fit = i[region_mask]

    # --- Fit weights (user-adjustable)
    w_ac = st.slider("Weight for active (Tafel) region (w_ac)", 0.01, 0.20, 0.08, step=0.01, format="%.2f")
    W = st.slider("Weight for diffusion (plateau) region (W)", 5, 120, 20, step=1)
    st.info(f"Fitting weights: w_ac={w_ac}, W={W}")

    if len(E_fit) < 7:
        st.error("Too few points in auto-selected window for fit. Try increasing window width or check data range.")
        st.stop()

    # --- Fit
    Polcurve = polcurvefit(E_fit, I_fit, sample_surface=area_cm2)
    Ecorr_fit = Polcurve._find_Ecorr()
    window = [np.min(E_fit) - Ecorr_fit, np.max(E_fit) - Ecorr_fit]
    try:
        result = Polcurve.mixed_pol_fit(window=window, apply_weight_distribution=True, w_ac=w_ac, W=W)
        [_, _], Ecorr_out, Icorr, anodic_slope, cathodic_slope, lim_current, _, *_ = result

        st.success("Fit completed! Tafel log plot below.")
        st.markdown(f"""
- **Ecorr (corrosion potential, fit, V):** {Ecorr_out:.4f}
- **Icorr (corrosion current, fit, A):** {Icorr:.3e}
- **Anodic Tafel slope (V/dec):** {anodic_slope:.4f}
- **Cathodic Tafel slope (V/dec):** {cathodic_slope:.4f}
- **Limiting current (fit, A):** {lim_current:.3e}
""")
        # -- Plot: log10(|i|) vs E with overlay in fit region
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(E, np.log10(np.abs(i)), 'o', alpha=0.3, label='All data')
        ax.plot(E_fit, np.log10(np.abs(i_fit)), 'ro', mfc='none', label='Fit region')
        try:
            fit_results = Polcurve.fit_results
            I_model, E_model = np.array(fit_results[0]), np.array(fit_results[1])
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
**Notes:**  
- The fit window is always **centered on the current Ecorr** for objective Tafel analysis.
- Tafel window width can be tuned above (increase if you want more data, decrease to focus narrowly).
- Tune the weights for your chemistry:
    - Lower W = less plateau influence
    - Higher w_ac = more Tafel slope influence  
- Use the log plot overlay to judge fit quality — the model (orange) should follow the red fit-region points.
""")
