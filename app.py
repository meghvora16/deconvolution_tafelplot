import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil

# 1. Import polcurvefit globally
try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit class not found. Please install it using 'pip install polcurvefit'.")
    st.stop()

st.title("Global Mixed-Control Tafel Fit (Final, Publication-Ready)")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file (columns: Potential applied (V), WE(1).Current (A))", 
    type=["csv", "xlsx"]
)

plot_output_folder = 'Visualization_mixed_control_fit'
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # 2. File reading
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload .csv or .xlsx.")
        st.stop()

    # 3. Columns - hardcoded for your data
    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    for col in [pot_col, cur_col]:
        if col not in df.columns:
            st.error(f"Expected column '{col}' not found! Columns: {df.columns.tolist()}")
            st.stop()

    E_raw = df[pot_col].values.astype(float)
    I_raw = df[cur_col].values.astype(float)

    st.write("**First 20 rows (V, A):**")
    st.dataframe(df[[pot_col, cur_col]].head(20))

    # 4. Surface area (cm², for current density normalization, user-editable)
    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=1e-6, value=1.0, format="%.4f")

    # 5. Clean data
    mask = (~np.isnan(E_raw)) & (~np.isnan(I_raw)) & (np.abs(I_raw) > 0)
    E = E_raw[mask]
    I = I_raw[mask]

    if len(E) < 7:
        st.error("Too few valid data points after cleaning. Please check your file.")
        st.stop()

    # 6. OCP from data: point with minimum |I| (or average of first 10 points)
    idx_ocp = np.argmin(np.abs(I))
    OCP = E[idx_ocp]
    st.info(f"Open Circuit Potential (OCP, from data): {OCP:.4f} V (where |I| is minimal)")

    # 7. POLCURVEFIT initialization (area in cm²!)
    Polcurve = polcurvefit(E, I, sample_surface=area_cm2)
    e_corr_guess = Polcurve._find_Ecorr()

    # 8. HARDCODED WINDOW & WEIGHTS (edit here for your data/fit)
    # window is relative to Ecorr
    window_vs_Ecorr = [-0.12, 0.10]  # -120 mV to +100 mV from Ecorr
    w_ac = 0.10                      # Weight for active (Tafel) region
    W = 15                           # Weight for diffusion/plateau

    st.info(
        f"Fitting window: {window_vs_Ecorr} V vs Ecorr (initial guess Ecorr = {e_corr_guess:.4f} V)"
        f"\nWeights: w_ac={w_ac}, W={W} (edit these 3 lines in code to tune)"
    )

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window=window_vs_Ecorr,      # Relative to Ecorr
            apply_weight_distribution=True,
            w_ac=w_ac,
            W=W
        )
        [_, _], Ecorr, Icorr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Fit completed!")
        st.markdown(f"""
        - **Ecorr (corrosion potential, fit):** {Ecorr:.4f} V  
        - **Icorr (corrosion current, fit):** {Icorr:.3e} A  
        - **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec  
        - **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec  
        - **Limiting current (fit):** {lim_current:.3e} A  
        - **R² (log|I|):** {r2:.4f}  
        - **OCP (from data):** {OCP:.4f} V  
        - **Δ(OCP - Ecorr):** {OCP - Ecorr:.4f} V
        """)
        if abs(OCP - Ecorr) > 0.020:
            st.warning("OCP and Ecorr differ by >20 mV. Check data stability, pre-polarization drift, or fitting window.")

        # 10. Save and show plots
        try:
            Polcurve.plotting(output_folder=plot_output_folder)
            st.markdown("### Fit Plots")
            for plot_file in sorted(os.listdir(plot_output_folder)):
                if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    st.image(os.path.join(plot_output_folder, plot_file), caption=plot_file)
        except Exception as plotexc:
            st.warning(f"Plotting failed: {plotexc}")

    except Exception as fit_exc:
        st.error(f"Fit failed: {fit_exc}")

st.markdown("""
---
#### Instructions / Notes:
- Upload your .csv/.xlsx file with columns: **Potential applied (V)**, **WE(1).Current (A)**
- Edit surface area if needed.
- The fitting window and weights are **hard-coded for reproducibility**; _edit `window_vs_Ecorr`, `w_ac`, and `W` in the code if you want to tune the fit "region" or aggressiveness_.
- OCP is extracted from your data (minimum |I|); Ecorr, Icorr, slopes, and R² from the model fit.
- You get full-curve plots for your records.

For advanced fitting, contact a corrosion analyst or see the [polcurvefit manual](https://github.com/uleangst/polcurvefit).
""")
