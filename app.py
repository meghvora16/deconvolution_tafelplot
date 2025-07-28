import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil

# --- Import polcurvefit, stop if not available ---
try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit not found. Please install it with 'pip install polcurvefit'.")
    st.stop()

st.title("Mixed-Control Corrosion (Tafel+Diffusion) Fit - Hardened Settings")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file (columns: 'Potential applied (V)', 'WE(1).Current (A)')", 
    type=["csv", "xlsx"]
)

plot_output_folder = 'Visualization_mixed_control_fit'
if os.path.exists(plot_output_folder):
    shutil.rmtree(plot_output_folder)
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    # --- Read file safely ---
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()

    # --- Hardcoded columns for your Gamry/EC-lab export ---
    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    for col in [pot_col, cur_col]:
        if col not in df.columns:
            st.error(f"Expected column '{col}' not found! Columns in file: {df.columns.tolist()}")
            st.stop()

    E = df[pot_col].astype(float).values
    I = df[cur_col].astype(float).values

    st.write("**Measured Data (first 20 rows):**")
    st.write(df[[pot_col, cur_col]].head(20))

    # --- Surface area (user can change for current density normalization) ---
    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=1e-8, value=1.0, format="%.4f")

    # --- Remove NaN/zeros to prevent fit issues ---
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    # --- Calculate OCP (mean/rest value in early data, or at I closest to zero) ---
    ind_OCP = np.argmin(np.abs(I_clean))
    OCP = E_clean[ind_OCP]

    st.info(f"Open Circuit Potential (OCP): {OCP:.4f} V (from measured data, where |I| min)")

    # --- Use polcurvefit ---
    Polcurve = polcurvefit(E_clean, I_clean, sample_surface=area_cm2)
    Ecorr_fit = Polcurve._find_Ecorr()

    # --- HARDCODED fitting window & weights: CHANGE HERE if needed
    # Window is relative to Ecorr: [min_V_vs_Ecorr, max_V_vs_Ecorr]
    window_vs_Ecorr = [-0.12, 0.10]   # <-- adjust for your kinetic/diffusion region in volts
    w_ac = 0.10                       # <-- kinetic weighting (0.05-0.15 typical)
    W = 15                            # <-- plateau weighting (10-40 typical; lower for more Tafel bias)

    st.info(f"Fit window [vs. Ecorr]: {window_vs_Ecorr} V | Hardcoded weights: w_ac={w_ac}, W={W}")

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window_vs_Ecorr,
            apply_weight_distribution=True,
            w_ac=w_ac,
            W=W
        )
        [_, _], Ecorr, Icorr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Fit completed!")
        st.write(f"- **Corrosion potential (Ecorr, fitted):** {Ecorr:.4f} V")
        st.write(f"- **Corrosion current (Icorr):** {Icorr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope * 1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope * 1000:.2f} mV/dec")
        st.write(f"- **Limiting current (Ilim):** {lim_current:.3e} A")
        st.write(f"- **Goodness of fit (R², log-I):** {r2:.4f}")

        st.write(f"- **Difference OCP - Ecorr:** {(OCP-Ecorr):.4f} V")
        if abs(OCP-Ecorr) > 0.020:
            st.warning("OCP and fitted Ecorr differ by >20 mV. Check data stability or fitting window.")

        # --- Save and display plots ---
        try:
            Polcurve.plotting(output_folder=plot_output_folder)
            st.markdown("### Fit Plots:")
            for plot_file in sorted(os.listdir(plot_output_folder)):
                if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    st.image(os.path.join(plot_output_folder, plot_file), caption=plot_file)
        except Exception as plotexc:
            st.warning(f"Plotting failed: {plotexc}")

    except Exception as fit_exc:
        st.error(f"Fit failed: {fit_exc}")

st.markdown("""
---
**How this works:**
- Uses your potential/current columns directly.
- Normalizes by area for true current density (editable above).
- OCP is taken from the point closest to zero current.
- Ecorr, Icorr, slopes, and R² are extracted from automated, hard-coded-window mixed-control fit.
- All weighting and fit region are set in the script (edit the lines with `window_vs_Ecorr`, `w_ac`, `W`).
- Plots show your measured and fitted polarization.

**Tip:** If the fit is not ideal, try changing these three lines in the script for fine-tuning:
