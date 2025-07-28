import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# --- Import polcurvefit, show message if absent
try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit is not installed. Please run 'pip install polcurvefit'")
    st.stop()

st.title("Mixed-Control Polarization Curve Fit (van Ede & Angst algorithm)")

st.markdown("""
This app analyzes your polarization curve using the **automated, human-minimized algorithm** of van Ede & Angst (2023, Cement and Concrete Research).
Just upload your data, set the area, and get Ecorr, Icorr, Tafel slopes, limiting current, and publication-ready plots.  
**No manual window selection!**
""")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file (columns: 'Potential applied (V)', 'WE(1).Current (A)')",
    type=["csv", "xlsx"]
)

plot_output_folder = 'Auto_MixedControl_Fit_Plots'
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

    # --- Required columns: hardcoded for your data structure
    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    for col in [pot_col, cur_col]:
        if col not in df.columns:
            st.error(f"Missing column '{col}'! Found: {df.columns.tolist()}")
            st.stop()

    st.write("**Data preview (first 20 rows):**")
    st.dataframe(df[[pot_col, cur_col]].head(20))

    # --- Area input
    area_cm2 = st.number_input('Sample surface area (cm², for current density)', min_value=1e-6, value=1.0, format="%.4f")

    # --- Data cleaning
    E = df[pot_col].astype(float).values
    I = df[cur_col].astype(float).values
    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E = E[mask]
    I = I[mask]

    if len(E) < 7:
        st.error("Too few valid data points after cleaning. Check your file!")
        st.stop()

    # --- OCP from data
    idx_OCP = np.argmin(np.abs(I))
    OCP = E[idx_OCP]
    st.info(f"Open Circuit Potential (OCP, from data): {OCP:.4f} V (point with |I| minimal)")

    # --- Automated mixed-control fit as per van Ede & Angst: fit the whole region
    Polcurve = polcurvefit(E, I, sample_surface=area_cm2)
    e_corr = Polcurve._find_Ecorr()
    window = [np.min(E) - e_corr, np.max(E) - e_corr]  # use full range (minimizes human factors)
    result = Polcurve.mixed_pol_fit(window=window, apply_weight_distribution=True)

    [_, _], Ecorr, Icorr, anodic_slope, cathodic_slope, lim_current, r2, *_ = result

    st.success("Fit completed (fully automated - van Ede & Angst method)!")
    st.markdown(f"""
- **Ecorr (corrosion potential, fit):** {Ecorr:.4f} V  
- **Icorr (corrosion current, fit):** {Icorr:.3e} A  
- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec  
- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec  
- **Limiting current (fit):** {lim_current:.3e} A  
- **R² (log|I|, fit quality):** {r2:.4f}  
- **OCP (from data):** {OCP:.4f} V  
- **Δ(OCP - Ecorr):** {OCP - Ecorr:+.4f} V
""")
    if abs(OCP - Ecorr) > 0.025:
        st.warning("OCP and Ecorr differ by >25 mV. Check data and fit.")

    # --- Plots
    try:
        Polcurve.plotting(output_folder=plot_output_folder)
        st.markdown("### Fit Plots (for publication/QC)")
        for plot_file in sorted(os.listdir(plot_output_folder)):
            if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                st.image(os.path.join(plot_output_folder, plot_file), caption=plot_file)
    except Exception as plotexc:
        st.warning(f"Plotting failed: {plotexc}")

st.markdown("""
----
**How this works:**  
- Uses the van Ede & Angst (2023) algorithm (no manual window selection)
- Analyzes the full curve region, minimizes human bias  
- Fitted Icorr, Ecorr, slopes, limiting current, and plots are all exported  
- See [the paper](https://www.sciencedirect.com/science/article/pii/S0008884623001869) and [polcurvefit documentation](https://github.com/uleangst/polcurvefit)
""")
