import streamlit as st
import pandas as pd
import numpy as np
import os

try:
    from polcurvefit import polcurvefit
except ImportError:
    st.error("polcurvefit is not installed. Please ensure polcurvefit.py or the package is present.")
    st.stop()

st.title("Weighted Automated Fit à la van Ede & Angst (2022)")

uploaded_file = st.file_uploader("Upload CSV or Excel (Potential, Current)", type=["csv", "xlsx"])
plot_output_folder = 'Visualization_weighted_fit'
os.makedirs(plot_output_folder, exist_ok=True)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format."); st.stop()

    pot_col = 'Potential applied (V)'
    cur_col = 'WE(1).Current (A)'
    E = df[pot_col].values
    I = df[cur_col].values

    mask = (~np.isnan(E)) & (~np.isnan(I)) & (np.abs(I) > 0)
    E_clean = E[mask]
    I_clean = I[mask]

    area_cm2 = st.number_input('Sample surface area (cm²)', min_value=0.0001, value=1.0)
    area_m2 = area_cm2 * 1e-4

    # Parameters as in van Ede & Angst (see Suppl. Info for recommended values)
    w_ac = st.number_input("Width of high-weight region around $E_{corr}$ (V)", value=0.04, min_value=0.001, max_value=1.0, step=0.01, format="%.3f")
    W = st.number_input("Weight percentage in $w_{ac}$ region (%)", value=80, min_value=10, max_value=100, step=5)
    
    Polcurve = polcurvefit(E_clean, I_clean, sample_surface=area_m2)
    e_corr = Polcurve._find_Ecorr()
    # Use the full available data span relative to Ecorr
    window = [np.min(E_clean) - e_corr, np.max(E_clean) - e_corr]

    st.info(f'Fitting with weighting: {W}% of the total fit weight is assigned to the region ±{w_ac} V around Ecorr (as in van Ede & Angst 2022, CORROSION).')

    try:
        fit_result = Polcurve.mixed_pol_fit(
            window,
            apply_weight_distribution=True,
            w_ac=w_ac,
            W=W
        )
        [_, _], E_corr_fit, I_corr, anodic_slope, cathodic_slope, lim_current, r2, *_ = fit_result

        st.success("Weighted fit completed!")
        st.write(f"- **E_corr:** {E_corr_fit:.4f} V")
        st.write(f"- **I_corr:** {I_corr:.3e} A")
        st.write(f"- **Anodic Tafel slope:** {anodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Cathodic Tafel slope:** {cathodic_slope*1000:.2f} mV/dec")
        st.write(f"- **Limiting current (I_lim):** {lim_current:.3e} A")

        # Show the fit plot
        Polcurve.plotting(output_folder=plot_output_folder)
        st.markdown("### Fitted plot (with weighting):")
        import matplotlib.pyplot as plt
        for plot_file in sorted(os.listdir(plot_output_folder)):
            if plot_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                plot_path = os.path.join(plot_output_folder, plot_file)
                st.image(plot_path, caption=plot_file)

        # R² on log(|I|) (for reporting)
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

    except Exception as fit_exc:
        st.error(f"Fit failed: {fit_exc}")

    st.markdown("""
    ---
    **This app follows the weighting recommendations of van Ede & Angst (2022, CORROSION): the region near $E_{corr}$ is fit with higher weight, minimizing user subjectivity and improving fit robustness for real-world, non-ideal curves. Try different $w_{ac}$ and $W$ values to explore sensitivity.**
    """)
