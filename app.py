"""
MS300 LPBF Residual Stress — Interactive GUI
=============================================
Models: M3 (GPR)  |  M5 (Physics-Informed GPR, Eagar-Tsai + VED prior)

Run with:
    streamlit run app.py
"""

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import streamlit as st
from scipy import integrate
from scipy.optimize import curve_fit, differential_evolution
from scipy.stats import norm as sp_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MS300 Residual Stress Optimizer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header { color: #666; font-size: 0.95rem; margin-top: 0; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 16px;
        border-left: 4px solid #667eea; margin: 6px 0;
    }
    .warn-card {
        background: #fff3cd; border-radius: 10px; padding: 12px;
        border-left: 4px solid #ffc107; margin: 6px 0;
    }
    .ok-card {
        background: #d4edda; border-radius: 10px; padding: 12px;
        border-left: 4px solid #28a745; margin: 6px 0;
    }
    .section-title { font-size: 1.15rem; font-weight: 600; color: #333; margin-top: 1rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DEFAULT MATERIAL PROPERTIES (MS300 maraging steel)
# ─────────────────────────────────────────────────────────────
class MaterialProperties:
    def __init__(self):
        self.T_liquidus   = 1410.0
        self.T_preheat    = 100.0
        self.delta_T      = self.T_liquidus - self.T_preheat
        self.k            = 25.0
        self.rho          = 8000.0
        self.cp           = 460.0
        self.alpha        = self.k / (self.rho * self.cp)
        self.absorptivity = 0.40
        self.beam_radius  = 35e-6

props = MaterialProperties()

# ─────────────────────────────────────────────────────────────
# EAGAR-TSAI MODEL
# ─────────────────────────────────────────────────────────────
def et_temperature(xi, y, z, power_W, speed_mms, props):
    v = speed_mms / 1000.0
    eta, alpha, k, r = props.absorptivity, props.alpha, props.k, props.beam_radius
    prefactor = (np.sqrt(2) * eta * power_W * alpha) / (np.pi**1.5 * k)

    def integrand(tau):
        if tau <= 0: return 0.0
        dxy = 2 * alpha * tau + r**2
        dz  = 4 * alpha * tau
        if dxy <= 0 or dz <= 0: return 0.0
        exponent = (-(xi**2)/(2*dxy) -(y**2)/(2*dxy)
                    -(z**2)/dz -(v*xi)/(2*alpha) -(v**2*tau)/(4*alpha))
        denom = dxy * np.sqrt(2 * alpha * tau)
        return np.exp(exponent) / denom if denom > 0 else 0.0

    result, _ = integrate.quad(integrand, 1e-9, 5e-3, limit=200,
                                epsabs=1e-8, epsrel=1e-6)
    return prefactor * result


def et_melt_pool(power_W, speed_mms, props):
    target = props.delta_T
    if et_temperature(0, 0, 0, power_W, speed_mms, props) < target:
        return 0.0, 0.0

    y_lo, y_hi = 0.0, 1.5e-3
    for _ in range(50):
        y_mid = (y_lo + y_hi) / 2
        T = et_temperature(0, y_mid, 0, power_W, speed_mms, props)
        if T > target: y_lo = y_mid
        else:          y_hi = y_mid
    width_um = 2 * ((y_lo + y_hi) / 2) * 1e6

    z_lo, z_hi = 0.0, 1.5e-3
    for _ in range(50):
        z_mid = (z_lo + z_hi) / 2
        T = et_temperature(0, 0, z_mid, power_W, speed_mms, props)
        if T > target: z_lo = z_mid
        else:          z_hi = z_mid
    depth_um = ((z_lo + z_hi) / 2) * 1e6
    return width_um, depth_um

# ─────────────────────────────────────────────────────────────
# GPR KERNELS & MODELS
# ─────────────────────────────────────────────────────────────
def make_kernel():
    return (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
            + DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 10.0))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0)))


def gpr_with_prior(X_train, y_train, prior_train,
                   X_pred,  prior_pred, n_restarts=3):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled  = scaler.transform(X_pred)
    disc_train = y_train - prior_train
    gpr = GaussianProcessRegressor(kernel=make_kernel(),
                                   n_restarts_optimizer=n_restarts,
                                   normalize_y=True, random_state=42)
    gpr.fit(X_train_scaled, disc_train)
    disc_pred, std_pred = gpr.predict(X_pred_scaled, return_std=True)
    y_pred = disc_pred + prior_pred
    return y_pred, std_pred, gpr, scaler


def gpr_vanilla(X_train, y_train, X_pred, n_restarts=3):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_pred_scaled  = scaler.transform(X_pred)
    gpr = GaussianProcessRegressor(kernel=make_kernel(),
                                   n_restarts_optimizer=n_restarts,
                                   normalize_y=True, random_state=42)
    gpr.fit(X_train_scaled, y_train)
    pred, std = gpr.predict(X_pred_scaled, return_std=True)
    return pred, std, gpr, scaler

# ─────────────────────────────────────────────────────────────
# TRAINING  (cached so it runs only once)
# ─────────────────────────────────────────────────────────────
FEATURES = ['Power_W', 'Speed_mms', 'Hatch_mm']
TARGET   = 'Res_Stress'
EXCEL_PATH = "residual_stress_maraging_steel_dataset.xlsx"

# Fixed: Added underscore to props argument to prevent hashing
@st.cache_resource(show_spinner="⚙️ Training models on your dataset — please wait…")
def load_and_train(excel_path, _props):
    # ── Load data ────────────────────────────────────────────
    df_raw = pd.read_excel(excel_path, sheet_name='Sheet1')
    df = df_raw[['LP','SS','HS','ED','Stress z=3.861']].copy()
    df.columns = ['Power_W','Speed_mms','Hatch_mm','ED_Jmm3','Res_Stress']
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── ET melt pool using the provided props ─────────────────
    widths, depths = [], []
    for _, row in df.iterrows():
        w, d = et_melt_pool(row['Power_W'], row['Speed_mms'], _props)
        widths.append(w); depths.append(d)
    df['ET_Width_um'] = widths
    df['ET_Depth_um'] = depths
    df['ET_WD_ratio']  = np.where(df['ET_Depth_um'] > 0,
                                   df['ET_Width_um'] / df['ET_Depth_um'], np.nan)

    # ── VED sigmoid prior ─────────────────────────────────────
    VED    = df['ED_Jmm3'].values
    stress = df['Res_Stress'].values

    def sigmoid(x, a, b, c, d):
        return a / (1 + np.exp(-b*(x - c))) + d

    def log_curve(x, a, b):
        return a * np.log(x) + b

    def power_fn(x, a, b):
        return a * np.power(x, b)

    fns = {'sigmoid': (sigmoid, [stress.max()-stress.min(), 0.01, np.median(VED), stress.min()],
                       [(-np.inf,-np.inf,-np.inf,-np.inf),(np.inf,np.inf,np.inf,np.inf)]),
           'log':     (log_curve, [1.0, 0.0], [(-np.inf,-np.inf),(np.inf,np.inf)]),
           'power':   (power_fn,  [1.0, 0.5], [(0,0),(np.inf,np.inf)])}

    fits = {}
    for name, (fn, p0, bounds_) in fns.items():
        try:
            popt, _ = curve_fit(fn, VED, stress, p0=p0, bounds=bounds_, maxfev=50000)
            yhat = fn(VED, *popt)
            ss_res = np.sum((stress - yhat)**2)
            ss_tot = np.sum((stress - stress.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            fits[name] = {'fn': fn, 'popt': popt, 'r2': r2}
        except Exception:
            pass

    best_name = max(fits, key=lambda x: fits[x]['r2'])
    best_fit  = fits[best_name]

    def ved_prior_fn(ved_arr):
        return best_fit['fn'](np.atleast_1d(ved_arr), *best_fit['popt'])

    df['VED_Prior_Stress'] = ved_prior_fn(VED)

    # ── Full-dataset scaler ───────────────────────────────────
    sc_full = MinMaxScaler()
    X_full  = sc_full.fit_transform(df[FEATURES].values)
    y_full  = df[TARGET].values
    ETW_full = df['ET_Width_um'].values
    ETD_full = df['ET_Depth_um'].values
    VP_full  = ved_prior_fn(df['ED_Jmm3'].values)

    # ── M5 Layer 1: Width ─────────────────────────────────────
    w5_full, _, gpr_w, gpr_w_sc = gpr_with_prior(
        X_full, ETW_full, ETW_full, X_full, ETW_full, n_restarts=3)

    # ── M5 Layer 2: Depth ─────────────────────────────────────
    sc5d_full = MinMaxScaler()
    X5d_full  = sc5d_full.fit_transform(
        np.hstack([X_full, w5_full.reshape(-1,1)]))
    d5_full, _, gpr_d, gpr_d_sc = gpr_with_prior(
        X5d_full, ETD_full, ETD_full, X5d_full, ETD_full, n_restarts=3)

    # ── M5 Layer 3: Residual Stress ───────────────────────────
    sc5_full = MinMaxScaler()
    X5_full  = sc5_full.fit_transform(
        np.hstack([X_full, w5_full.reshape(-1,1), d5_full.reshape(-1,1)]))
    disc5 = y_full - VP_full
    g5_full = GaussianProcessRegressor(
        kernel=make_kernel(), n_restarts_optimizer=5,
        normalize_y=True, random_state=42)
    g5_full.fit(X5_full, disc5)

    # ── M3: Vanilla GPR with ET width proxy ───────────────────
    X3_full = np.hstack([X_full, ETW_full.reshape(-1,1)])
    _, _, gpr_m3, gpr_m3_sc = gpr_vanilla(X3_full, y_full, X3_full, n_restarts=3)

    return dict(
        df=df, sc_full=sc_full, sc5d_full=sc5d_full, sc5_full=sc5_full,
        g5_full=g5_full, w5_full=w5_full, d5_full=d5_full,
        gpr_w=gpr_w, gpr_w_sc=gpr_w_sc,
        gpr_d=gpr_d, gpr_d_sc=gpr_d_sc,
        gpr_m3=gpr_m3, gpr_m3_sc=gpr_m3_sc,
        ved_prior_fn=ved_prior_fn,
        best_prior_name=best_name, best_prior_r2=best_fit['r2'],
        fits=fits
    )


# ─────────────────────────────────────────────────────────────
# FAST PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────
def predict_m5(power_W, speed_mms, hatch_mm, models, layer_t_mm, props):
    df   = models['df']
    sc   = models['sc_full']
    sc5d = models['sc5d_full']
    sc5  = models['sc5_full']
    g5   = models['g5_full']
    w5   = models['w5_full']
    vfn  = models['ved_prior_fn']

    ved   = power_W / (speed_mms * hatch_mm * layer_t_mm)
    prior = vfn(np.array([ved]))[0]

    X_pt = sc.transform(np.array([[power_W, speed_mms, hatch_mm]]))

    # Width
    w_corr, _, _, _ = gpr_with_prior(
        sc.transform(df[FEATURES].values), df['ET_Width_um'].values,
        df['ET_Width_um'].values, X_pt,
        np.array([et_melt_pool(power_W, speed_mms, props)[0]]), n_restarts=2)

    # Depth
    X_d_tr = np.hstack([sc.transform(df[FEATURES].values), w5.reshape(-1,1)])
    X_d_new = np.hstack([X_pt, w_corr.reshape(-1,1)])
    d_corr, _, _, _ = gpr_with_prior(
        sc5d.transform(X_d_tr), df['ET_Depth_um'].values,
        df['ET_Depth_um'].values,
        sc5d.transform(X_d_new),
        np.array([et_melt_pool(power_W, speed_mms, props)[1]]), n_restarts=2)

    # Stress
    X5_new = sc5.transform(
        np.hstack([X_pt, w_corr.reshape(-1,1), d_corr.reshape(-1,1)]))
    disc, std = g5.predict(X5_new, return_std=True)
    stress = float(disc[0]) + prior
    return stress, float(std[0]), float(w_corr[0]), float(d_corr[0])


def predict_m3(power_W, speed_mms, hatch_mm, models, props):
    df    = models['df']
    sc    = models['sc_full']
    gpr3  = models['gpr_m3']
    gpr3s = models['gpr_m3_sc']

    w_et, _ = et_melt_pool(power_W, speed_mms, props)
    X_full   = np.hstack([sc.transform(df[FEATURES].values),
                           df['ET_Width_um'].values.reshape(-1,1)])
    X_full_s = gpr3s.transform(X_full)
    gpr3.fit(X_full_s, df['Res_Stress'].values)

    X_pt = sc.transform(np.array([[power_W, speed_mms, hatch_mm]]))
    X_new = gpr3s.transform(np.hstack([X_pt, np.array([[w_et]])]))
    pred, std = gpr3.predict(X_new, return_std=True)
    return float(pred[0]), float(std[0])


# ─────────────────────────────────────────────────────────────
# OPTIMISER
# ─────────────────────────────────────────────────────────────
def run_optimization(models, mode='minimize', target_stress=0.0,
                     layer_t_mm=0.04, top_n=10, confidence=0.95, props=None,
                     progress_cb=None):
    df  = models['df']
    P_min, P_max = df['Power_W'].min(),   df['Power_W'].max()
    v_min, v_max = df['Speed_mms'].min(), df['Speed_mms'].max()
    h_min, h_max = df['Hatch_mm'].min(),  df['Hatch_mm'].max()
    BOUNDS = [(P_min, P_max), (v_min, v_max), (h_min, h_max)]
    NEAR   = (abs(P_max-P_min)*0.05 + abs(v_max-v_min)*0.05) * 0.1

    _best = [1e9]
    _eval = [0]

    def objective(x):
        P, v, h = x
        try:
            stress, std, w_um, d_um = predict_m5(P, v, h, models, layer_t_mm, props)
        except Exception:
            return 1e9
        if mode == 'minimize':
            cost = abs(stress)
        else:
            cost = (stress - target_stress)**2
        gap = abs(stress - (0 if mode=='minimize' else target_stress))
        if gap <= NEAR:
            w_k, w_s = 0.2, 0.15
        else:
            w_k, w_s = 0.3, 0.1
        if w_um == 0.0: cost += 1e6
        if d_um > 0 and (w_um/d_um) < 1.5: cost += abs(cost) * w_k
        cost += w_s * std
        _eval[0] += 1
        if cost < _best[0]: _best[0] = cost
        if progress_cb and _eval[0] % 30 == 0:
            progress_cb(_eval[0])
        return cost

    res = differential_evolution(
        objective, bounds=BOUNDS, strategy='best1bin',
        maxiter=150, popsize=12, tol=1e-5,
        mutation=(0.5,1.5), recombination=0.9,
        seed=42, polish=True, init='sobol', workers=1)

    best_P, best_v, best_h = res.x

    # Neighbourhood scan
    P_lo = max(P_min, best_P - 0.20*(P_max-P_min))
    P_hi = min(P_max, best_P + 0.20*(P_max-P_min))
    v_lo = max(v_min, best_v - 0.20*(v_max-v_min))
    v_hi = min(v_max, best_v + 0.20*(v_max-v_min))
    h_lo = max(h_min, best_h - 0.20*(h_max-h_min))
    h_hi = min(h_max, best_h + 0.20*(h_max-h_min))

    candidates = []
    for P, v, h in itertools.product(
            np.linspace(P_lo, P_hi, 10),
            np.linspace(v_lo, v_hi, 10),
            np.linspace(h_lo, h_hi, 6)):
        try:
            stress, std, w_um, d_um = predict_m5(P, v, h, models, layer_t_mm, props)
        except Exception:
            continue
        if np.isnan(stress) or w_um == 0.0: continue
        ved = P / (v * h * layer_t_mm)
        wd  = w_um/d_um if d_um > 0 else np.nan
        score = abs(stress) if mode=='minimize' else abs(stress-target_stress)
        candidates.append(dict(
            Power_W=P, Speed_mms=v, Hatch_mm=h,
            VED=ved, Stress=stress, Std=std,
            Width_um=w_um, Depth_um=d_um, WD_ratio=wd, Score=score))

    candidates.sort(key=lambda x: x['Score'])
    top = candidates[:top_n]

    best_stress, best_std, best_w, best_d = predict_m5(best_P, best_v, best_h,
                                                        models, layer_t_mm, props)
    best_ved  = best_P / (best_v * best_h * layer_t_mm)
    best_prior = models['ved_prior_fn'](np.array([best_ved]))[0]
    best_wd   = best_w/best_d if best_d > 0 else np.nan
    z = sp_norm.ppf((1+confidence)/2)
    ci_lo = best_stress - z*best_std
    ci_hi = best_stress + z*best_std

    return dict(
        best_P=best_P, best_v=best_v, best_h=best_h,
        best_stress=best_stress, best_std=best_std,
        best_w=best_w, best_d=best_d, best_wd=best_wd,
        best_ved=best_ved, best_prior=best_prior,
        ci_lo=ci_lo, ci_hi=ci_hi,
        candidates=candidates, top=top,
        P_min=P_min, P_max=P_max, v_min=v_min, v_max=v_max,
        h_min=h_min, h_max=h_max
    )


# ─────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────
def plot_optimization_results(opt, mode, target_stress, confidence, top_n):
    candidates = opt['candidates']
    top        = opt['top']
    best_P, best_v, best_h = opt['best_P'], opt['best_v'], opt['best_h']
    best_stress = opt['best_stress']
    best_std    = opt['best_std']
    best_ved    = opt['best_ved']
    best_w      = opt['best_w']
    best_d      = opt['best_d']
    best_wd     = opt['best_wd']
    z           = sp_norm.ppf((1+confidence)/2)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f'Residual Stress Optimization — MS300 LPBF  |  Mode: {mode.upper()}'
        + (f'  Target = {target_stress}' if mode=='target' else ''),
        fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    # 8a: Stress vs Power
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter([c['Power_W'] for c in candidates],
                      [c['Stress']  for c in candidates],
                      c=[c['Speed_mms'] for c in candidates],
                      cmap='plasma', s=20, alpha=0.5, edgecolors='none')
    plt.colorbar(sc1, ax=ax1, label='Speed (mm/s)')
    ax1.scatter(best_P, best_stress, color='red', s=200, zorder=10, marker='*', label='Best')
    for c in top[:5]:
        ax1.scatter(c['Power_W'], c['Stress'], color='orange', s=55, zorder=9,
                    edgecolors='k', linewidths=0.5)
    ax1.set_xlabel('Power (W)'); ax1.set_ylabel('Stress')
    ax1.set_title('Stress vs Power', fontweight='bold')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # 8b: Stress vs Speed
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter([c['Speed_mms'] for c in candidates],
                      [c['Stress']    for c in candidates],
                      c=[c['Hatch_mm'] for c in candidates],
                      cmap='viridis', s=20, alpha=0.5, edgecolors='none')
    plt.colorbar(sc2, ax=ax2, label='Hatch (mm)')
    ax2.scatter(best_v, best_stress, color='red', s=200, zorder=10, marker='*')
    for c in top[:5]:
        ax2.scatter(c['Speed_mms'], c['Stress'], color='orange', s=55, zorder=9,
                    edgecolors='k', linewidths=0.5)
    ax2.set_xlabel('Speed (mm/s)'); ax2.set_ylabel('Stress')
    ax2.set_title('Stress vs Speed', fontweight='bold'); ax2.grid(alpha=0.3)

    # 8c: Stress vs VED coloured by W/D
    ax3 = fig.add_subplot(gs[0, 2])
    sc3 = ax3.scatter([c['VED']    for c in candidates],
                      [c['Stress'] for c in candidates],
                      c=[c['WD_ratio'] for c in candidates],
                      cmap='RdYlGn', s=20, alpha=0.5, edgecolors='none', vmin=1, vmax=6)
    plt.colorbar(sc3, ax=ax3, label='W/D ratio')
    ax3.scatter(best_ved, best_stress, color='red', s=200, zorder=10, marker='*', label='Best')
    ax3.errorbar(best_ved, best_stress, yerr=z*best_std,
                 fmt='none', color='red', capsize=5, linewidth=1.5)
    ax3.set_xlabel('VED (J/mm³)'); ax3.set_ylabel('Stress')
    ax3.set_title('Stress vs VED  (colour = W/D)', fontweight='bold')
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # 8d: 2D heatmap Power x Speed at best hatch
    ax4 = fig.add_subplot(gs[1, 0])
    P_hm = np.linspace(opt['P_min'], opt['P_max'], 20)
    v_hm = np.linspace(opt['v_min'], opt['v_max'], 20)
    Z_hm = np.full((len(v_hm), len(P_hm)), np.nan)
    for ip, P in enumerate(P_hm):
        for iv, v in enumerate(v_hm):
            try:
                s, _, w, _ = predict_m5(P, v, best_h, models_global, 0.04, props)
                if not np.isnan(s) and w > 0:
                    Z_hm[iv, ip] = s
            except Exception:
                pass
    im4 = ax4.contourf(P_hm, v_hm, Z_hm, levels=15, cmap='coolwarm_r')
    plt.colorbar(im4, ax=ax4, label='Residual Stress')
    ax4.scatter(best_P, best_v, color='red', s=200, marker='*', zorder=10,
                label=f'Best h={best_h*1000:.1f}µm')
    ax4.contour(P_hm, v_hm, Z_hm, levels=8, colors='k', linewidths=0.4, alpha=0.3)
    ax4.set_xlabel('Power (W)'); ax4.set_ylabel('Speed (mm/s)')
    ax4.set_title(f'Stress map  (h={best_h*1000:.1f}µm fixed)', fontweight='bold')
    ax4.legend(fontsize=8)

    # 8e: Top-N bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    labels_b = [f'#{i+1}  P={c["Power_W"]:.0f}  v={c["Speed_mms"]:.0f}  h={c["Hatch_mm"]*1000:.0f}µm'
                for i, c in enumerate(top)]
    ax5.barh(range(len(top)), [c['Stress'] for c in top],
             xerr=[c['Std'] for c in top],
             color=['crimson'] + ['steelblue']*(len(top)-1),
             alpha=0.75, error_kw={'ecolor':'k','capsize':3,'linewidth':1})
    ax5.set_yticks(range(len(top))); ax5.set_yticklabels(labels_b, fontsize=7.5)
    ax5.invert_yaxis()
    ax5.axvline(best_stress, color='crimson', linestyle='--', linewidth=1, alpha=0.6)
    ax5.set_xlabel('Predicted Residual Stress')
    ax5.set_title(f'Top-{top_n} candidates  (red = best)', fontweight='bold')
    ax5.grid(alpha=0.3, axis='x')

    # 8f: Melt pool cross-section
    ax6 = fig.add_subplot(gs[1, 2])
    if best_w > 0 and best_d > 0:
        ax6.add_patch(Ellipse((0, 0), width=best_w, height=best_d*2,
                      edgecolor='royalblue', facecolor='lightskyblue', alpha=0.5,
                      linewidth=2, label=f'W={best_w:.0f}  D={best_d:.0f}  W/D={best_wd:.2f}'))
        ax6.set_xlim(-best_w*0.85, best_w*0.85)
        ax6.set_ylim(-best_d*1.7,  best_d*1.7)
        ax6.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Surface')
        ax6.set_xlabel('Width (µm)'); ax6.set_ylabel('Depth (µm)')
        ax6.set_aspect('equal', adjustable='datalim')
        ax6.legend(fontsize=8, loc='lower right')
    ax6.set_title('ET Melt Pool — Best Solution', fontweight='bold')
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prediction_result(power_W, speed_mms, hatch_mm,
                           stress_m5, std_m5, stress_m3, std_m3,
                           width_um, depth_um, ved, prior, models, layer_t_mm=0.04):
    df       = models['df']
    vfn      = models['ved_prior_fn']

    fig = plt.figure(figsize=(18, 9))
    fig.suptitle(f'M5 Prediction  |  P={power_W:.0f}W  v={speed_mms:.0f}mm/s  h={hatch_mm*1000:.1f}µm',
                 fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # ── 1: VED prior curve + this point
    ax1 = fig.add_subplot(gs[0, 0])
    VED_line = np.linspace(df['ED_Jmm3'].min()*0.9, df['ED_Jmm3'].max()*1.1, 200)
    ax1.scatter(df['ED_Jmm3'], df['Res_Stress'], c='steelblue', s=40,
                alpha=0.6, edgecolors='k', linewidths=0.3, label='Data')
    ax1.plot(VED_line, vfn(VED_line), 'r--', linewidth=2, label=f'Physics prior')
    ax1.scatter(ved, prior, color='gold', s=150, zorder=10, edgecolors='k',
                linewidths=1.5, marker='D', label=f'Prior={prior:.2f}')
    ax1.scatter(ved, stress_m5, color='red', s=200, zorder=11, marker='*',
                label=f'M5={stress_m5:.2f}')
    ax1.set_xlabel('VED (J/mm³)'); ax1.set_ylabel('Residual Stress')
    ax1.set_title('VED Prior + M5 Prediction', fontweight='bold')
    ax1.legend(fontsize=7.5); ax1.grid(alpha=0.3)

    # ── 2: Processing map with prediction marker
    ax2 = fig.add_subplot(gs[0, 1])
    sc_ = ax2.scatter(df['Speed_mms'], df['Power_W'], c=df['Res_Stress'],
                      s=60, edgecolors='k', linewidths=0.3, cmap='coolwarm_r')
    plt.colorbar(sc_, ax=ax2, label='Residual Stress')
    ax2.scatter(speed_mms, power_W, color='lime', s=250, zorder=10,
                marker='*', edgecolors='k', label='Your input')
    ax2.set_xlabel('Speed (mm/s)'); ax2.set_ylabel('Power (W)')
    ax2.set_title('Processing Map (Stress)', fontweight='bold')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── 3: M3 vs M5 comparison bar
    ax3 = fig.add_subplot(gs[0, 2])
    models_list = ['M3 (GPR)', 'M5 (PI-GPR)']
    stresses    = [stress_m3, stress_m5]
    stds        = [std_m3,    std_m5]
    colors      = ['#e74c3c', '#9b59b6']
    bars = ax3.bar(models_list, stresses, yerr=stds, color=colors, alpha=0.75,
                   error_kw={'ecolor':'k','capsize':8,'linewidth':1.5}, width=0.5)
    ax3.set_ylabel('Predicted Residual Stress')
    ax3.set_title('M3 vs M5 Predictions', fontweight='bold')
    ax3.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    for bar, s, sd in zip(bars, stresses, stds):
        ax3.text(bar.get_x()+bar.get_width()/2, s + (sd if s>=0 else -sd) + 2,
                 f'{s:.2f}\n±{sd:.2f}', ha='center', va='bottom', fontsize=9)
    ax3.grid(alpha=0.3, axis='y')

    # ── 4: Confidence interval visualization
    ax4 = fig.add_subplot(gs[1, 0])
    z = sp_norm.ppf(0.975)
    ci_lo_m5 = stress_m5 - z*std_m5
    ci_hi_m5 = stress_m5 + z*std_m5
    x = np.linspace(stress_m5 - 4*std_m5, stress_m5 + 4*std_m5, 400)
    y = sp_norm.pdf(x, loc=stress_m5, scale=std_m5)
    ax4.plot(x, y, 'purple', linewidth=2)
    mask = (x >= ci_lo_m5) & (x <= ci_hi_m5)
    ax4.fill_between(x, y, where=mask, alpha=0.3, color='purple', label='95% CI')
    ax4.axvline(stress_m5, color='red', linewidth=1.5, linestyle='--', label=f'Mean={stress_m5:.2f}')
    ax4.set_xlabel('Residual Stress'); ax4.set_ylabel('Probability Density')
    ax4.set_title('M5 Prediction Distribution', fontweight='bold')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # ── 5: Melt pool cross-section with colored background text
    ax5 = fig.add_subplot(gs[1, 1])
    if width_um > 0 and depth_um > 0:
        wd = width_um / depth_um
        ax5.add_patch(Ellipse((0, 0), width=width_um, height=depth_um*2,
                      edgecolor='royalblue', facecolor='lightskyblue', alpha=0.5,
                      linewidth=2, label=f'W={width_um:.0f}µm  D={depth_um:.0f}µm  W/D={wd:.2f}'))
        ax5.set_xlim(-width_um*0.85, width_um*0.85)
        ax5.set_ylim(-depth_um*1.7,  depth_um*1.7)
        ax5.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Surface')
        ax5.set_aspect('equal', adjustable='datalim')
        ax5.legend(fontsize=8, loc='lower right')
        col = 'orange' if wd < 1.5 else 'green'
        ax5.set_title(f'ET Melt Pool  (W/D={"⚠️ " if wd<1.5 else "✓ "}{wd:.2f})',
                      fontweight='bold', color=col)
    else:
        ax5.text(0.5, 0.5, '⚠️ No melting\ndetected', ha='center', va='center',
                 transform=ax5.transAxes, fontsize=14, color='red')
        ax5.set_title('ET Melt Pool', fontweight='bold')
    ax5.set_xlabel('Width (µm)'); ax5.set_ylabel('Depth (µm)')
    ax5.grid(alpha=0.3)

    # ── 6: Stress vs hatch sweep (Power, Speed fixed)
    ax6 = fig.add_subplot(gs[1, 2])
    df_ref = models['df']
    h_sweep = np.linspace(df_ref['Hatch_mm'].min(), df_ref['Hatch_mm'].max(), 25)
    s_sweep, sd_sweep = [], []
    for h in h_sweep:
        try:
            s, sd, _, _ = predict_m5(power_W, speed_mms, h, models, layer_t_mm, props)
            s_sweep.append(s); sd_sweep.append(sd)
        except Exception:
            s_sweep.append(np.nan); sd_sweep.append(np.nan)
    s_arr  = np.array(s_sweep)
    sd_arr = np.array(sd_sweep)
    ax6.plot(h_sweep*1000, s_arr, 'purple', linewidth=2)
    ax6.fill_between(h_sweep*1000, s_arr-sd_arr, s_arr+sd_arr,
                     alpha=0.2, color='purple', label='±1σ')
    ax6.axvline(hatch_mm*1000, color='red', linestyle='--', linewidth=1.5,
                label=f'Current h={hatch_mm*1000:.1f}µm')
    ax6.set_xlabel('Hatch Spacing (µm)'); ax6.set_ylabel('Predicted Stress')
    ax6.set_title('Stress vs Hatch  (P & v fixed)', fontweight='bold')
    ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR — Data upload & settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ MS300 Stress GUI")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload dataset (.xlsx)", type=["xlsx"],
        help="Must contain: LP, SS, HS, ED, 'Stress z=3.861'")

    if uploaded:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        tmp.write(uploaded.read()); tmp.close()
        data_path = tmp.name
    else:
        data_path = EXCEL_PATH

    # Fixed layer thickness display
    st.markdown("### Layer thickness")
    st.info("📏 Layer thickness: **40 µm** (fixed)")
    layer_t_mm = 0.04  # Fixed at 40µm

    st.markdown("### Confidence level")
    confidence = st.slider("Confidence (%)", 80, 99, 95) / 100.0

    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("🔵 **M3** — GPR with ET width proxy")
    st.markdown("🟣 **M5** — Physics-Informed GPR\n(ET + VED prior + layered GPR)")
    st.markdown("---")
    st.caption("Built on Eagar-Tsai thermal model + Bayesian GPR")

# ─────────────────────────────────────────────────────────────
# MATERIAL PROPERTIES TAB
# ─────────────────────────────────────────────────────────────
with st.expander("📊 Material Properties (Advanced Settings)", expanded=False):
    st.markdown("### Customize Material Properties")
    st.markdown("Default values are for MS300 maraging steel. Modify for other materials.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        use_custom = st.checkbox("Use custom material properties", value=False)
    
    if use_custom:
        col1, col2, col3 = st.columns(3)
        with col1:
            T_liquidus = st.number_input("Liquidus Temperature (°C)", value=1410.0, step=10.0)
            T_preheat = st.number_input("Preheat Temperature (°C)", value=100.0, step=10.0)
            absorptivity = st.number_input("Laser Absorptivity", value=0.40, step=0.01, min_value=0.0, max_value=1.0)
        with col2:
            k = st.number_input("Thermal Conductivity (W/m·K)", value=25.0, step=1.0)
            rho = st.number_input("Density (kg/m³)", value=8000.0, step=100.0)
            cp = st.number_input("Specific Heat Capacity (J/kg·K)", value=460.0, step=10.0)
        with col3:
            beam_radius = st.number_input("Beam Radius (µm)", value=35.0, step=5.0) / 1e6
            st.markdown("---")
            st.markdown("**Calculated Values:**")
            delta_T = T_liquidus - T_preheat
            alpha = k / (rho * cp)
            st.metric("ΔT (°C)", f"{delta_T:.1f}")
            st.metric("Thermal Diffusivity (m²/s)", f"{alpha:.2e}")
        
        # Update props with custom values
        props.T_liquidus = T_liquidus
        props.T_preheat = T_preheat
        props.delta_T = delta_T
        props.k = k
        props.rho = rho
        props.cp = cp
        props.alpha = alpha
        props.absorptivity = absorptivity
        props.beam_radius = beam_radius
        st.warning("⚠️ Material properties changed. The GPR models are still trained on MS300 data. For accurate predictions with this material, please upload a dataset for this material.")
    else:
        # Reset to MS300 defaults
        props = MaterialProperties()
        st.info("Using MS300 maraging steel properties:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Liquidus Temperature", f"{props.T_liquidus}°C")
            st.metric("Absorptivity", f"{props.absorptivity}")
            st.metric("Thermal Conductivity", f"{props.k} W/m·K")
        with col2:
            st.metric("Density", f"{props.rho} kg/m³")
            st.metric("Specific Heat", f"{props.cp} J/kg·K")
            st.metric("Beam Radius", f"{props.beam_radius*1e6:.0f} µm")

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
try:
    # Pass props with underscore to prevent hashing issues
    models_global = load_and_train(data_path, props)
    df_data = models_global['df']
    TRAINED = True
    st.success(f"✅ Models loaded successfully with {len(df_data)} data points!")
except Exception as e:
    TRAINED = False
    st.error(f"❌ Could not load/train models: {e}")
    st.info("Place `residual_stress_maraging_steel_dataset.xlsx` in the same folder, or upload it in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">MS300 LPBF Residual Stress Optimizer</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Physics-Informed Bayesian ML — {len(df_data)} data points loaded  •  '
            f'Prior: {models_global["best_prior_name"].upper()} (R²={models_global["best_prior_r2"]:.3f})</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict  (single point)",
    "🎯 Optimize  (find best params)",
    "📊 Dataset & Model Info"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE POINT PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Process Parameters")
    
    # Add option for manual input or sliders
    input_mode = st.radio("Input method:", ["Sliders", "Manual Entry"], horizontal=True)
    
    if input_mode == "Sliders":
        c1, c2, c3 = st.columns(3)
        with c1:
            pred_P = st.slider("Laser Power (W)",
                               float(df_data.Power_W.min()), float(df_data.Power_W.max()),
                               float(df_data.Power_W.median()), step=5.0)
        with c2:
            pred_v = st.slider("Scan Speed (mm/s)",
                               float(df_data.Speed_mms.min()), float(df_data.Speed_mms.max()),
                               float(df_data.Speed_mms.median()), step=50.0)
        with c3:
            pred_h = st.slider("Hatch Spacing (µm)",
                               float(df_data.Hatch_mm.min()*1000), float(df_data.Hatch_mm.max()*1000),
                               float(df_data.Hatch_mm.median()*1000), step=5.0)
        pred_h_mm = pred_h / 1000.0
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pred_P = st.number_input("Laser Power (W)",
                                     min_value=float(df_data.Power_W.min()), 
                                     max_value=float(df_data.Power_W.max()),
                                     value=float(df_data.Power_W.median()), step=5.0)
        with c2:
            pred_v = st.number_input("Scan Speed (mm/s)",
                                     min_value=float(df_data.Speed_mms.min()),
                                     max_value=float(df_data.Speed_mms.max()),
                                     value=float(df_data.Speed_mms.median()), step=50.0)
        with c3:
            pred_h_um = st.number_input("Hatch Spacing (µm)",
                                        min_value=float(df_data.Hatch_mm.min()*1000),
                                        max_value=float(df_data.Hatch_mm.max()*1000),
                                        value=float(df_data.Hatch_mm.median()*1000), step=5.0)
            pred_h_mm = pred_h_um / 1000.0

    run_pred = st.button("▶ Run Prediction", type="primary", use_container_width=True)

    if run_pred:
        with st.spinner("Computing ET melt pool + GPR predictions…"):
            w_um, d_um = et_melt_pool(pred_P, pred_v, props)
            ved_val    = pred_P / (pred_v * pred_h_mm * layer_t_mm)
            prior_val  = models_global['ved_prior_fn'](np.array([ved_val]))[0]

            try:
                s_m5, sd_m5, w_pred, d_pred = predict_m5(
                    pred_P, pred_v, pred_h_mm, models_global, layer_t_mm, props)
            except Exception as ex:
                st.error(f"M5 prediction failed: {ex}"); st.stop()

            try:
                s_m3, sd_m3 = predict_m3(pred_P, pred_v, pred_h_mm, models_global, props)
            except Exception:
                s_m3, sd_m3 = float('nan'), float('nan')

        z = sp_norm.ppf((1+confidence)/2)
        ci_lo = s_m5 - z*sd_m5
        ci_hi = s_m5 + z*sd_m5
        wd    = w_pred/d_pred if d_pred > 0 else float('nan')

        # Metrics row
        st.markdown("### Results")
        m1, m2, m3_, m4 = st.columns(4)
        m1.metric("M5 Stress (MPa)", f"{s_m5:.3f}", f"±{sd_m5:.3f}")
        m2.metric("M3 Stress (MPa)", f"{s_m3:.3f}" if not np.isnan(s_m3) else "—",
                  f"±{sd_m3:.3f}" if not np.isnan(sd_m3) else "")
        m3_.metric("VED (J/mm³)",  f"{ved_val:.1f}")
        m4.metric("Physics Prior", f"{prior_val:.3f}")

        r1, r2 = st.columns(2)
        r1.metric("Melt Pool Width (µm)",  f"{w_pred:.1f}")
        r2.metric("Melt Pool Depth (µm)", f"{d_pred:.1f}")

        # Color-coded stability message
        if w_pred > 0 and d_pred > 0 and wd < 1.5:
            st.markdown(f'<div class="warn-card">⚠️ <b>Keyholing Risk</b> — W/D = {wd:.2f} &lt; 1.5. '
                        f'Consider reducing power or increasing speed.</div>', unsafe_allow_html=True)
        elif w_pred == 0:
            st.markdown('<div class="warn-card">⚠️ <b>No Melting Detected</b> — ET predicts the '
                        'laser energy is too low to reach liquidus temperature.</div>', unsafe_allow_html=True)
        else:
            # Changed to green background for stable melt pool
            st.markdown(f'<div class="ok-card" style="background: #64945a; border-left-color: #28a745;">✅ <b>Stable melt pool</b> — W/D = {wd:.2f} ≥ 1.5. '
                        f'No keyholing expected.</div>', unsafe_allow_html=True)

        st.markdown(f"**{confidence*100:.0f}% Confidence Interval:** [{ci_lo:.3f}, {ci_hi:.3f}]")

        st.markdown("### Prediction Charts")
        with st.spinner("Generating charts…"):
            fig_pred = plot_prediction_result(
                pred_P, pred_v, pred_h_mm,
                s_m5, sd_m5, s_m3, sd_m3,
                w_pred, d_pred, ved_val, prior_val,
                models_global, layer_t_mm)
        st.pyplot(fig_pred, use_container_width=True)
        plt.close(fig_pred)

# ══════════════════════════════════════════════════════════════
# TAB 2 — OPTIMIZATION
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Optimization Settings")
    oc1, oc2 = st.columns(2)
    with oc1:
        opt_mode = st.radio("Optimization Mode",
                            ["minimize", "target"],
                            captions=["Find lowest absolute stress",
                                      "Hit a specific target stress value"])
    with oc2:
        target_val = st.number_input(
            "Target Stress (only for 'target' mode)",
            value=-300.0, step=10.0,
            disabled=(opt_mode == 'minimize'))
        top_n = st.slider("Number of top candidates", 5, 20, 10)

    st.markdown(f"""
    **Search bounds from your dataset:**
    - Power: {df_data.Power_W.min():.0f} – {df_data.Power_W.max():.0f} W
    - Speed: {df_data.Speed_mms.min():.0f} – {df_data.Speed_mms.max():.0f} mm/s
    - Hatch: {df_data.Hatch_mm.min()*1000:.1f} – {df_data.Hatch_mm.max()*1000:.1f} µm
    """)

    run_opt = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

    if run_opt:
        prog_bar = st.progress(0, text="Running Differential Evolution…")
        prog_txt = st.empty()

        def update_prog(n_eval):
            frac = min(n_eval / (150*12*3), 0.95)
            prog_bar.progress(frac, text=f"Evaluations: {n_eval}")

        with st.spinner("Optimizing — this takes ~1–2 minutes…"):
            opt_res = run_optimization(
                models_global,
                mode=opt_mode,
                target_stress=target_val,
                layer_t_mm=layer_t_mm,
                top_n=top_n,
                confidence=confidence,
                props=props,
                progress_cb=update_prog)

        prog_bar.progress(1.0, text="✅ Done!")

        # Best solution
        st.markdown("### 🏆 Best Solution")
        b1, b2, b3 = st.columns(3)
        b1.metric("Laser Power",    f"{opt_res['best_P']:.1f} W")
        b2.metric("Scan Speed",     f"{opt_res['best_v']:.1f} mm/s")
        b3.metric("Hatch Spacing",  f"{opt_res['best_h']*1000:.1f} µm")

        b4, b5, b6, b7 = st.columns(4)
        b4.metric("Predicted Stress", f"{opt_res['best_stress']:.4f}",
                  f"±{opt_res['best_std']:.4f}")
        b5.metric("VED",  f"{opt_res['best_ved']:.2f} J/mm³")
        b6.metric("Width", f"{opt_res['best_w']:.1f} µm")
        b7.metric("Depth", f"{opt_res['best_d']:.1f} µm")

        best_wd = opt_res['best_wd']
        if not np.isnan(best_wd) and best_wd < 1.5:
            safer = [c for c in opt_res['top'] if c['WD_ratio'] >= 1.5]
            msg = (f"⚠️ <b>Keyholing risk</b> — W/D = {best_wd:.2f} &lt; 1.5. "
                   + (f"Safer alternative: P={safer[0]['Power_W']:.1f}W  "
                      f"v={safer[0]['Speed_mms']:.1f}  h={safer[0]['Hatch_mm']*1000:.1f}µm  "
                      f"stress={safer[0]['Stress']:.4f}"
                      if safer else "No keyhole-safe alternative in top candidates."))
            st.markdown(f'<div class="warn-card">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ok-card" style="background: #d4edda; border-left-color: #28a745;">✅ W/D = {best_wd:.2f} — healthy melt pool, '
                        f'no keyholing expected.</div>', unsafe_allow_html=True)

        st.markdown(f"**{confidence*100:.0f}% CI:** [{opt_res['ci_lo']:.4f}, {opt_res['ci_hi']:.4f}]")

        # Top-N table
        st.markdown(f"### Top-{top_n} Candidates")
        rows = []
        for i, c in enumerate(opt_res['top']):
            flags = ""
            if c['WD_ratio'] < 1.5: flags += "⚠️ KEYHOLE "
            if c['Std'] > 2*opt_res['best_std']: flags += "🔶 HIGH-UNC"
            rows.append({
                "Rank": f"{'🥇' if i==0 else i+1}",
                "Power (W)":  f"{c['Power_W']:.1f}",
                "Speed (mm/s)": f"{c['Speed_mms']:.1f}",
                "Hatch (µm)": f"{c['Hatch_mm']*1000:.1f}",
                "VED":        f"{c['VED']:.1f}",
                "Stress":     f"{c['Stress']:.4f}",
                "±σ":         f"{c['Std']:.4f}",
                "W/D":        f"{c['WD_ratio']:.2f}",
                "Flags":      flags
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Plots
        st.markdown("### Optimization Charts")
        with st.spinner("Generating 6-panel optimization plot…"):
            fig_opt = plot_optimization_results(
                opt_res, opt_mode, target_val, confidence, top_n)
        st.pyplot(fig_opt, use_container_width=True)
        plt.close(fig_opt)

# ══════════════════════════════════════════════════════════════
# TAB 3 — DATASET INFO
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Dataset Summary")
    st.dataframe(df_data.describe().round(3), use_container_width=True)
    st.markdown("### Sample Data")
    st.dataframe(df_data.head(20), use_container_width=True)

    st.markdown("### Exploratory Data Analysis")
    fig_eda = plt.figure(figsize=(18, 10))
    fig_eda.suptitle('LPBF Residual Stress — EDA', fontsize=14, fontweight='bold')
    gs_eda = gridspec.GridSpec(2, 3, figure=fig_eda, hspace=0.4, wspace=0.35)

    ax = fig_eda.add_subplot(gs_eda[0, 0])
    ax.hist(df_data['Res_Stress'], bins=20, edgecolor='white')
    ax.axvline(df_data['Res_Stress'].mean(), linestyle='--', linewidth=1.8,
               label=f"Mean={df_data['Res_Stress'].mean():.2f}")
    ax.set_title('Stress Distribution', fontweight='bold'); ax.legend(); ax.set_xlabel('Residual Stress')

    ax = fig_eda.add_subplot(gs_eda[0, 1:])
    sc_ = ax.scatter(df_data['ED_Jmm3'], df_data['Res_Stress'],
                     c=df_data['Hatch_mm'], s=55, edgecolors='k', linewidths=0.4)
    plt.colorbar(sc_, ax=ax, label='Hatch (mm)')
    ax.set_xlabel('VED (J/mm³)'); ax.set_ylabel('Residual Stress')
    ax.set_title('VED vs Stress', fontweight='bold'); ax.grid(alpha=0.3)

    ax = fig_eda.add_subplot(gs_eda[1, :2])
    sc_ = ax.scatter(df_data['Speed_mms'], df_data['Power_W'],
                     c=df_data['Res_Stress'], s=65, edgecolors='k', linewidths=0.4,
                     cmap='coolwarm_r')
    plt.colorbar(sc_, ax=ax).set_label('Residual Stress')
    ax.set_xlabel('Speed (mm/s)'); ax.set_ylabel('Power (W)')
    ax.set_title('Processing Map', fontweight='bold'); ax.grid(alpha=0.3)

    ax = fig_eda.add_subplot(gs_eda[1, 2])
    hs_vals = sorted(df_data['Hatch_mm'].unique())
    groups  = [df_data[df_data['Hatch_mm']==hs]['Res_Stress'].values for hs in hs_vals]
    bp = ax.boxplot(groups, patch_artist=True, labels=[f'{h:.2f}' for h in hs_vals])
    for patch in bp['boxes']: patch.set_alpha(0.7)
    ax.set_xlabel('Hatch (mm)'); ax.set_ylabel('Residual Stress')
    ax.set_title('Stress by Hatch', fontweight='bold'); ax.grid(alpha=0.3, axis='y')

    st.pyplot(fig_eda, use_container_width=True)
    plt.close(fig_eda)

    st.markdown("### VED Prior Fit")
    vfn = models_global['ved_prior_fn']
    VED_plot = np.linspace(df_data['ED_Jmm3'].min()*0.9, df_data['ED_Jmm3'].max()*1.1, 300)
    fig_prior, ax_p = plt.subplots(figsize=(8, 4))
    ax_p.scatter(df_data['ED_Jmm3'], df_data['Res_Stress'],
                 c='steelblue', s=50, edgecolors='k', linewidths=0.3, label='Data', zorder=5)
    ax_p.plot(VED_plot, vfn(VED_plot), 'r--', linewidth=2,
              label=f'{models_global["best_prior_name"].upper()} prior  R²={models_global["best_prior_r2"]:.3f}')
    ax_p.set_xlabel('VED (J/mm³)'); ax_p.set_ylabel('Residual Stress')
    ax_p.set_title('VED-Based Physics Prior', fontweight='bold')
    ax_p.legend(); ax_p.grid(alpha=0.3)
    st.pyplot(fig_prior, use_container_width=True)
    plt.close(fig_prior)