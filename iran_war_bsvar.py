"""
================================================================================
 COUNTERFACTUAL STRUCTURAL BAYESIAN VAR (BSVAR) ANALYSIS
 Impact of Iran–Israel–US War Oil Price Shock on Nigeria's Economy
================================================================================
 Methodology: Conditional Forecasts in Large Bayesian VARs with Multiple
              Equality and Inequality Constraints
              (Chan, Poon & Zhu, 2025 — Journal of Econometrics)

 Transmission channel:
   Oil Price → Exchange Rate → Inflation → Fiscal Revenue
             → External Reserves → Liquidity

 Budget Baseline (Nigeria 2026):
   Oil benchmark  : $64.85/bbl
   Production     : 1.84 mb/d
   Exchange rate  : ₦1,400/$
   Oil revenue    : $43.55 bn
   Ext. reserves  : $50.45 bn

 Scenarios:
   S0 – Baseline    : Oil ≈ $64.85/bbl (no war shock)
   S1 – Mild        : Oil ≈ $82/bbl    (4–6 week war)
   S2 – Moderate    : Oil ≈ $100/bbl   (3–6 month war)
   S3 – Severe      : Oil ≈ $130/bbl   (>6 months, Hormuz closure)

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.linalg import cholesky, solve_triangular, inv
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── COLOUR PALETTE ──────────────────────────────────────────────────────────
COLS = {
    'S0': '#2C4A6E',   # steel blue  — baseline
    'S1': '#2A7D4F',   # green       — mild
    'S2': '#C9A227',   # amber       — moderate
    'S3': '#C8392B',   # red         — severe
    'shade_S0': '#2C4A6E22',
    'shade_S1': '#2A7D4F22',
    'shade_S2': '#C9A22722',
    'shade_S3': '#C8392B22',
    'hist': '#555555',
    'bg': '#FAFAF8',
    'grid': '#E0DBD4',
    'text': '#0D0F14',
}
SCENARIO_LABELS = {
    'S0': 'S0: Baseline ($64.85/bbl)',
    'S1': 'S1: Mild Shock ($82/bbl, 4–6 wks)',
    'S2': 'S2: Moderate Shock ($100/bbl, 3–6 mo)',
    'S3': 'S3: Severe Shock ($130/bbl, Hormuz)',
}
OIL_PRICES = {'S0': 64.85, 'S1': 82.0, 'S2': 100.0, 'S3': 130.0}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.facecolor': COLS['bg'],
    'figure.facecolor': COLS['bg'],
    'axes.edgecolor': COLS['grid'],
    'axes.labelcolor': COLS['text'],
    'xtick.color': COLS['text'],
    'ytick.color': COLS['text'],
    'grid.color': COLS['grid'],
    'grid.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — DATA LOADING & PREPARATION                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 72)
print(" BSVAR ANALYSIS: Iran–Israel–US War Oil Shock on Nigeria")
print("=" * 72)
print("\n[1] Loading and preparing data...")

# ── 1A. Core Variables (monthly) ─────────────────────────────────────────────
cv_raw = pd.read_excel('/mnt/user-data/uploads/Core_Variables.xlsx', header=None)

# Extract data block starting from row 2
cv = cv_raw.iloc[2:].copy()
cv.columns = ['Date', 'Inflation', 'FiscalRevenue', 'ExchRate', 'OilPrice_WTI', 'ExtReserves']
cv = cv.dropna(subset=['Date'])
cv['Date'] = pd.to_datetime(cv['Date'], errors='coerce')
cv = cv.dropna(subset=['Date'])
cv = cv.set_index('Date').sort_index()
cv = cv.apply(pd.to_numeric, errors='coerce')
cv = cv.dropna(subset=['OilPrice_WTI'])   # keep rows with at least oil price

# Drop last row if it has no date (the trailing NaT)
cv = cv[cv.index.notna()]

# Compute log-transformed fiscal revenue (in billions NGN)
cv['FiscRevBn'] = cv['FiscalRevenue'] / 1e9
cv['ExtResBn']  = cv['ExtReserves']  / 1e9

print(f"  Core variables: {cv.index[0].strftime('%b %Y')} → {cv.index[-1].strftime('%b %Y')}  "
      f"({len(cv)} obs)")

# ── 1B. Liquidity (daily → monthly avg) ─────────────────────────────────────
liq_raw = pd.read_excel('/mnt/user-data/uploads/Liquidity.xlsx', header=None)
liq = liq_raw.iloc[3:, [1, 3]].copy()
liq.columns = ['Date', 'Liquidity']
liq['Date'] = pd.to_datetime(liq['Date'], errors='coerce')
liq['Liquidity'] = pd.to_numeric(liq['Liquidity'], errors='coerce')
liq = liq.dropna()
liq = liq.set_index('Date').sort_index()

# Monthly average
liq_monthly = liq.resample('MS').mean()
liq_monthly.columns = ['Liquidity']

print(f"  Liquidity:      {liq_monthly.index[0].strftime('%b %Y')} → "
      f"{liq_monthly.index[-1].strftime('%b %Y')}  ({len(liq_monthly)} obs)")

# ── 1C. Merge ────────────────────────────────────────────────────────────────
df = cv[['Inflation', 'ExchRate', 'OilPrice_WTI', 'FiscRevBn', 'ExtResBn']].join(
     liq_monthly, how='left')

# Forward-fill short gaps (≤ 2 months)
df = df.ffill(limit=2)

# Keep rows from 2010 onward with all key vars
df = df.loc['2010':].dropna(subset=['Inflation', 'ExchRate', 'OilPrice_WTI',
                                     'FiscRevBn', 'ExtResBn'])

# Fill any remaining Liquidity NaNs with col mean
df['Liquidity'] = df['Liquidity'].fillna(df['Liquidity'].mean())

print(f"  Merged dataset: {df.index[0].strftime('%b %Y')} → "
      f"{df.index[-1].strftime('%b %Y')}  ({len(df)} obs, "
      f"{df.shape[1]} variables)")
print(f"  Variables: {list(df.columns)}")

# ── 1D. Print descriptive statistics ─────────────────────────────────────────
print("\n  Descriptive Statistics (full sample):")
desc = df.describe().round(2)
print(desc.to_string())

# ── 1E. Last observed values (anchor for counterfactuals) ────────────────────
last_obs   = df.iloc[-1]
last_date  = df.index[-1]
# Budget baseline anchors
BUDGET_OIL   = 64.85
BUDGET_EXCH  = 1400.0
BUDGET_INFL  = 15.0          # projected H1 2026
BUDGET_OREV  = 43.55e9       # USD
BUDGET_PROD  = 1.84          # mb/d
BUDGET_RESRV = 50.45e9       # USD
BUDGET_LIQ   = df['Liquidity'].iloc[-12:].mean()  # recent 12-month avg


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — BAYESIAN VAR SPECIFICATION & ESTIMATION                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[2] Estimating Bayesian VAR model...")

# ── Variable ordering (Cholesky identification — recursive SVAR) ─────────────
# Causal ordering follows transmission channel:
#   Oil → ExchRate → Inflation → FiscRev → ExtRes → Liquidity
VAR_NAMES  = ['OilPrice_WTI', 'ExchRate', 'Inflation', 'FiscRevBn', 'ExtResBn', 'Liquidity']
VAR_LABELS = ['Oil Price (WTI, $/bbl)', 'Exch. Rate (₦/$)',
              'Inflation (%)', 'Fiscal Revenue (₦ bn)',
              'External Reserves ($bn)', 'Banking Liquidity (₦ bn)']
K = len(VAR_NAMES)

Y_full = df[VAR_NAMES].values.astype(float)

def build_var_matrices(Y, p=2):
    """Build Y_dep and X_lag matrices for VAR(p)."""
    T, k = Y.shape
    T_eff = T - p
    Y_dep = Y[p:].copy()           # (T_eff × k)

    X_list = [np.ones((T_eff, 1))]
    for lag in range(1, p + 1):
        X_list.append(Y[p - lag: T - lag])
    X_lag = np.hstack(X_list)      # (T_eff × (1 + p*k))
    return Y_dep, X_lag, T_eff

P_LAG = 2         # VAR lag order
Y_dep, X_lag, T_eff = build_var_matrices(Y_full, p=P_LAG)
k  = K
kx = X_lag.shape[1]   # 1 + p*k

# ── Minnesota Prior (Litterman) ───────────────────────────────────────────────
# Prior: B ~ MN(B0, Omega_B, Sigma)  with conjugate Sigma ~ IW(S0, nu0)
# Chan-Poon-Zhu spirit: we use the standard Normal-Inverse-Wishart conjugate
# prior which admits closed-form posterior draws (Gibbs sampler).

lambda1 = 0.2    # overall tightness
lambda2 = 0.5    # cross-variable tightness
lambda3 = 1.0    # lag decay

def minnesota_prior(Y, p, lambda1=0.2, lambda2=0.5, lambda3=1.0):
    """
    Build Minnesota prior moments for each equation.
    Returns vectorised B0, Omega_B (block-diagonal across equations)
    in the SUR form for Normal-IW.
    """
    T, k  = Y.shape
    kx    = 1 + p * k

    # OLS residual variances (for scaling)
    sig2 = np.zeros(k)
    for i in range(k):
        yi   = Y[p:, i]
        xi   = np.column_stack([np.ones(len(yi))] +
                               [Y[p - l : T - l, i] for l in range(1, p + 1)])
        b, _, _, _ = np.linalg.lstsq(xi, yi, rcond=None)
        e         = yi - xi @ b
        sig2[i]   = np.var(e, ddof=xi.shape[1])

    # Prior mean: random walk on own first lag, zero otherwise
    B0 = np.zeros((kx, k))
    for i in range(k):
        B0[1 + i, i] = 1.0    # first own lag = 1

    # Prior variance for each element B[r, i]
    # Var(B_ij^(l)) = (lambda1 / l^lambda3)^2  * sig2[i] / sig2[j]   (cross)
    #               = (lambda1 / l^lambda3)^2                          (own)
    Omega_B = np.zeros((kx, k))
    Omega_B[0, :] = 1e6   # intercept — diffuse

    for lag in range(1, p + 1):
        for j in range(k):   # variable j in equation
            for i in range(k):   # equation i
                if i == j:
                    scale = (lambda1 / lag**lambda3)**2
                else:
                    scale = (lambda1 * lambda2 / lag**lambda3)**2 * sig2[i] / sig2[j]
                row = 1 + (lag - 1) * k + j
                Omega_B[row, i] = scale

    # IW prior: S0 = diag(sig2), nu0 = k + 2
    S0  = np.diag(sig2 * (k + 2))
    nu0 = k + 2

    return B0, Omega_B, S0, nu0, sig2

B0_prior, Omega_B_prior, S0_prior, nu0_prior, sig2_eq = minnesota_prior(
    Y_full, P_LAG, lambda1, lambda2, lambda3)

# ── Posterior (Normal-Inverse-Wishart closed form) ────────────────────────────
def posterior_NIW(Y_dep, X_lag, B0, Omega_B, S0, nu0):
    """
    Conjugate Normal-IW posterior for VAR.
    Returns B_post, Omega_post, S_post, nu_post.
    """
    T, k  = Y_dep.shape
    kx    = X_lag.shape[1]

    # Posterior precision for B (per equation): use vec trick with diag(Omega_B)
    # Since Minnesota is equation-by-equation:
    B_post    = np.zeros((kx, k))
    Omega_post = np.zeros((kx, k))

    for i in range(k):
        Om_i  = Omega_B[:, i]           # prior variance vector (kx,)
        Om_i  = np.where(Om_i < 1e-10, 1e-10, Om_i)
        prec_prior = 1.0 / Om_i         # prior precision
        # Posterior: use OLS precision + prior precision
        # Simplified (diagonal Omega → equation-by-equation):
        XtX   = X_lag.T @ X_lag
        Xty   = X_lag.T @ Y_dep[:, i]
        b0_i  = B0[:, i]
        V_post_inv = np.diag(prec_prior) + XtX / max(sig2_eq[i], 1e-6)
        V_post     = np.linalg.pinv(V_post_inv)
        b_post_i   = V_post @ (np.diag(prec_prior) @ b0_i +
                                XtX @ np.linalg.lstsq(X_lag, Y_dep[:, i], rcond=None)[0] /
                                max(sig2_eq[i], 1e-6))
        B_post[:, i]    = b_post_i
        Omega_post[:, i] = np.diag(V_post)

    # S_post (for IW draw of Sigma)
    E = Y_dep - X_lag @ B_post
    S_post  = S0 + E.T @ E
    nu_post = nu0 + T

    return B_post, Omega_post, S_post, nu_post

B_post, Omega_post, S_post, nu_post = posterior_NIW(
    Y_dep, X_lag, B0_prior, Omega_B_prior, S0_prior, nu0_prior)

print(f"  VAR({P_LAG}) estimated on {T_eff} effective observations")
print(f"  Posterior nu = {nu_post}, S_post diagonal: "
      f"{np.diag(S_post).round(2)[:4]} ...")

# ── OLS companion estimates (for impulse responses) ──────────────────────────
B_ols = np.linalg.lstsq(X_lag, Y_dep, rcond=None)[0]   # (kx × k)
E_ols = Y_dep - X_lag @ B_ols
Sigma_ols = (E_ols.T @ E_ols) / (T_eff - kx)

# Structural identification — Cholesky of Sigma (recursive, ordering above)
try:
    A0_inv = cholesky(Sigma_ols, lower=True)   # lower-triangular impact matrix
except Exception:
    A0_inv = np.linalg.cholesky(Sigma_ols + np.eye(k) * 1e-8)

print(f"  OLS Sigma diagonal (residual variances):\n  "
      f"{np.diag(Sigma_ols).round(4)}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — GIBBS SAMPLER (Posterior draws)                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[3] Running Gibbs sampler for posterior draws...")

MCMC_DRAWS  = 5000
MCMC_BURN   = 2000
THIN        = 2
KEEP        = (MCMC_DRAWS - MCMC_BURN) // THIN

B_draws     = np.zeros((KEEP, kx, k))
Sigma_draws = np.zeros((KEEP, k, k))

# Initialise
B_curr     = B_post.copy()
Sigma_curr = Sigma_ols.copy()

keep_idx = 0
for draw in range(MCMC_DRAWS):
    # ── Step 1: Draw B | Sigma, Y (equation-by-equation) ────────────────────
    Sigma_inv = np.linalg.pinv(Sigma_curr)
    B_new = np.zeros_like(B_curr)
    for i in range(k):
        Om_i = Omega_B_prior[:, i]
        Om_i = np.where(Om_i < 1e-10, 1e-10, Om_i)
        V_prior_inv = np.diag(1.0 / Om_i)
        b0_i = B0_prior[:, i]
        sii  = Sigma_curr[i, i]
        V_post_inv  = V_prior_inv + X_lag.T @ X_lag / sii
        V_post_i    = np.linalg.pinv(V_post_inv)
        b_post_i    = V_post_i @ (V_prior_inv @ b0_i +
                                   X_lag.T @ Y_dep[:, i] / sii)
        L = np.linalg.cholesky(V_post_i + np.eye(kx) * 1e-10)
        B_new[:, i] = b_post_i + L @ np.random.randn(kx)
    B_curr = B_new

    # ── Step 2: Draw Sigma | B, Y (Inverse-Wishart) ─────────────────────────
    E_curr  = Y_dep - X_lag @ B_curr
    S_draw  = S0_prior + E_curr.T @ E_curr
    # IW draw: Sigma ~ IW(S_draw, nu_post)
    # Standard method: Sigma = (W W')^{-1} * S_draw where W ~ N(0,I), size nu_post × k
    try:
        Lc = np.linalg.cholesky(S_draw)
    except np.linalg.LinAlgError:
        Lc = np.linalg.cholesky(S_draw + np.eye(k) * 1e-6)
    W  = np.random.randn(nu_post, k) @ np.linalg.inv(Lc).T
    try:
        Sigma_curr = np.linalg.inv(W.T @ W)
    except np.linalg.LinAlgError:
        pass   # keep previous draw
    # Ensure PD
    Sigma_curr = (Sigma_curr + Sigma_curr.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma_curr))
    if min_eig < 1e-8:
        Sigma_curr += (1e-8 - min_eig) * np.eye(k)

    # ── Store ────────────────────────────────────────────────────────────────
    if draw >= MCMC_BURN and (draw - MCMC_BURN) % THIN == 0:
        B_draws[keep_idx]     = B_curr
        Sigma_draws[keep_idx] = Sigma_curr
        keep_idx += 1

B_draws     = B_draws[:keep_idx]
Sigma_draws = Sigma_draws[:keep_idx]
print(f"  Gibbs sampler complete. Kept {keep_idx} posterior draws.")

# Posterior means
B_mean     = B_draws.mean(0)
Sigma_mean = Sigma_draws.mean(0)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — IMPULSE RESPONSE FUNCTIONS                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[4] Computing structural impulse responses (Cholesky)...")

def companion_matrix(B, k, p):
    """Build companion matrix from VAR coefficients (excluding intercept)."""
    B_lags = B[1:].T   # (k × p*k)
    F = np.zeros((k * p, k * p))
    F[:k, :] = B_lags
    if p > 1:
        F[k:, :-k] = np.eye(k * (p - 1))
    return F

def irf_cholesky(B, Sigma, k, p, horizon=24):
    """
    Compute IRFs via Cholesky identification.
    Returns IRF array: (horizon+1, k_response, k_shock)
    """
    try:
        A0_inv = np.linalg.cholesky(Sigma + np.eye(k) * 1e-8)
    except np.linalg.LinAlgError:
        A0_inv = np.eye(k)

    F = companion_matrix(B, k, p)
    n = k * p
    e_k = np.zeros((n, k))
    e_k[:k, :] = np.eye(k)

    IRF = np.zeros((horizon + 1, k, k))
    Fh  = np.eye(n)
    for h in range(horizon + 1):
        IRF[h] = (e_k.T @ Fh @ e_k)[:k, :k] @ A0_inv
        Fh = Fh @ F
    return IRF

H_IRF   = 24
IRF_mean = irf_cholesky(B_mean, Sigma_mean, k, P_LAG, H_IRF)

# Posterior bands for IRFs
IRF_draws = np.zeros((keep_idx, H_IRF + 1, k, k))
for d in range(keep_idx):
    IRF_draws[d] = irf_cholesky(B_draws[d], Sigma_draws[d], k, P_LAG, H_IRF)

IRF_lo  = np.percentile(IRF_draws, 16, axis=0)
IRF_hi  = np.percentile(IRF_draws, 84, axis=0)

print(f"  IRFs computed for {H_IRF}-month horizon with 68% posterior bands.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — CONDITIONAL FORECAST ENGINE                                   ║
# ║  Chan–Poon–Zhu (2025): Constraints imposed via Algorithm 1                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[5] Setting up conditional forecast engine (Chan–Poon–Zhu, 2025)...")

"""
Chan, Poon & Zhu (2025) approach:
  Given BVAR: y_t = B' x_t + u_t,  u_t ~ N(0, Sigma)

  Conditional forecast satisfies:
    R vec(Y_fcast) = r    [equality constraints]
    Q vec(Y_fcast) <= q   [inequality constraints — not binding here]

  Algorithm 1 (CPZ2025):
    1. Draw unconditional forecast trajectory from predictive posterior
    2. Project onto constraint manifold using QR / pseudoinverse
    3. Repeat for each posterior draw → posterior predictive bands

  We apply equality constraints for each scenario:
    - OilPrice path is fixed at scenario-specific values
    - Other variables are freely forecast conditional on oil path
  This is implemented via the partition-based conditional expectation:
    y2|y1 ~ N(mu2 + Sigma21 Sigma11^{-1}(y1 - mu1), Sigma22.1)
"""

H_FC  = 18   # 18-month forecast horizon (2026–2027)

def build_forecast_prior(B, Sigma, Y_full, p, H):
    """
    Build unconditional joint predictive distribution over forecast horizon.
    Returns: mu_fcast (H*k,), Sigma_fcast (H*k × H*k)
    """
    T, k  = Y_full.shape
    kx    = 1 + p * k

    # Build X_T (the conditioning lagged values)
    x_last = np.concatenate([[1.0],
                              Y_full[-1],         # lag 1
                              Y_full[-2] if p >= 2 else np.zeros(k)])
    # For p > 2, extend; here p=2
    A = B[1:1+k].T   # coefficients on lag-1  (k × k)
    C = B[1+k:1+2*k].T if p >= 2 else np.zeros((k, k))   # lag-2
    intercept = B[0]  # (k,)

    # Iterative forecast mean
    mu     = np.zeros((H, k))
    Y_last = Y_full[-p:].copy()   # (p × k)

    for h in range(H):
        x_h     = np.concatenate([[1.0], Y_last[-1], Y_last[-2]])
        mu_h    = B[0] + B[1:1+k].T @ Y_last[-1] + \
                  (B[1+k:1+2*k].T @ Y_last[-2] if p >= 2 else 0)
        mu[h]   = mu_h
        Y_last  = np.vstack([Y_last[1:], mu_h])

    mu_vec = mu.flatten()   # (H*k,)

    # Joint forecast covariance — build via MA representation
    # Sigma_fcast[hk + i, h'k + j] = Cov(y_{T+h,i}, y_{T+h',j})
    # Use companion matrix propagation
    F  = companion_matrix(B, k, p)
    n  = k * p
    Sigma_e = np.zeros((n, n))
    Sigma_e[:k, :k] = Sigma

    Sigma_fc = np.zeros((H * k, H * k))
    # Cov(y_{T+h}, y_{T+h'}) = e_k' F^{h-1} Sigma_e (F^{h'-1})' e_k  (h <= h')
    e_sel = np.zeros((n, k))
    e_sel[:k, :] = np.eye(k)

    Fh_list = [np.eye(n)]
    for h in range(1, H):
        Fh_list.append(Fh_list[-1] @ F)

    for h1 in range(H):
        for h2 in range(h1, H):
            C_h1h2 = e_sel.T @ Fh_list[h1] @ Sigma_e @ Fh_list[h2].T @ e_sel
            Sigma_fc[h1*k:(h1+1)*k, h2*k:(h2+1)*k] = C_h1h2
            Sigma_fc[h2*k:(h2+1)*k, h1*k:(h1+1)*k] = C_h1h2.T

    return mu_vec, Sigma_fc

def conditional_forecast_cpz(B, Sigma, Y_full, p, H, oil_path, n_draws=500):
    """
    Chan–Poon–Zhu (2025) Algorithm 1:
    Conditional forecast given equality constraint on OilPrice_WTI path.

    oil_path: (H,) array of constrained oil prices
    Returns: draws (n_draws × H × k)
    """
    k    = Y_full.shape[1]
    oil_idx = 0    # OilPrice is variable 0 in VAR_NAMES ordering
    mu_vec, Sigma_fc = build_forecast_prior(B, Sigma, Y_full, p, H)

    # Partition indices
    # y1 = oil prices (constrained), y2 = remaining variables (free)
    idx1 = [h * k + oil_idx for h in range(H)]         # (H,)
    idx2 = [h * k + j for h in range(H)
            for j in range(k) if j != oil_idx]         # (H*(k-1),)

    mu1       = mu_vec[idx1]
    mu2       = mu_vec[idx2]
    S11       = Sigma_fc[np.ix_(idx1, idx1)]
    S12       = Sigma_fc[np.ix_(idx1, idx2)]
    S21       = S12.T
    S22       = Sigma_fc[np.ix_(idx2, idx2)]

    # Regularise
    S11_reg = S11 + np.eye(len(idx1)) * 1e-6

    try:
        S11_inv = np.linalg.pinv(S11_reg)
    except Exception:
        S11_inv = np.eye(len(idx1)) / (np.diag(S11_reg).mean() + 1e-6)

    # Ensure oil_path is exactly length H
    oil_path_h = np.asarray(oil_path).flatten()[:H]
    if len(oil_path_h) < H:
        oil_path_h = np.pad(oil_path_h, (0, H - len(oil_path_h)), 'edge')

    # Conditional mean of free variables
    delta      = oil_path_h - mu1         # (H,) — constraint residual
    mu2_cond   = mu2 + S21 @ S11_inv @ delta   # (H*(k-1),)

    # Conditional covariance
    S22_cond   = S22 - S21 @ S11_inv @ S12
    # Regularise
    S22_cond   = (S22_cond + S22_cond.T) / 2
    min_eig    = np.min(np.linalg.eigvalsh(S22_cond))
    if min_eig < 1e-8:
        S22_cond += (1e-8 - min_eig + 1e-6) * np.eye(S22_cond.shape[0])

    # Draw conditional samples
    try:
        L22 = np.linalg.cholesky(S22_cond)
        samples2 = mu2_cond[:, None] + L22 @ np.random.randn(len(idx2), n_draws)
    except np.linalg.LinAlgError:
        samples2 = np.tile(mu2_cond[:, None], (1, n_draws))

    # Assemble full draws: (H × k × n_draws)
    draws = np.zeros((H, k, n_draws))
    draws[:, oil_idx, :] = oil_path_h[:, None]    # oil is fixed at constraint
    for d_idx, draw_i in enumerate(range(n_draws)):
        free_draw = samples2[:, draw_i]         # (H*(k-1),)
        pos = 0
        for h in range(H):
            for j in range(k):
                if j != oil_idx:
                    draws[h, j, draw_i] = free_draw[pos]
                    pos += 1

    return draws.transpose(2, 0, 1)   # (n_draws × H × k)


# ─── Oil price paths for each scenario ───────────────────────────────────────
def build_oil_path(scenario, H, last_oil=None):
    """
    Build H-month oil price path for each scenario.
    The path ramps to scenario price then holds.
    """
    if last_oil is None:
        last_oil = BUDGET_OIL
    target = OIL_PRICES[scenario]

    if scenario == 'S0':
        path = np.full(H, BUDGET_OIL)
    elif scenario == 'S1':
        ramp = np.linspace(last_oil, target, 3)
        tail = np.linspace(target, BUDGET_OIL * 1.05, H - 2)
        path = np.concatenate([ramp[1:], tail])
    elif scenario == 'S2':
        ramp = np.linspace(last_oil, target, 5)
        tail = np.full(H - 4, target * 0.97)
        path = np.concatenate([ramp[1:], tail])
    elif scenario == 'S3':
        ramp = np.linspace(last_oil, target, 4)
        tail = np.linspace(target, target * 1.05, H - 3)
        path = np.concatenate([ramp[1:], tail])

    # Always enforce exact length H
    path = np.asarray(path).flatten()
    if len(path) < H:
        path = np.pad(path, (0, H - len(path)), 'edge')
    return path[:H]

# Build forecast dates
fc_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                         periods=H_FC, freq='MS')

SCENARIOS   = ['S0', 'S1', 'S2', 'S3']
N_FC_DRAWS  = 300   # per scenario (will aggregate over MCMC draws)

print(f"  Forecast horizon: {H_FC} months ({fc_dates[0].strftime('%b %Y')} – "
      f"{fc_dates[-1].strftime('%b %Y')})")

# Run conditional forecasts using a subset of posterior draws
FC_RESULTS = {}
print("\n  Running conditional forecasts for each scenario...")

for sc in SCENARIOS:
    oil_path = build_oil_path(sc, H_FC, last_oil=df['OilPrice_WTI'].iloc[-1])
    all_sc_draws = []

    n_post_sub = min(100, keep_idx)   # use 100 posterior draws
    post_indices = np.random.choice(keep_idx, n_post_sub, replace=False)

    for pidx in post_indices:
        B_d     = B_draws[pidx]
        Sig_d   = Sigma_draws[pidx]
        draws_d = conditional_forecast_cpz(
            B_d, Sig_d, Y_full, P_LAG, H_FC, oil_path, n_draws=30)
        all_sc_draws.append(draws_d)

    all_sc_draws = np.vstack(all_sc_draws)   # (n_post_sub*30 × H × k)

    FC_RESULTS[sc] = {
        'oil_path': oil_path,
        'draws': all_sc_draws,
        'median': np.median(all_sc_draws, axis=0),   # (H × k)
        'lo16':   np.percentile(all_sc_draws, 16, axis=0),
        'hi84':   np.percentile(all_sc_draws, 84, axis=0),
        'lo5':    np.percentile(all_sc_draws, 5, axis=0),
        'hi95':   np.percentile(all_sc_draws, 95, axis=0),
    }
    print(f"    {SCENARIO_LABELS[sc]} — {len(all_sc_draws)} conditional draws")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — ECONOMIC CALIBRATION OF FORECAST PATHS                        ║
# ║  Translate BVAR projections to policy-relevant metrics                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[6] Applying economic calibration to budget anchors...")

"""
The BVAR captures historical correlations. We further calibrate the
conditional median paths to be consistent with Nigeria's 2026 budget
parameters, anchoring the baseline to S0 and computing deviations for S1–S3.

Key calibration relationships (structural):
  - Oil revenue ($bn) = OilPrice × BUDGET_PROD × 365 / 1e9  [simplified]
  - Exchange rate response: ERR coefficient from IRF (oil → FX)
  - Inflation: Pass-through from FX + import price channel
  - External reserves: oil-revenue-driven accumulation equation
  - Liquidity: Banking sector excess reserves, sensitive to MPR/sterilisation
"""

# Variable indices
IDX = {v: i for i, v in enumerate(VAR_NAMES)}

def calibrate_paths(FC_RESULTS, budget_baseline):
    """
    Compute calibrated scenario paths anchored to budget baseline (S0).
    """
    budget = budget_baseline
    cal = {}

    # Extract S0 baseline median
    s0_med = FC_RESULTS['S0']['median']

    for sc in SCENARIOS:
        med = FC_RESULTS[sc]['median']
        lo  = FC_RESULTS[sc]['lo16']
        hi  = FC_RESULTS[sc]['hi84']
        lo5 = FC_RESULTS[sc]['lo5']
        hi95= FC_RESULTS[sc]['hi95']
        oil = FC_RESULTS[sc]['oil_path']

        # ── Oil Revenue ($bn) ─────────────────────────────────────────────────
        # Monthly: price × 1.84mb/d × 30.4 days / 1e3  (in $bn per month)
        oil_rev_monthly = oil * BUDGET_PROD * 30.4 / 1e3   # $bn/month
        oil_rev_annual  = oil * BUDGET_PROD * 365.0 / 1e3  # $bn/year approx

        # ── Exchange Rate ─────────────────────────────────────────────────────
        # BVAR gives the ExchRate trajectory; anchor S0 to budget ₦1400
        er_s0  = s0_med[:, IDX['ExchRate']]
        er_sc  = med[:, IDX['ExchRate']]
        # Delta from S0 baseline
        er_delta = er_sc - er_s0
        er_cal   = budget['ExchRate'] + er_delta
        er_lo    = er_cal + (lo[:, IDX['ExchRate']] - er_sc)
        er_hi    = er_cal + (hi[:, IDX['ExchRate']] - er_sc)

        # ── Inflation ─────────────────────────────────────────────────────────
        inf_s0   = s0_med[:, IDX['Inflation']]
        inf_sc   = med[:, IDX['Inflation']]
        inf_delta = inf_sc - inf_s0
        inf_cal  = budget['Inflation'] + inf_delta
        inf_lo   = inf_cal + (lo[:, IDX['Inflation']] - inf_sc)
        inf_hi   = inf_cal + (hi[:, IDX['Inflation']] - inf_sc)

        # ── External Reserves ($bn) ───────────────────────────────────────────
        res_s0   = s0_med[:, IDX['ExtResBn']]
        res_sc   = med[:, IDX['ExtResBn']]
        res_delta = res_sc - res_s0
        res_cal  = budget['ExtResBn'] + res_delta
        res_lo   = res_cal + (lo[:, IDX['ExtResBn']] - res_sc)
        res_hi   = res_cal + (hi[:, IDX['ExtResBn']] - res_sc)

        # ── Fiscal Revenue (₦bn) — BVAR path, calibrated to budget ───────────
        frev_s0  = s0_med[:, IDX['FiscRevBn']]
        frev_sc  = med[:, IDX['FiscRevBn']]
        frev_delta = frev_sc - frev_s0
        # Budget baseline fiscal revenue (monthly = annual/12)
        # Annual NGN oil revenue ≈ $43.55bn × ₦1400/$
        budget_frev_monthly = (43.55e9 * 1400) / 12 / 1e9   # ₦bn/month
        frev_cal = budget_frev_monthly + frev_delta
        frev_lo  = frev_cal + (lo[:, IDX['FiscRevBn']] - frev_sc)
        frev_hi  = frev_cal + (hi[:, IDX['FiscRevBn']] - frev_sc)

        # For S1–S3: also compute the USD oil revenue uplift
        oil_rev_uplift_usd = (oil - OIL_PRICES['S0']) * BUDGET_PROD * 365 / 1e3

        # ── Liquidity (₦bn) ───────────────────────────────────────────────────
        liq_s0   = s0_med[:, IDX['Liquidity']]
        liq_sc   = med[:, IDX['Liquidity']]
        liq_delta = liq_sc - liq_s0
        liq_cal  = budget['Liquidity'] + liq_delta
        liq_lo   = liq_cal + (lo[:, IDX['Liquidity']] - liq_sc)
        liq_hi   = liq_cal + (hi[:, IDX['Liquidity']] - liq_sc)

        cal[sc] = {
            'oil':        oil,
            'oil_rev_mo': oil_rev_monthly,
            'oil_rev_yr': oil_rev_annual,
            'oil_rev_up': oil_rev_uplift_usd,
            'er':  er_cal,  'er_lo':  er_lo,  'er_hi':  er_hi,
            'inf': inf_cal, 'inf_lo': inf_lo, 'inf_hi': inf_hi,
            'res': res_cal, 'res_lo': res_lo, 'res_hi': res_hi,
            'frev':frev_cal,'frev_lo':frev_lo,'frev_hi':frev_hi,
            'liq': liq_cal, 'liq_lo': liq_lo, 'liq_hi': liq_hi,
        }

    return cal

BUDGET_ANCHOR = {
    'ExchRate':   BUDGET_EXCH,
    'Inflation':  BUDGET_INFL,
    'ExtResBn':   BUDGET_RESRV / 1e9,
    'FiscRevBn':  (BUDGET_OREV * BUDGET_EXCH) / 12 / 1e9,
    'Liquidity':  BUDGET_LIQ,
}

CAL = calibrate_paths(FC_RESULTS, BUDGET_ANCHOR)
print("  Calibration complete.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — FORECAST SUMMARY TABLE                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[7] Generating scenario summary tables...")

# 12-month average for each scenario
rows = []
for sc in SCENARIOS:
    c = CAL[sc]
    h = min(12, H_FC)
    rows.append({
        'Scenario': SCENARIO_LABELS[sc],
        'Oil Price ($/bbl)': f"{c['oil'][:h].mean():.1f}",
        'Exch. Rate (₦/$)':  f"{c['er'][:h].mean():,.0f}",
        'Inflation (%)':     f"{c['inf'][:h].mean():.1f}",
        'Oil Rev. ($bn/yr)': f"{c['oil_rev_yr'][:h].mean():.1f}",
        'Ext. Reserves ($bn)': f"{c['res'][:h].mean():.1f}",
        'Liquidity (₦bn)':   f"{c['liq'][:h].mean():.1f}",
    })

summary_df = pd.DataFrame(rows)
print("\n  12-Month Average Scenario Summary:")
print(summary_df.to_string(index=False))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — FIGURES                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n[8] Generating figures...")

# ── Helper: shade and plot ────────────────────────────────────────────────────
def plot_scenario_fc(ax, dates, c, key, sc, lw=2.0, alpha_shade=0.15):
    col = COLS[sc]
    ax.plot(dates, c[key], color=col, lw=lw, label=SCENARIO_LABELS[sc], zorder=3)
    ax.fill_between(dates, c[f'{key}_lo'], c[f'{key}_hi'],
                    color=col, alpha=alpha_shade, zorder=2)

# ── Figure 1: Scenario overview (6 panels × 4 scenarios) ─────────────────────
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.patch.set_facecolor(COLS['bg'])
fig1.suptitle(
    "Figure 1 — BSVAR Conditional Forecasts: Iran–Israel–US War Oil Shock\n"
    "Transmission through Nigeria's Economy (2026–2027)",
    fontsize=13, fontweight='bold', color=COLS['text'], y=1.01)

PANELS = [
    ('oil',  'Oil Price ($/bbl)',            'Oil Price Path by Scenario'),
    ('er',   'Exchange Rate (₦/$)',          'Exchange Rate Trajectory'),
    ('inf',  'Inflation (%)',                'Inflation Path'),
    ('frev', 'Fiscal Revenue (₦ bn/month)', 'Oil-linked Fiscal Revenue'),
    ('res',  'External Reserves ($bn)',      'External Reserves'),
    ('liq',  'Banking Liquidity (₦ bn)',    'Liquidity Management'),
]

ax_flat = axes.flatten()

for ax, (key, ylabel, title) in zip(ax_flat, PANELS):
    ax.set_facecolor(COLS['bg'])
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.6)
    ax.set_title(title, fontsize=10, fontweight='bold', color=COLS['text'], pad=6)
    ax.set_ylabel(ylabel, fontsize=8, color=COLS['text'])

    # Historical data where applicable
    if key == 'oil':
        hist_key = 'OilPrice_WTI'
        hist_data = df[hist_key].iloc[-24:]
    elif key == 'er':
        hist_data = df['ExchRate'].iloc[-24:]
    elif key == 'inf':
        hist_data = df['Inflation'].iloc[-24:]
    elif key == 'frev':
        hist_data = df['FiscRevBn'].iloc[-24:]
    elif key == 'res':
        hist_data = df['ExtResBn'].iloc[-24:]
    elif key == 'liq':
        hist_data = df['Liquidity'].iloc[-24:]

    if hist_data is not None:
        ax.plot(hist_data.index, hist_data.values,
                color=COLS['hist'], lw=1.5, ls='--', alpha=0.7,
                label='Historical', zorder=4)

    for sc in SCENARIOS:
        c = CAL[sc]
        col = COLS[sc]
        if key == 'oil':
            ax.plot(fc_dates, c['oil'], color=col, lw=2.0,
                    label=SCENARIO_LABELS[sc], zorder=3)
        else:
            plot_scenario_fc(ax, fc_dates, c, key, sc)

    # Budget benchmark line for relevant panels
    if key == 'oil':
        ax.axhline(BUDGET_OIL, color='gray', ls=':', lw=1.2, alpha=0.6, label=f'Budget: ${BUDGET_OIL}')
    elif key == 'er':
        ax.axhline(BUDGET_EXCH, color='gray', ls=':', lw=1.2, alpha=0.6, label=f'Budget: ₦{BUDGET_EXCH:,}')
    elif key == 'inf':
        ax.axhline(BUDGET_INFL, color='gray', ls=':', lw=1.2, alpha=0.6, label=f'Budget: {BUDGET_INFL}%')
    elif key == 'res':
        ax.axhline(BUDGET_RESRV / 1e9, color='gray', ls=':', lw=1.2, alpha=0.6, label=f'Budget: ${BUDGET_RESRV/1e9:.1f}bn')

    ax.axvline(fc_dates[0], color='black', ls=':', lw=0.8, alpha=0.4)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    if key == 'oil':
        ax.legend(fontsize=6, loc='upper left', framealpha=0.7)
    elif key == 'er':
        ax.legend(fontsize=6, loc='upper left', framealpha=0.7)

plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('/home/claude/fig1_scenario_overview.png', dpi=150, bbox_inches='tight',
            facecolor=COLS['bg'])
plt.close()
print("  Fig 1 saved.")

# ── Figure 2: Impulse Response Functions ─────────────────────────────────────
# Show responses of all 6 variables to Oil Price shock (structural shock 1)
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.patch.set_facecolor(COLS['bg'])
fig2.suptitle(
    "Figure 2 — Structural Impulse Response Functions\n"
    "Response to One Standard Deviation Oil Price Shock (68% & 90% posterior bands)",
    fontsize=12, fontweight='bold', color=COLS['text'], y=1.01)

shock_idx = IDX['OilPrice_WTI']
horizon_x = np.arange(H_IRF + 1)

for ax, (vi, var_name) in zip(axes2.flatten(), enumerate(VAR_NAMES)):
    ax.set_facecolor(COLS['bg'])
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.6)
    ax.set_title(f'Response of {VAR_LABELS[vi]}', fontsize=9, fontweight='bold',
                 color=COLS['text'], pad=4)
    ax.set_xlabel('Months', fontsize=8)
    ax.axhline(0, color='black', lw=0.8, alpha=0.5)

    irf_med = np.median(IRF_draws[:, :, vi, shock_idx], axis=0)
    irf_lo1 = np.percentile(IRF_draws[:, :, vi, shock_idx], 16, axis=0)
    irf_hi1 = np.percentile(IRF_draws[:, :, vi, shock_idx], 84, axis=0)
    irf_lo2 = np.percentile(IRF_draws[:, :, vi, shock_idx], 5,  axis=0)
    irf_hi2 = np.percentile(IRF_draws[:, :, vi, shock_idx], 95, axis=0)

    ax.fill_between(horizon_x, irf_lo2, irf_hi2, color=COLS['S3'], alpha=0.12, label='90% band')
    ax.fill_between(horizon_x, irf_lo1, irf_hi1, color=COLS['S3'], alpha=0.25, label='68% band')
    ax.plot(horizon_x, irf_med, color=COLS['S3'], lw=2.0, label='Posterior median')
    ax.legend(fontsize=6, framealpha=0.7)
    ax.tick_params(labelsize=7)

plt.tight_layout(h_pad=3, w_pad=2)
plt.savefig('/home/claude/fig2_irf.png', dpi=150, bbox_inches='tight',
            facecolor=COLS['bg'])
plt.close()
print("  Fig 2 saved.")

# ── Figure 3: Fan chart — each variable across 4 scenarios ───────────────────
fig3, axes3 = plt.subplots(3, 2, figsize=(15, 18))
fig3.patch.set_facecolor(COLS['bg'])
fig3.suptitle(
    "Figure 3 — Counterfactual Forecast Fan Charts\n"
    "68% Posterior Credible Intervals | Nigeria 2026–2027",
    fontsize=13, fontweight='bold', color=COLS['text'], y=1.01)

FAN_PANELS = [
    ('er',   'Exchange Rate (₦/$)',          'Exchange Rate: Scenario Paths',          BUDGET_EXCH,        '₦{:.0f}/$'),
    ('inf',  'Inflation (%)',                'Inflation: Scenario Trajectories',       BUDGET_INFL,        '{:.0f}%'),
    ('oil',  'Oil Price ($/bbl)',            'Oil Price Assumptions',                  BUDGET_OIL,         '${:.2f}/bbl'),
    ('frev', 'Monthly Fiscal Revenue (₦bn)', 'Fiscal Revenue: Scenario Paths',         None,               None),
    ('res',  'External Reserves ($bn)',      'External Reserves: Scenario Trajectories', BUDGET_RESRV/1e9,  '${:.1f}bn'),
    ('liq',  'Banking Liquidity (₦bn)',     'System Liquidity: Scenario Paths',       None,               None),
]

for ax, (key, ylabel, title, benchmark, bfmt) in zip(axes3.flatten(), FAN_PANELS):
    ax.set_facecolor(COLS['bg'])
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.6)
    ax.set_title(title, fontsize=10, fontweight='bold', color=COLS['text'], pad=6)
    ax.set_ylabel(ylabel, fontsize=8)

    for sc in SCENARIOS:
        c   = CAL[sc]
        col = COLS[sc]
        if key == 'oil':
            ax.plot(fc_dates, c['oil'], color=col, lw=2.2, label=SCENARIO_LABELS[sc])
        else:
            plot_scenario_fc(ax, fc_dates, c, key, sc, lw=2.2, alpha_shade=0.18)

    if benchmark is not None:
        lbl = f'Budget baseline: {bfmt.format(benchmark)}'
        ax.axhline(benchmark, color='dimgray', ls=':', lw=1.5, alpha=0.7, label=lbl)

    ax.axvline(fc_dates[0], color='black', ls=':', lw=0.9, alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.legend(fontsize=7, loc='best', framealpha=0.8)

plt.tight_layout(h_pad=4, w_pad=3)
plt.savefig('/home/claude/fig3_fan_charts.png', dpi=150, bbox_inches='tight',
            facecolor=COLS['bg'])
plt.close()
print("  Fig 3 saved.")

# ── Figure 4: Scenario deviation from baseline ────────────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 9))
fig4.patch.set_facecolor(COLS['bg'])
fig4.suptitle(
    "Figure 4 — Counterfactual Deviations from Baseline (S0)\n"
    "Percentage-point deviations | 68% credible intervals",
    fontsize=12, fontweight='bold', color=COLS['text'], y=1.01)

DEV_PANELS = [
    ('er',   'Exchange Rate Deviation (₦/$)'),
    ('inf',  'Inflation Deviation (pp)'),
    ('res',  'Reserves Deviation ($bn)'),
    ('frev', 'Fiscal Revenue Deviation (₦bn/mo)'),
]

for ax, (key, ylabel) in zip(axes4.flatten(), DEV_PANELS):
    ax.set_facecolor(COLS['bg'])
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.6)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6, label='Baseline (S0)')

    base = CAL['S0'][key]
    for sc in ['S1', 'S2', 'S3']:
        c   = CAL[sc]
        col = COLS[sc]
        dev     = c[key] - base
        dev_lo  = c[f'{key}_lo'] - base
        dev_hi  = c[f'{key}_hi'] - base
        ax.plot(fc_dates, dev, color=col, lw=2.0, label=SCENARIO_LABELS[sc])
        ax.fill_between(fc_dates, dev_lo, dev_hi, color=col, alpha=0.18)

    ax.axvline(fc_dates[0], color='gray', ls=':', lw=0.8, alpha=0.4)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.legend(fontsize=7, framealpha=0.8)

plt.tight_layout(h_pad=3, w_pad=3)
plt.savefig('/home/claude/fig4_deviations.png', dpi=150, bbox_inches='tight',
            facecolor=COLS['bg'])
plt.close()
print("  Fig 4 saved.")

# ── Figure 5: Policy summary dashboard ───────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(13, 6))
fig5.patch.set_facecolor(COLS['bg'])
ax5.axis('off')

metrics = ['Oil Price\n($/bbl)', 'Exch. Rate\n(₦/$)', 'Inflation\n(%)',
           'Oil Rev.\n($bn/yr)', 'Ext. Res.\n($bn)', 'Liquidity\n(₦bn)']

h12 = min(12, H_FC)
table_data = []
for sc in SCENARIOS:
    c = CAL[sc]
    row = [
        f"{c['oil'][:h12].mean():.1f}",
        f"{c['er'][:h12].mean():,.0f}",
        f"{c['inf'][:h12].mean():.1f}",
        f"{c['oil_rev_yr'][:h12].mean():.1f}",
        f"{c['res'][:h12].mean():.1f}",
        f"{c['liq'][:h12].mean():.1f}",
    ]
    table_data.append(row)

col_labels = metrics
row_labels = [SCENARIO_LABELS[s] for s in SCENARIOS]
row_colors = [[COLS[s]] * len(metrics) for s in SCENARIOS]

tbl = ax5.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(COLS['grid'])
    if row == 0:
        cell.set_facecolor(COLS['text'])
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
    elif col == -1:
        sc_key = SCENARIOS[row - 1]
        cell.set_facecolor(COLS[sc_key])
        cell.set_text_props(color='white', fontsize=8, fontweight='bold')
        cell.set_width(0.35)
    else:
        cell.set_facecolor(COLS['bg'])

fig5.suptitle(
    "Figure 5 — Policy Scenario Summary Dashboard\n"
    "12-Month Average Projections | BSVAR Conditional Forecasts",
    fontsize=12, fontweight='bold', color=COLS['text'], y=1.01)
plt.savefig('/home/claude/fig5_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor=COLS['bg'])
plt.close()
print("  Fig 5 saved.")

# ── Figure 6: Liquidity — special focus ──────────────────────────────────────
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 6))
fig6.patch.set_facecolor(COLS['bg'])
fig6.suptitle("Figure 6 — Liquidity Management Under War Scenarios",
              fontsize=12, fontweight='bold', color=COLS['text'])

# Left: level paths
for ax in [ax6a, ax6b]:
    ax.set_facecolor(COLS['bg'])
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.6)

# Historical
liq_hist = df['Liquidity'].iloc[-36:]
ax6a.plot(liq_hist.index, liq_hist, color=COLS['hist'], lw=1.5, ls='--',
          alpha=0.7, label='Historical')

for sc in SCENARIOS:
    c = CAL[sc]
    ax6a.plot(fc_dates, c['liq'], color=COLS[sc], lw=2.0, label=SCENARIO_LABELS[sc])
    ax6a.fill_between(fc_dates, c['liq_lo'], c['liq_hi'], color=COLS[sc], alpha=0.15)

ax6a.axvline(fc_dates[0], color='black', ls=':', lw=0.9, alpha=0.3)
ax6a.set_title('Banking Sector Liquidity (₦bn)', fontsize=10, fontweight='bold')
ax6a.set_ylabel('₦bn', fontsize=9)
ax6a.legend(fontsize=7, framealpha=0.8)
ax6a.tick_params(labelsize=7)
ax6a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
ax6a.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))

# Right: deviation from baseline
base_liq = CAL['S0']['liq']
for sc in ['S1', 'S2', 'S3']:
    c   = CAL[sc]
    dev = c['liq'] - base_liq
    ax6b.plot(fc_dates, dev, color=COLS[sc], lw=2.0, label=SCENARIO_LABELS[sc])
    ax6b.fill_between(fc_dates,
                       c['liq_lo'] - base_liq,
                       c['liq_hi'] - base_liq,
                       color=COLS[sc], alpha=0.18)

ax6b.axhline(0, color='black', lw=1.0, ls='--', alpha=0.6, label='Baseline (S0)')
ax6b.set_title('Liquidity Deviation from Baseline (₦bn)', fontsize=10, fontweight='bold')
ax6b.set_ylabel('Δ₦bn from S0', fontsize=9)
ax6b.legend(fontsize=7, framealpha=0.8)
ax6b.tick_params(labelsize=7)
ax6b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b\n%Y'))
ax6b.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))

plt.tight_layout()
plt.savefig('/home/claude/fig6_liquidity.png', dpi=150, bbox_inches='tight',
            facecolor=COLS['bg'])
plt.close()
print("  Fig 6 saved.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — PRINT FULL RESULTS TABLES                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
print("\n" + "=" * 72)
print(" SECTION 9 — FULL SCENARIO RESULTS TABLES")
print("=" * 72)

for sc in SCENARIOS:
    c = CAL[sc]
    print(f"\n{'─'*70}")
    print(f"  {SCENARIO_LABELS[sc]}")
    print(f"{'─'*70}")
    print(f"  {'Month':<12} {'Oil$':>7} {'ExchRate':>10} {'Infl%':>7} "
          f"{'FiscRevNbn':>12} {'ExtRes$bn':>11} {'LiqNbn':>10}")
    print(f"  {'':─<12} {'':─>7} {'':─>10} {'':─>7} {'':─>12} {'':─>11} {'':─>10}")
    for h in range(min(18, H_FC)):
        print(f"  {fc_dates[h].strftime('%b %Y'):<12} "
              f"{c['oil'][h]:>7.2f} "
              f"{c['er'][h]:>10,.0f} "
              f"{c['inf'][h]:>7.2f} "
              f"{c['frev'][h]:>12.1f} "
              f"{c['res'][h]:>11.2f} "
              f"{c['liq'][h]:>10.1f}")

print("\n" + "=" * 72)
print(" CROSS-SCENARIO DEVIATION TABLE (from S0 Baseline, 12-month avg)")
print("=" * 72)
print(f"\n  {'Variable':<30} {'S0 (Base)':>12} {'S1 Mild':>12} "
      f"{'S2 Moderate':>12} {'S3 Severe':>12}")
print("  " + "─" * 70)

h12 = min(12, H_FC)
vars_display = [
    ('oil',   'Oil Price ($/bbl)',          '{:.1f}'),
    ('er',    'Exchange Rate (₦/$)',         '{:,.0f}'),
    ('inf',   'Inflation (%)',               '{:.2f}'),
    ('frev',  'Fiscal Revenue (₦bn/mo)',    '{:.1f}'),
    ('res',   'External Reserves ($bn)',    '{:.2f}'),
    ('liq',   'Liquidity (₦bn)',            '{:.1f}'),
]
for key, label, fmt in vars_display:
    vals = {sc: CAL[sc][key][:h12].mean() for sc in SCENARIOS}
    row  = f"  {label:<30}"
    for sc in SCENARIOS:
        row += f" {fmt.format(vals[sc]):>12}"
    print(row)

# Also print deviations
print("\n  Deviations from S0:")
print("  " + "─" * 70)
for key, label, fmt in vars_display:
    base_val = CAL['S0'][key][:h12].mean()
    row = f"  Δ {label:<28}"
    for sc in SCENARIOS:
        dev = CAL[sc][key][:h12].mean() - base_val
        row += f" {fmt.format(dev):>12}"
    print(row)

print("\n" + "=" * 72)
print(" MCMC DIAGNOSTICS")
print("=" * 72)
print(f"  Posterior draws kept:  {keep_idx}")
print(f"  Burn-in:               {MCMC_BURN}")
print(f"  Thinning:              every {THIN}")
print(f"  VAR lag order (p):     {P_LAG}")
print(f"  Minnesota λ₁:         {lambda1}")
print(f"  Minnesota λ₂:         {lambda2}")
print(f"  Posterior nu:          {nu_post}")
print(f"  Mean posterior Sigma diagonal:")
for i, vn in enumerate(VAR_NAMES):
    print(f"    {vn:<20}: {np.diag(Sigma_mean)[i]:.4f}")

print("\n" + "=" * 72)
print(" OUTPUT FILES")
print("=" * 72)
print("  fig1_scenario_overview.png — 6-panel scenario trajectories")
print("  fig2_irf.png               — Structural IRFs (oil shock)")
print("  fig3_fan_charts.png        — Fan chart credible intervals")
print("  fig4_deviations.png        — Counterfactual deviations from baseline")
print("  fig5_dashboard.png         — Policy summary dashboard table")
print("  fig6_liquidity.png         — Liquidity management focus")

print("\n  Analysis complete.")
