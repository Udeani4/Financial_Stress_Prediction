"""
Shared model code for the Liquidity Stress app.

Used by BOTH claude_solution.ipynb (export section) and streamlit_app.py, so the
features the app computes are exactly the ones the models were trained on, and
the pickled StressBlend object can be loaded by the app.

The feature engineering block below is copied verbatim from section 3a of
claude_solution.ipynb. If you change features in the notebook, change them here
too and re-export the model.
"""
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------
# Raw input schema
# --------------------------------------------------------------------------
STATIC_COLS = ["arpu", "age", "gender", "region", "smartphone", "segment",
               "earning_pattern", "x_90_d_activity_rate"]
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
NUM_STATIC = [c for c in STATIC_COLS if c not in CAT_COLS]

# --------------------------------------------------------------------------
# Feature engineering (verbatim from claude_solution.ipynb, section 3a)
# --------------------------------------------------------------------------
CATS = {
    "paybill": "companies", "merchantpay": "merchants", "transfer_from_bank": "banks",
    "mm_send": "recipients", "received": "senders", "deposit": "agents", "withdraw": "agents",
}
INFLOW = ["transfer_from_bank", "received", "deposit"]
OUTFLOW = ["paybill", "merchantpay", "mm_send", "withdraw"]

EPS = 1e-6
T = np.array([6, 5, 4, 3, 2, 1], dtype=float)   # M1 (recent) = 6 ... M6 (old) = 1
TC = T - T.mean()
DENOM = TC @ TC


def _arr(df, cat, metric):
    cols = [f"m{i}_{cat}_{metric}" for i in range(1, 7)]
    return df[cols].fillna(0).to_numpy(dtype=float)


def _slope(x):
    return (x - x.mean(axis=1, keepdims=True)) @ TC / DENOM


def _recent_older(x):
    return (x[:, :3].mean(axis=1) + EPS) / (x[:, 3:].mean(axis=1) + EPS)


def add_features(df):
    new = {}

    # 1. Per-category trend features
    for cat, cp in CATS.items():
        for metric in ["volume", "total_value", "highest_amount", cp]:
            x = _arr(df, cat, metric)
            f = f"{cat}_{metric}"
            mean = x.mean(axis=1)
            new[f"{f}_mean"] = mean
            new[f"{f}_std"] = x.std(axis=1, ddof=1)
            new[f"{f}_ratio"] = _recent_older(x)
            if metric in ("volume", "total_value"):
                s = _slope(x)
                new[f"{f}_slope"] = s
                new[f"{f}_rel_slope"] = s / (mean + EPS)

    # 2. Balance features
    b = df[[f"m{i}_daily_avg_bal" for i in range(1, 7)]]
    bal = b.to_numpy(dtype=float)
    bal_mean = b.mean(axis=1).to_numpy()
    new["bal_mean"] = bal_mean
    new["bal_std"] = b.std(axis=1).to_numpy()
    new["bal_ratio"] = ((b.iloc[:, :3].mean(axis=1) + EPS) / (b.iloc[:, 3:].mean(axis=1) + EPS)).to_numpy()
    bs = _slope(bal)
    new["bal_slope"] = bs
    new["bal_rel_slope"] = bs / (bal_mean + EPS)

    # 3. Cash-flow features
    val = {c: _arr(df, c, "total_value") for c in CATS}
    inflow = sum(val[c] for c in INFLOW)
    outflow = sum(val[c] for c in OUTFLOW)
    net = inflow - outflow

    for name, x in [("inflow", inflow), ("outflow", outflow)]:
        new[f"{name}_mean"] = x.mean(axis=1)
        new[f"{name}_std"] = x.std(axis=1, ddof=1)
        new[f"{name}_ratio"] = _recent_older(x)
        new[f"{name}_slope"] = _slope(x)

    new["net_mean"] = net.mean(axis=1)
    new["net_std"] = net.std(axis=1, ddof=1)
    new["net_slope"] = _slope(net)
    new["net_recent_minus_older"] = net[:, :3].mean(axis=1) - net[:, 3:].mean(axis=1)
    new["neg_net_months"] = (net < 0).sum(axis=1)
    new["neg_net_months_last3"] = (net[:, :3] < 0).sum(axis=1)

    new["out_in_ratio_6m"] = outflow.sum(axis=1) / (inflow.sum(axis=1) + EPS)
    new["out_in_ratio_last3"] = outflow[:, :3].sum(axis=1) / (inflow[:, :3].sum(axis=1) + EPS)
    new["inflow_cv"] = inflow.std(axis=1, ddof=1) / (inflow.mean(axis=1) + EPS)
    new["inflow_months"] = (inflow > 0).sum(axis=1)

    runway = bal / (outflow + EPS)
    new["runway_m1"] = runway[:, 0]
    new["runway_mean"] = np.nanmean(runway, axis=1)
    new["bal_vs_mean_outflow"] = bal_mean / (outflow.mean(axis=1) + EPS)
    max_withdrawal = _arr(df, "withdraw", "highest_amount").max(axis=1)
    new["bal_vs_max_withdrawal"] = bal[:, 0] / (max_withdrawal + EPS)

    new["cash_out_share"] = val["withdraw"].sum(axis=1) / (inflow.sum(axis=1) + EPS)
    new["deposit_to_withdraw"] = val["deposit"].sum(axis=1) / (val["withdraw"].sum(axis=1) + EPS)
    new["paybill_share_outflow"] = val["paybill"].sum(axis=1) / (outflow.sum(axis=1) + EPS)

    # 4-5. Activity, size and concentration
    for cat, cp in CATS.items():
        vol = _arr(df, cat, "volume")
        active = vol > 0
        new[f"{cat}_active_months"] = active.sum(axis=1)
        new[f"{cat}_active_last3"] = active[:, :3].sum(axis=1)
        new[f"{cat}_zero_streak"] = np.cumprod((~active).astype(int), axis=1).sum(axis=1)
        v = val[cat]
        hi = _arr(df, cat, "highest_amount")
        cpv = _arr(df, cat, cp)
        new[f"{cat}_avg_ticket"] = v.sum(axis=1) / (vol.sum(axis=1) + EPS)
        new[f"{cat}_spike"] = hi.max(axis=1) / (v.mean(axis=1) + EPS)
        new[f"{cat}_txn_per_counterparty"] = vol.mean(axis=1) / (cpv.mean(axis=1) + EPS)

    # 6. Profile interactions
    arpu = df["arpu"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)
    new["arpu_to_bal"] = arpu / (np.abs(bal_mean) + EPS)
    new["bal_per_age"] = bal_mean / (age + 1)
    new["arpu_to_outflow"] = arpu / (outflow.mean(axis=1) + EPS)

    # Batch A: balance dynamics
    b1, b2 = bal[:, 0], bal[:, 1]
    scale = np.abs(np.nanmean(bal, axis=1)) + EPS
    peak = np.nanmax(bal, axis=1)
    new["bal_m1_minus_m2_rel"] = (b1 - b2) / scale
    new["bal_m1_vs_rest"] = b1 / (np.abs(np.nanmean(bal[:, 1:], axis=1)) + EPS)
    new["bal_last2_vs_older"] = (np.nanmean(bal[:, :2], axis=1) + EPS) / (np.nanmean(bal[:, 2:], axis=1) + EPS)
    new["bal_drop_from_peak"] = (peak - b1) / (np.abs(peak) + EPS)
    new["bal_above_min"] = (b1 - np.nanmin(bal, axis=1)) / scale
    dec = bal[:, :-1] < bal[:, 1:]
    new["bal_decline_months"] = dec.sum(axis=1)
    new["bal_decline_last3"] = dec[:, :3].sum(axis=1)

    # Batch B: shorter windows on cash flow
    new["out_in_ratio_last1"] = outflow[:, 0] / (inflow[:, 0] + EPS)
    new["out_in_ratio_last2"] = outflow[:, :2].sum(axis=1) / (inflow[:, :2].sum(axis=1) + EPS)
    new["out_in_max_month"] = (outflow / (inflow + EPS)).max(axis=1)
    new["inflow_last2_vs_older"] = (inflow[:, :2].mean(axis=1) + EPS) / (inflow[:, 2:].mean(axis=1) + EPS)
    new["inflow_m1_vs_rest"] = (inflow[:, 0] + EPS) / (inflow[:, 1:].mean(axis=1) + EPS)
    new["outflow_last2_vs_older"] = (outflow[:, :2].mean(axis=1) + EPS) / (outflow[:, 2:].mean(axis=1) + EPS)
    w = val["withdraw"]
    new["withdraw_last2_vs_older"] = (w[:, :2].mean(axis=1) + EPS) / (w[:, 2:].mean(axis=1) + EPS)
    cash_share = w / (outflow + EPS)
    new["cash_share_last3_vs_older"] = (cash_share[:, :3].mean(axis=1) + EPS) / (cash_share[:, 3:].mean(axis=1) + EPS)
    new["runway_last3_vs_older"] = (np.nanmean(runway[:, :3], axis=1) + EPS) / (np.nanmean(runway[:, 3:], axis=1) + EPS)

    # ---- NEW (Claude): per-month net flow and cover ratios, kept as raw columns
    for i in range(3):
        new[f"net_m{i+1}"] = net[:, i]
        new[f"net_m{i+1}_rel"] = net[:, i] / (inflow.mean(axis=1) + EPS)
        new[f"bal_m{i+1}_cover_out"] = bal[:, i] / (outflow[:, i] + EPS)
    new["bal_m1_z_own"] = (b1 - np.nanmean(bal[:, 1:], axis=1)) / (np.nanstd(bal[:, 1:], axis=1) + EPS)
    new["bal_min_m1_m3"] = np.nanmin(bal[:, :3], axis=1)
    new["bal_m1_log"] = np.sign(b1) * np.log1p(np.abs(b1))

    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# --------------------------------------------------------------------------
# Raw column list (same order as Train.csv / Test.csv)
# --------------------------------------------------------------------------
def metrics_for(cat):
    return ["volume", "total_value", "highest_amount", CATS[cat]]


TXN_COLS = [f"m{i}_{cat}_{metric}"
            for cat in CATS for i in range(1, 7) for metric in metrics_for(cat)]
BAL_COLS = [f"m{i}_daily_avg_bal" for i in range(1, 7)]
RAW_FEATURES = STATIC_COLS + TXN_COLS + BAL_COLS        # 182 input columns


# --------------------------------------------------------------------------
# Group-relative features. In the notebook the medians are computed on
# Train+Test together; here they are fitted once at export time and stored,
# so scoring a single customer gives the same values.
# --------------------------------------------------------------------------
GROUP_COLS = ["region", "segment", "earning_pattern"]
GROUP_FEATS = ["bal_mean", "outflow_mean", "inflow_mean", "bal_rel_slope"]


def fit_group_stats(df):
    """df must already contain the add_features() columns."""
    return {g: {f: df.groupby(g, observed=True)[f].median().astype(float).to_dict()
                for f in GROUP_FEATS}
            for g in GROUP_COLS}


def add_group_features(df, stats):
    new = {}
    for g in GROUP_COLS:
        key = df[g].astype(str)
        for f in GROUP_FEATS:
            med = key.map(stats[g][f]).astype(float)
            new[f"{f}_vs_{g}"] = df[f] / (med.abs() + EPS)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def validate_raw(df, cat_levels):
    """Return (clean_df, errors, warnings). errors -> cannot score."""
    errors, warns = [], []
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        errors.append(f"{len(missing)} required column(s) are missing: {preview}")
        return df, errors, warns

    out = df.copy()
    for c in RAW_FEATURES:
        if c in CAT_COLS:
            out[c] = out[c].astype(str).str.strip()
            bad = sorted(set(out[c]) - set(cat_levels[c]))
            if bad:
                errors.append(f"Column '{c}' has unknown value(s) {bad[:5]}. "
                              f"Allowed: {cat_levels[c]}")
        else:
            before = out[c].isna().sum()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            n_bad = int(out[c].isna().sum() - before)
            if n_bad:
                warns.append(f"Column '{c}': {n_bad} non-numeric value(s) treated as missing.")

    n_missing = int(out[[c for c in RAW_FEATURES if c not in CAT_COLS]].isna().sum().sum())
    if n_missing:
        warns.append(f"{n_missing} empty numeric cell(s). Transactions are treated as 0; "
                     "missing balances are left empty.")
    neg = (out[TXN_COLS] < 0).sum().sum()
    if neg:
        warns.append(f"{int(neg)} negative transaction value(s) found; these are unusual.")
    return out, errors, warns


# --------------------------------------------------------------------------
# The deployable model
# --------------------------------------------------------------------------
def _clip(p):
    return np.clip(p, 1e-6, 1 - 1e-6)


def _logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class StressBlend:
    """Fold models of each family -> mean probability per family -> weighted
    logit blend -> Platt calibration. Mirrors sections 6-8 of the notebook."""

    def __init__(self, fitted, names, weights, cal_coef, cal_intercept,
                 features, cat_levels, group_stats, metadata):
        self.fitted = fitted                  # {"lgb": [models], "cat": [...], ...}
        self.names = list(names)
        self.weights = np.asarray(weights, dtype=float)
        self.cal_coef = cal_coef              # None -> no calibration
        self.cal_intercept = cal_intercept
        self.features = list(features)
        self.cat_levels = cat_levels
        self.group_stats = group_stats
        self.metadata = metadata

    # ---- features ---------------------------------------------------------
    def prepare(self, raw):
        df = raw[RAW_FEATURES].copy().reset_index(drop=True)
        df = add_features(df)
        df = add_group_features(df, self.group_stats)
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in CAT_COLS:
            df[c] = pd.Categorical(df[c].astype(str), categories=self.cat_levels[c])
        return df[self.features]

    def _cat_frame(self, X):
        Xc = X.copy()
        for c in CAT_COLS:
            Xc[c] = Xc[c].astype(str)
        return Xc

    # ---- prediction -------------------------------------------------------
    def family_probs(self, X):
        out = {}
        for n in self.names:
            Xn = self._cat_frame(X) if n == "cat" else X
            out[n] = np.mean([m.predict_proba(Xn)[:, 1] for m in self.fitted[n]], axis=0)
        return out

    def _finish(self, fam):
        z = np.column_stack([_logit(fam[n]) for n in self.names]) @ self.weights
        if self.cal_coef is not None:
            z = self.cal_coef * z + self.cal_intercept
        return _sigmoid(z)

    def predict_from_features(self, X):
        return self._finish(self.family_probs(X))

    def predict(self, raw):
        """raw: DataFrame with the 182 raw columns -> stress probabilities."""
        return self.predict_from_features(self.prepare(raw))

    def predict_proba(self, raw):
        p = self.predict(raw)
        return np.column_stack([1 - p, p])

    # ---- explanation ------------------------------------------------------
    def explain(self, X_row):
        """Approximate SHAP contributions (log-odds) for ONE prepared row.
        Each family's fold models are averaged, then families are combined
        with the blend weights (and the calibration slope)."""
        contrib = np.zeros(len(self.features))
        base = 0.0
        for n, w in zip(self.names, self.weights):
            vals = []
            for m in self.fitted[n]:
                if n == "cat":
                    from catboost import Pool
                    Xc = self._cat_frame(X_row)
                    v = m.get_feature_importance(Pool(Xc, cat_features=CAT_COLS),
                                                 type="ShapValues")[0]
                elif n == "lgb":
                    v = m.predict(X_row, pred_contrib=True)[0]
                elif n == "xgb":
                    import xgboost as xgb
                    dm = xgb.DMatrix(X_row, enable_categorical=True)
                    v = m.get_booster().predict(dm, pred_contribs=True)[0]
                else:
                    continue
                vals.append(np.asarray(v, dtype=float))
            v = np.mean(vals, axis=0)
            contrib += w * v[:-1]
            base += w * v[-1]
        scale = self.cal_coef if self.cal_coef is not None else 1.0
        return pd.Series(contrib * scale, index=self.features), base * scale


# --------------------------------------------------------------------------
# Risk bands (base rate in training data = 15%)
# --------------------------------------------------------------------------
RISK_BANDS = [   # (upper bound, name, colour, short meaning)
    (0.10, "Low", "#2e9e6b", "Clearly below the average customer's risk."),
    (0.30, "Moderate", "#d9a521", "Around or somewhat above the average risk; worth watching."),
    (0.60, "High", "#e0703a", "Several warning signs; consider proactive support."),
    (1.01, "Very high", "#d64545", "Strong signs of an upcoming cash shortfall."),
]


def risk_band(p):
    for upper, name, colour, meaning in RISK_BANDS:
        if p < upper:
            return name, colour, meaning
    return RISK_BANDS[-1][1:]


# --------------------------------------------------------------------------
# Human-readable feature names
# --------------------------------------------------------------------------
CAT_LABELS = {
    "paybill": "Bill payments (Paybill)", "merchantpay": "Merchant payments",
    "transfer_from_bank": "Bank-to-wallet transfers", "mm_send": "Money sent to others",
    "received": "Money received", "deposit": "Cash deposits", "withdraw": "Cash withdrawals",
}
METRIC_LABELS = {
    "volume": "number of transactions", "total_value": "total value",
    "highest_amount": "largest single transaction", "companies": "distinct companies paid",
    "merchants": "distinct merchants", "banks": "distinct banks",
    "recipients": "distinct recipients", "senders": "distinct senders",
    "agents": "distinct agents",
}
STAT_LABELS = {
    "mean": "6-month average", "std": "month-to-month variation",
    "ratio": "last 3 months ÷ previous 3", "slope": "6-month trend",
    "rel_slope": "6-month trend relative to its average",
}
CAT_EXTRA = {
    "active_months": "months with any activity (of 6)",
    "active_last3": "active months among the last 3",
    "zero_streak": "inactive months in a row up to the latest",
    "avg_ticket": "average transaction size",
    "spike": "largest transaction ÷ average monthly value",
    "txn_per_counterparty": "transactions per counterparty",
}
STATIC_LABELS = {
    "arpu": "Average revenue per user (ARPU)", "age": "Age", "gender": "Gender",
    "region": "Region", "smartphone": "Has a smartphone", "segment": "Customer segment",
    "earning_pattern": "Earning pattern",
    "x_90_d_activity_rate": "Share of days active in the last 90 days",
}
MONEY_IN = "Money in (bank transfers + received + deposits)"
MONEY_OUT = "Money out (bills + merchants + sent + withdrawals)"
EXPLICIT_LABELS = {
    "bal_mean": "Average daily balance (6-month average)",
    "bal_std": "Balance month-to-month variation",
    "bal_ratio": "Balance: last 3 months ÷ previous 3",
    "bal_slope": "Balance trend per month",
    "bal_rel_slope": "Balance trend relative to average balance",
    "net_mean": "Net cash flow (in − out), 6-month average",
    "net_std": "Net cash flow month-to-month variation",
    "net_slope": "Net cash flow trend",
    "net_recent_minus_older": "Net cash flow: last 3 months minus previous 3",
    "neg_net_months": "Months where money out exceeded money in (of 6)",
    "neg_net_months_last3": "Months where money out exceeded money in (of last 3)",
    "out_in_ratio_6m": "Money out ÷ money in (6 months)",
    "out_in_ratio_last3": "Money out ÷ money in (last 3 months)",
    "out_in_ratio_last2": "Money out ÷ money in (last 2 months)",
    "out_in_ratio_last1": "Money out ÷ money in (latest month)",
    "out_in_max_month": "Worst month: money out ÷ money in",
    "inflow_cv": "Income irregularity (variation ÷ average money in)",
    "inflow_months": "Months with any money in (of 6)",
    "runway_m1": "Balance cover: latest balance ÷ latest month's spending",
    "runway_mean": "Balance cover, 6-month average",
    "runway_last3_vs_older": "Balance cover: last 3 months ÷ previous 3",
    "bal_vs_mean_outflow": "Average balance ÷ average monthly spending",
    "bal_vs_max_withdrawal": "Latest balance ÷ largest cash withdrawal",
    "cash_out_share": "Cash withdrawals ÷ money in (6 months)",
    "deposit_to_withdraw": "Cash deposits ÷ cash withdrawals",
    "paybill_share_outflow": "Bill payments as share of money out",
    "arpu_to_bal": "ARPU ÷ average balance",
    "bal_per_age": "Average balance per year of age",
    "arpu_to_outflow": "ARPU ÷ average monthly spending",
    "bal_m1_minus_m2_rel": "Balance change from M2 to M1 (relative)",
    "bal_m1_vs_rest": "Latest balance ÷ average of earlier months",
    "bal_last2_vs_older": "Balance: last 2 months ÷ earlier 4",
    "bal_drop_from_peak": "Fall of latest balance from its 6-month peak",
    "bal_above_min": "Latest balance above its 6-month low (relative)",
    "bal_decline_months": "Month-to-month balance falls (of 5)",
    "bal_decline_last3": "Balance falls in the 3 most recent steps",
    "inflow_last2_vs_older": "Money in: last 2 months ÷ earlier 4",
    "inflow_m1_vs_rest": "Money in: latest month ÷ earlier average",
    "outflow_last2_vs_older": "Money out: last 2 months ÷ earlier 4",
    "withdraw_last2_vs_older": "Cash withdrawals: last 2 months ÷ earlier 4",
    "cash_share_last3_vs_older": "Share of spending taken as cash: last 3 ÷ previous 3",
    "bal_m1_z_own": "Latest balance vs own earlier months (z-score)",
    "bal_min_m1_m3": "Lowest balance in the last 3 months",
    "bal_m1_log": "Latest balance (log scale)",
}
_CAT_RE = "|".join(sorted(CATS, key=len, reverse=True))
_METRIC_RE = "|".join(METRIC_LABELS)


def month_label(i):
    i = int(i)
    return "M1 (latest)" if i == 1 else ("M6 (oldest)" if i == 6 else f"M{i}")


def describe(name):
    """Plain-English name for a raw or engineered feature."""
    if name in STATIC_LABELS:
        return STATIC_LABELS[name]
    if name in EXPLICIT_LABELS:
        return EXPLICIT_LABELS[name]
    m = re.fullmatch(r"(.+)_vs_(region|segment|earning_pattern)", name)
    if m:
        return f"{describe(m.group(1))}, compared with the {m.group(2).replace('_', ' ')} median"
    m = re.fullmatch(r"m(\d)_daily_avg_bal", name)
    if m:
        return f"Average daily balance, {month_label(m.group(1))}"
    m = re.fullmatch(rf"m(\d)_({_CAT_RE})_({_METRIC_RE})", name)
    if m:
        return f"{CAT_LABELS[m.group(2)]}: {METRIC_LABELS[m.group(3)]}, {month_label(m.group(1))}"
    m = re.fullmatch(rf"({_CAT_RE})_({_METRIC_RE})_(mean|std|ratio|slope|rel_slope)", name)
    if m:
        return f"{CAT_LABELS[m.group(1)]}: {METRIC_LABELS[m.group(2)]}, {STAT_LABELS[m.group(3)]}"
    m = re.fullmatch(rf"({_CAT_RE})_({'|'.join(CAT_EXTRA)})", name)
    if m:
        return f"{CAT_LABELS[m.group(1)]}: {CAT_EXTRA[m.group(2)]}"
    m = re.fullmatch(r"(inflow|outflow)_(mean|std|ratio|slope)", name)
    if m:
        return f"{MONEY_IN if m.group(1) == 'inflow' else MONEY_OUT}: {STAT_LABELS[m.group(2)]}"
    m = re.fullmatch(r"net_m(\d)(_rel)?", name)
    if m:
        lbl = f"Net cash flow (in − out), {month_label(m.group(1))}"
        return lbl + (" ÷ average money in" if m.group(2) else "")
    m = re.fullmatch(r"bal_m(\d)_cover_out", name)
    if m:
        return f"Balance ÷ spending, {month_label(m.group(1))}"
    return name.replace("_", " ")
