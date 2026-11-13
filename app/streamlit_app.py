"""
Liquidity Stress Predictor - Streamlit app for the model built in
claude_solution.ipynb (LightGBM + CatBoost + XGBoost blend, calibrated).

Run locally (from the project folder):
    venv\\Scripts\\activate      (or your own environment)
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

The model file app/model/stress_model.joblib is created by section 10 of
claude_solution.ipynb.
"""
import io
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import stress_model as sm

APP_DIR = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("STRESS_MODEL_PATH", APP_DIR / "model" / "stress_model.joblib"))

MONTH_COLS = ["M1 (latest)", "M2", "M3", "M4", "M5", "M6 (oldest)"]
CHART_MONTHS = ["M6", "M5", "M4", "M3", "M2", "M1"]          # oldest -> latest
UP_COLOUR, DOWN_COLOUR = "#d64545", "#2e9e6b"                  # raises / lowers risk

st.set_page_config(page_title="Liquidity Stress Predictor", page_icon=":material/savings:",
                   layout="wide")


# =============================================================================
# Loading and cached computation
# =============================================================================
@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.title("Liquidity Stress Predictor")
    st.error("The model file was not found.", icon=":material/error:")
    st.markdown(
        f"Expected **`{MODEL_PATH.relative_to(APP_DIR.parent)}`**.\n\n"
        "Create it by running **section 10 (Export the model for the Streamlit app)** "
        "of `claude_solution.ipynb`, then refresh this page."
    )
    st.stop()

MODEL = load_model()
META = MODEL.metadata
BASE_RATE = META.get("base_rate", 0.15)
REF = np.asarray(META.get("oof_reference", []))
TYPICAL = META.get("feature_medians", {})


@st.cache_data(show_spinner=False, max_entries=64)
def explain_raw(raw_row: pd.DataFrame):
    """Prediction + approximate SHAP contributions for one raw row."""
    X = MODEL.prepare(raw_row)
    fam = MODEL.family_probs(X)
    p = float(MODEL._finish(fam)[0])
    contrib, base = MODEL.explain(X)
    return p, {k: float(v[0]) for k, v in fam.items()}, contrib, base, X.iloc[0]


@st.cache_data(show_spinner=False, max_entries=3)
def score_file(data: bytes):
    df = pd.read_csv(io.BytesIO(data))
    clean, errors, warns = sm.validate_raw(df, MODEL.cat_levels)
    if errors:
        return {"errors": errors, "warns": warns}
    ids = (clean["ID"].astype(str) if "ID" in clean.columns
           else pd.Series([f"row_{i + 1}" for i in range(len(clean))]))
    clean = clean.assign(ID=ids.values).reset_index(drop=True)
    X = MODEL.prepare(clean)
    p = MODEL.predict_from_features(X)
    res = pd.DataFrame({
        "ID": clean["ID"].values,
        "probability": p,
        "risk_band": [sm.risk_band(v)[0] for v in p],
        "bal_ratio": X["bal_ratio"].values if "bal_ratio" in X else np.nan,
        "out_in_ratio_last3": X["out_in_ratio_last3"].values if "out_in_ratio_last3" in X else np.nan,
        "neg_net_months_last3": X["neg_net_months_last3"].values if "neg_net_months_last3" in X else np.nan,
    })
    return {"errors": [], "warns": warns, "results": res,
            "raw": clean[["ID"] + sm.RAW_FEATURES]}


# =============================================================================
# Small helpers
# =============================================================================
def pct(p, digits=1):
    return f"{p * 100:.{digits}f}%"


def percentile_of(p):
    return float(np.searchsorted(REF, p) / len(REF) * 100) if len(REF) else None


def fmt_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (str, bool)) or not np.isscalar(v):
        return str(v)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1 or v == 0:
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}"


def band_badge(p, size="1.05rem"):
    name, colour, _ = sm.risk_band(p)
    return (f"<span style='display:inline-block;padding:0.2rem 0.75rem;border-radius:999px;"
            f"border:1.5px solid {colour};background:{colour}22;color:{colour};"
            f"font-weight:700;font-size:{size}'>{name} risk</span>")


def shorten(text, n=72):
    return text if len(text) <= n else text[: n - 1] + "…"


def empty_row():
    row = {c: 0.0 for c in sm.RAW_FEATURES}
    for c in sm.CAT_COLS:
        row[c] = MODEL.cat_levels[c][0]
    row.update(age=35, arpu=10000.0, x_90_d_activity_rate=0.5)
    return row


def examples():
    ex = dict(META.get("examples", {}))
    ex["Empty form (all transactions 0)"] = empty_row()
    return ex


def set_form_row(row, source):
    st.session_state.form_row = dict(row)
    st.session_state.form_source = source
    st.session_state.form_version = st.session_state.get("form_version", 0) + 1
    st.session_state.pop("single_result", None)


if "form_row" not in st.session_state:
    first = next(iter(examples().items()))
    set_form_row(first[1], first[0])


def consistency_warnings(row):
    out = []
    for cat in sm.CATS:
        cp = sm.CATS[cat]
        for i in range(1, 7):
            vol = row[f"m{i}_{cat}_volume"] or 0
            tot = row[f"m{i}_{cat}_total_value"] or 0
            hi = row[f"m{i}_{cat}_highest_amount"] or 0
            n_cp = row[f"m{i}_{cat}_{cp}"] or 0
            where = f"{sm.CAT_LABELS[cat]}, M{i}"
            if vol == 0 and (tot > 0 or hi > 0):
                out.append(f"{where}: value entered but number of transactions is 0.")
            if hi > tot + 1e-9:
                out.append(f"{where}: largest transaction is bigger than the total value.")
            if n_cp > vol and vol > 0:
                out.append(f"{where}: more {sm.METRIC_LABELS[cp]} than transactions.")
    return out


# =============================================================================
# Charts
# =============================================================================
def risk_scale_chart(p):
    fig = go.Figure()
    lo = 0.0
    for upper, name, colour, _ in sm.RISK_BANDS:
        hi = min(upper, 1.0)
        fig.add_trace(go.Bar(x=[(hi - lo) * 100], y=[""], base=[lo * 100], orientation="h",
                             marker=dict(color=colour, opacity=0.35, line=dict(width=0)),
                             hoverinfo="skip", showlegend=False, width=0.5))
        fig.add_annotation(x=(lo + hi) / 2 * 100, y=-0.55, text=name, showarrow=False,
                           font=dict(size=12), yref="y")
        lo = hi
    fig.add_trace(go.Scatter(x=[p * 100], y=[""], mode="markers+text",
                             marker=dict(symbol="diamond", size=20, color=sm.risk_band(p)[1],
                                         line=dict(width=2, color="white")),
                             text=[pct(p)], textposition="top center",
                             textfont=dict(size=14), hoverinfo="skip", showlegend=False))
    fig.add_vline(x=BASE_RATE * 100, line_dash="dot", line_width=1.5,
                  annotation_text=f"average {pct(BASE_RATE, 0)}", annotation_position="bottom right",
                  annotation_font_size=11)
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10), barmode="overlay",
                      xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
                      yaxis=dict(showticklabels=False, range=[-0.9, 0.6]))
    return fig


def drivers_chart(df):
    df = df.iloc[::-1]
    colours = [UP_COLOUR if v > 0 else DOWN_COLOUR for v in df["contribution"]]
    fig = go.Figure(go.Bar(
        x=df["contribution"], y=[shorten(t, 60) for t in df["label"]], orientation="h",
        marker_color=colours,
        customdata=np.stack([df["label"], df["value_txt"]], axis=1),
        hovertemplate="<b>%{customdata[0]}</b><br>Value: %{customdata[1]}"
                      "<br>Effect: %{x:+.3f} log-odds<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1)
    fig.update_layout(height=34 * len(df) + 70, margin=dict(l=10, r=10, t=10, b=40),
                      xaxis_title="← lowers risk          effect on risk (log-odds)          raises risk →")
    return fig


def cashflow_charts(row):
    bal = [row.get(f"m{i}_daily_avg_bal") for i in range(6, 0, -1)]
    inflow = [sum((row.get(f"m{i}_{c}_total_value") or 0) for c in sm.INFLOW) for i in range(6, 0, -1)]
    outflow = [sum((row.get(f"m{i}_{c}_total_value") or 0) for c in sm.OUTFLOW) for i in range(6, 0, -1)]

    f1 = go.Figure(go.Scatter(x=CHART_MONTHS, y=bal, mode="lines+markers",
                              line=dict(width=3), name="Average daily balance",
                              hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
    f1.update_layout(title="Average daily balance", height=320,
                     margin=dict(l=10, r=10, t=50, b=10), xaxis_title="month (oldest → latest)")

    f2 = go.Figure([
        go.Bar(x=CHART_MONTHS, y=inflow, name="Money in", marker_color=DOWN_COLOUR,
               hovertemplate="%{x}: %{y:,.0f}<extra>Money in</extra>"),
        go.Bar(x=CHART_MONTHS, y=outflow, name="Money out", marker_color=UP_COLOUR,
               hovertemplate="%{x}: %{y:,.0f}<extra>Money out</extra>"),
    ])
    f2.update_layout(title="Money in vs money out", barmode="group", height=320,
                     margin=dict(l=10, r=10, t=50, b=10), xaxis_title="month (oldest → latest)",
                     legend=dict(orientation="h", y=1.12, x=1, xanchor="right"))
    return f1, f2


# =============================================================================
# Shared result views
# =============================================================================
def render_headline(p, previous=None):
    perc = percentile_of(p)
    name, colour, meaning = sm.risk_band(p)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Probability of liquidity stress (next 30 days)", pct(p),
                  delta=f"{(p - BASE_RATE) * 100:+.1f} pts vs average customer",
                  delta_color="inverse", border=True)
    with c2:
        with st.container(border=True):
            st.caption("Risk band")
            st.markdown(band_badge(p, "1.3rem"), unsafe_allow_html=True)
            st.caption(meaning)
    with c3:
        with st.container(border=True):
            st.caption("Compared with the training population")
            if perc is not None:
                st.markdown(f"<span style='font-size:1.9rem;font-weight:700'>{perc:.0f}%</span>",
                            unsafe_allow_html=True)
                st.caption(f"of customers had a lower predicted risk. "
                           f"That is {p / BASE_RATE:.1f}× the average risk.")
    if previous is not None and abs(previous - p) > 1e-9:
        arrow = "up" if p > previous else "down"
        st.caption(f"Your previous prediction was {pct(previous)}; this one is "
                   f"{abs(p - previous) * 100:.1f} points {arrow}.")
    st.plotly_chart(risk_scale_chart(p), width="stretch", config={"displayModeBar": False})


def render_drivers(contrib, X_row, key):
    n = st.slider("Number of factors to show", 5, 25, 10, key=f"n_drivers_{key}")
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(n)
    df = pd.DataFrame({
        "feature": top.index,
        "label": [sm.describe(f) for f in top.index],
        "contribution": top.values,
        "value_txt": [fmt_value(X_row[f]) for f in top.index],
    })
    st.plotly_chart(drivers_chart(df), width="stretch", config={"displayModeBar": False})

    table = pd.DataFrame({
        "Factor": df["label"],
        "This customer": df["value_txt"],
        "Typical – not stressed": [fmt_value(TYPICAL.get("not_stressed", {}).get(f)) for f in df["feature"]],
        "Typical – stressed": [fmt_value(TYPICAL.get("stressed", {}).get(f)) for f in df["feature"]],
        "Effect": ["▲ raises risk" if v > 0 else "▼ lowers risk" for v in df["contribution"]],
    })
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "Effects are approximate SHAP values from the tree models, averaged over the fold models "
        "and combined with the blend weights. They show which inputs pushed **this** prediction "
        "up or down; they do not prove cause and effect. “Typical” columns are medians "
        "of training customers who were / were not stressed."
    )


def render_model_details(fam):
    rows = []
    for n, w in zip(MODEL.names, MODEL.weights):
        rows.append({"Model": {"lgb": "LightGBM", "cat": "CatBoost", "xgb": "XGBoost"}.get(n, n),
                     "Blend weight": f"{w:.1%}", "Probability": pct(fam[n], 2),
                     "Fold models": len(MODEL.fitted[n])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The final probability combines the models in log-odds space with the weights "
               "above, then applies the calibration fitted on out-of-fold predictions.")


# =============================================================================
# Page 1: score one customer
# =============================================================================
def single_page():
    st.title("Score a customer")
    st.markdown("Enter a customer's profile and the last **6 months** of mobile-money activity "
                "to estimate the chance of **liquidity stress in the next 30 days**. "
                "Start from an example and change any value to run a *what-if*.")

    ex = examples()
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 3], vertical_alignment="bottom")
        choice = c1.selectbox("Start from", list(ex), key="example_choice")
        if c2.button("Load", width="stretch"):
            set_form_row(ex[choice], choice)
            st.rerun()
        c3.caption(f"Currently loaded: **{st.session_state.form_source}**")

    row = st.session_state.form_row
    v = st.session_state.form_version

    with st.form("customer_form", border=False):
        t_prof, t_bal, t_txn = st.tabs([":material/person: Profile",
                                        ":material/account_balance_wallet: Balances",
                                        ":material/swap_horiz: Transactions"])
        new = dict(row)

        with t_prof:
            a, b = st.columns(2)
            new["age"] = a.number_input("Age", 18, 100, int(row["age"]), key=f"age_{v}")
            new["arpu"] = b.number_input(
                sm.STATIC_LABELS["arpu"], 0.0, value=float(row["arpu"]), step=100.0,
                key=f"arpu_{v}", help="Average revenue per user over the observation window.")
            new["x_90_d_activity_rate"] = a.slider(
                sm.STATIC_LABELS["x_90_d_activity_rate"], 0.0, 1.0,
                float(row["x_90_d_activity_rate"]), 0.01, key=f"act_{v}",
                help="Proportion of the last 90 days with any mobile-money activity.")
            for i, c in enumerate(sm.CAT_COLS):
                col = b if i % 2 == 0 else a
                levels = MODEL.cat_levels[c]
                idx = levels.index(row[c]) if row[c] in levels else 0
                new[c] = col.selectbox(sm.STATIC_LABELS[c], levels, index=idx, key=f"{c}_{v}")

        with t_bal:
            st.caption("Average daily wallet balance per month. M1 is the most recent month.")
            bal_df = pd.DataFrame([[row[f"m{i}_daily_avg_bal"] for i in range(1, 7)]],
                                  columns=MONTH_COLS, index=["Average daily balance"])
            bal_out = st.data_editor(
                bal_df, key=f"bal_{v}", width="stretch",
                column_config={m: st.column_config.NumberColumn(m, format="%.2f") for m in MONTH_COLS})

        with t_txn:
            st.caption("For each transaction type: count, total value, largest single amount and "
                       "number of distinct counterparties per month. Leave unused months at 0.")
            txn_out = {}
            cat_tabs = st.tabs([sm.CAT_LABELS[c] for c in sm.CATS])
            for tab, cat in zip(cat_tabs, sm.CATS):
                with tab:
                    metrics = sm.metrics_for(cat)
                    df = pd.DataFrame(
                        [[row[f"m{i}_{cat}_{m}"] for i in range(1, 7)] for m in metrics],
                        columns=MONTH_COLS,
                        index=[sm.METRIC_LABELS[m].capitalize() for m in metrics])
                    txn_out[cat] = st.data_editor(
                        df, key=f"txn_{cat}_{v}", width="stretch",
                        column_config={m: st.column_config.NumberColumn(m, min_value=0)
                                       for m in MONTH_COLS})

        submitted = st.form_submit_button("Predict stress risk", type="primary",
                                          icon=":material/insights:", width="stretch")

    if submitted:
        for j, i in enumerate(range(1, 7)):
            new[f"m{i}_daily_avg_bal"] = bal_out.iloc[0, j]
        for cat, out in txn_out.items():
            for r, m in enumerate(sm.metrics_for(cat)):
                for j, i in enumerate(range(1, 7)):
                    val = out.iloc[r, j]
                    new[f"m{i}_{cat}_{m}"] = 0.0 if pd.isna(val) else float(val)

        raw = pd.DataFrame([new])[sm.RAW_FEATURES]
        clean, errors, warns = sm.validate_raw(raw, MODEL.cat_levels)
        if errors:
            for e in errors:
                st.error(e)
            return
        prev = st.session_state.get("single_result", {}).get("p")
        with st.spinner("Scoring and explaining..."):
            p, fam, contrib, base, X_row = explain_raw(clean)
        st.session_state.form_row = new
        st.session_state.single_result = dict(p=p, fam=fam, contrib=contrib, X_row=X_row,
                                              row=new, prev=prev,
                                              warns=warns + consistency_warnings(new))

    res = st.session_state.get("single_result")
    if not res:
        st.info("Fill in or adjust the form, then press **Predict stress risk**.",
                icon=":material/arrow_upward:")
        return

    st.divider()
    st.subheader("Result")
    if res["warns"]:
        with st.expander(f"{len(res['warns'])} input check(s) to review", icon=":material/warning:"):
            for w in res["warns"][:30]:
                st.markdown(f"- {w}")
    render_headline(res["p"], res["prev"])

    t1, t2, t3 = st.tabs([":material/psychology: Why this score",
                          ":material/monitoring: Cash-flow picture",
                          ":material/tune: Model details"])
    with t1:
        render_drivers(res["contrib"], res["X_row"], "single")
    with t2:
        X = res["X_row"]
        k = st.columns(4)
        kpis = [("out_in_ratio_last3", "Money out ÷ money in (last 3 months)"),
                ("neg_net_months_last3", "Months spending exceeded income (last 3)"),
                ("bal_ratio", "Balance: last 3 months ÷ previous 3"),
                ("runway_m1", "Latest balance ÷ latest month's spending")]
        for col, (f, label) in zip(k, kpis):
            typ = TYPICAL.get("not_stressed", {}).get(f)
            col.metric(label, fmt_value(X.get(f)), border=True,
                       help=f"Typical for a customer who was not stressed: {fmt_value(typ)}")
        f1, f2 = cashflow_charts(res["row"])
        a, b = st.columns(2)
        a.plotly_chart(f1, width="stretch", config={"displayModeBar": False})
        b.plotly_chart(f2, width="stretch", config={"displayModeBar": False})
    with t3:
        render_model_details(res["fam"])


# =============================================================================
# Page 2: score a file
# =============================================================================
def template_csv():
    rows = [dict(r, ID=f"EXAMPLE_{i + 1}") for i, r in enumerate(META.get("examples", {}).values())]
    if not rows:
        rows = [dict(empty_row(), ID="EXAMPLE_1")]
    return pd.DataFrame(rows)[["ID"] + sm.RAW_FEATURES].to_csv(index=False).encode()


def batch_page():
    st.title("Score a file")
    st.markdown(
        "Upload a CSV with one row per customer snapshot, in the same format as the "
        f"competition's `Test.csv`: **{len(sm.RAW_FEATURES)} input columns** plus an optional `ID`. "
        "Extra columns are ignored.")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    up = c1.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    c2.download_button("Download a template", template_csv(), "stress_template.csv", "text/csv",
                       icon=":material/download:", width="stretch")

    if up is None:
        st.info("No file uploaded yet. The template contains example rows you can edit.",
                icon=":material/upload_file:")
        return

    with st.spinner(f"Scoring {up.name}... large files can take a minute."):
        out = score_file(up.getvalue())
    if out["errors"]:
        for e in out["errors"]:
            st.error(e, icon=":material/error:")
        return
    if out["warns"]:
        with st.expander(f"{len(out['warns'])} data warning(s)", icon=":material/warning:"):
            for w in out["warns"][:50]:
                st.markdown(f"- {w}")

    res = out["results"]
    n = len(res)
    high = res["risk_band"].isin(["High", "Very high"]).sum()
    k = st.columns(4)
    k[0].metric("Customers scored", f"{n:,}", border=True)
    k[1].metric("Average probability", pct(res["probability"].mean()), border=True,
                help=f"Training base rate: {pct(BASE_RATE)}")
    k[2].metric("High or very high risk", f"{high:,}", f"{high / n:.1%} of file",
                delta_color="off", border=True)
    k[3].metric("Highest probability", pct(res["probability"].max()), border=True)

    a, b = st.columns(2)
    counts = res["risk_band"].value_counts()
    bands = [b_[1] for b_ in sm.RISK_BANDS]
    fig = go.Figure(go.Bar(x=bands, y=[int(counts.get(x, 0)) for x in bands],
                           marker_color=[b_[2] for b_ in sm.RISK_BANDS],
                           text=[f"{int(counts.get(x, 0)):,}" for x in bands], textposition="outside",
                           hovertemplate="%{x}: %{y:,}<extra></extra>"))
    fig.update_layout(title="Customers per risk band", height=340, margin=dict(l=10, r=10, t=50, b=10))
    a.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    fig = go.Figure(go.Histogram(x=res["probability"] * 100, nbinsx=50,
                                 hovertemplate="%{x}%: %{y:,} customers<extra></extra>"))
    for upper, *_ in sm.RISK_BANDS[:-1]:
        fig.add_vline(x=upper * 100, line_dash="dot", line_width=1)
    fig.update_layout(title="Distribution of predicted probabilities", height=340,
                      margin=dict(l=10, r=10, t=50, b=10), xaxis_title="probability (%)",
                      yaxis_title="customers", bargap=0.05)
    b.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.subheader("Results")
    f1, f2 = st.columns([2, 2])
    pick = f1.multiselect("Risk bands", bands, default=bands)
    q = f2.text_input("Search ID", placeholder="e.g. ID_ED91AC7A28")
    view = res[res["risk_band"].isin(pick)]
    if q:
        view = view[view["ID"].str.contains(q.strip(), case=False, regex=False)]
    view = view.sort_values("probability", ascending=False)
    show = view.assign(probability=view["probability"] * 100)
    st.dataframe(
        show, hide_index=True, width="stretch", height=380,
        column_config={
            "ID": "ID",
            "probability": st.column_config.ProgressColumn(
                "Stress probability", min_value=0, max_value=100, format="%.1f%%"),
            "risk_band": "Risk band",
            "bal_ratio": st.column_config.NumberColumn(
                "Balance: last 3 ÷ prev 3", format="%.2f", help=sm.describe("bal_ratio")),
            "out_in_ratio_last3": st.column_config.NumberColumn(
                "Out ÷ in (last 3)", format="%.2f", help=sm.describe("out_in_ratio_last3")),
            "neg_net_months_last3": st.column_config.NumberColumn(
                "Deficit months (last 3)", format="%d", help=sm.describe("neg_net_months_last3")),
        })
    st.caption(f"Showing {len(view):,} of {n:,} rows, riskiest first.")

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download results (CSV)",
        res.rename(columns={"probability": "stress_probability"}).to_csv(index=False).encode(),
        "stress_predictions.csv", "text/csv", icon=":material/download:", width="stretch")
    d2.download_button(
        "Download in Zindi submission format",
        res[["ID", "probability"]].rename(columns={"probability": "Target"}).to_csv(index=False).encode(),
        "submission_from_app.csv", "text/csv", icon=":material/upload:", width="stretch")

    st.subheader("Explain a customer")
    options = view["ID"].head(2000).tolist()
    if not options:
        st.caption("No rows match the current filters.")
        return
    sel = st.selectbox("Customer ID (riskiest first, within the current filters)", options)
    raw_row = out["raw"][out["raw"]["ID"] == sel].head(1)
    raw_row = raw_row[sm.RAW_FEATURES].reset_index(drop=True)
    with st.spinner("Explaining..."):
        p, fam, contrib, base, X_row = explain_raw(raw_row)
    c1, c2 = st.columns([3, 1], vertical_alignment="center")
    c1.markdown(f"**{sel}**: {pct(p)} &nbsp; {band_badge(p, '0.95rem')}", unsafe_allow_html=True)
    if c2.button("Open in what-if editor", icon=":material/edit:", width="stretch"):
        set_form_row(raw_row.iloc[0].to_dict(), f"{sel} (from {up.name})")
        st.switch_page(PAGES["single"])
    render_drivers(contrib, X_row, "batch")


# =============================================================================
# Page 3: about
# =============================================================================
def about_page():
    st.title("About the model")
    st.markdown(
        "This app predicts the probability that a mobile-money customer experiences **liquidity "
        "stress in the next 30 days**, using their profile and 6 months of transaction history. "
        "It was built for the Zindi *Liquidity Stress Prediction* challenge.")

    k = st.columns(4)
    k[0].metric("Cross-validated log loss", f"{META.get('cv_logloss', float('nan')):.4f}",
                border=True, help="Lower is better. Out-of-fold, after calibration.")
    k[1].metric("Cross-validated ROC-AUC", f"{META.get('cv_auc', float('nan')):.4f}",
                border=True, help="Higher is better. Out-of-fold.")
    k[2].metric("Training rows", f"{META.get('n_train', 0):,}", border=True)
    k[3].metric("Model features", f"{len(MODEL.features):,}", border=True,
                help=f"Built from {len(sm.RAW_FEATURES)} raw input columns.")

    st.subheader("How a prediction is made")
    st.markdown(
        f"""
1. **Feature engineering.** The {len(sm.RAW_FEATURES)} raw columns become {len(MODEL.features)} features:
   trends, recent-vs-older ratios, money in vs money out, balance cover, activity streaks, and
   comparisons with the customer's region, segment and earning-pattern medians.
2. **Three model families.** LightGBM, CatBoost and XGBoost are each trained with
   {META.get('n_folds', 5)}-fold cross-validation and several random seeds. Every fold model is
   kept, and their predictions are averaged.
3. **Blend.** The families are combined in log-odds space, with weights chosen to minimise
   out-of-fold log loss.
4. **Calibration.** A final Platt-scaling step keeps the probabilities honest, so that, for example,
   customers given 20% are stressed about 20% of the time.
""")
    fm = META.get("family_metrics", {})
    if fm:
        rows = [{"Model": {"lgb": "LightGBM", "cat": "CatBoost", "xgb": "XGBoost"}.get(n, n),
                 "Blend weight": f"{w:.1%}",
                 "OOF log loss": f"{fm[n]['logloss']:.4f}", "OOF ROC-AUC": f"{fm[n]['auc']:.4f}",
                 "Fold models": len(MODEL.fitted[n])}
                for n, w in zip(MODEL.names, MODEL.weights)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Risk bands")
    lo = 0.0
    rows = []
    for upper, name, colour, meaning in sm.RISK_BANDS:
        rows.append({"Band": name, "Probability": f"{pct(lo, 0)} – {pct(min(upper, 1), 0)}",
                     "Meaning": meaning})
        lo = upper
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"For reference, {pct(BASE_RATE, 0)} of training customers were stressed.")

    st.subheader("Limitations")
    st.markdown(
        """
- The model learned from the competition data only. Its probabilities assume new customers
  look like that population, so treat scores for very different customers with caution.
- Explanations are approximations of how the trees used each input. They show association,
  not cause.
- Uploaded files are processed in memory for this session and are not stored by the app.
""")
    st.caption(f"Model exported {META.get('exported_at', 'unknown')}.")


# =============================================================================
# Navigation
# =============================================================================
PAGES = {
    "single": st.Page(single_page, title="Score a customer", icon=":material/person_search:",
                      default=True),
    "batch": st.Page(batch_page, title="Score a file", icon=":material/upload_file:"),
    "about": st.Page(about_page, title="About the model", icon=":material/info:"),
}
nav = st.navigation(list(PAGES.values()))

with st.sidebar:
    st.markdown("### :material/savings: Liquidity Stress Predictor")
    st.caption("Chance that a mobile-money customer runs short of cash in the next 30 days.")
    with st.container(border=True):
        st.caption("Model")
        st.markdown(
            f"Blend of **{', '.join({'lgb': 'LightGBM', 'cat': 'CatBoost', 'xgb': 'XGBoost'}.get(n, n) for n in MODEL.names)}**  \n"
            f"{sum(len(m) for m in MODEL.fitted.values())} fold models · "
            f"{len(MODEL.features)} features  \n"
            f"CV log loss **{META.get('cv_logloss', float('nan')):.4f}** · "
            f"AUC **{META.get('cv_auc', float('nan')):.4f}**")

nav.run()
