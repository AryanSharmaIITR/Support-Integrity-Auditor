import numpy as np
import pandas as pd
import streamlit as st

from script.train_classifier import SeverityClassifier
from script.evidence_dossier_generation import EvidenceDossierGenerator

MODEL_PATH = "./Artifacts/severity_classifier.joblib"
SEVERITY_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
CATEGORIES = ["General Inquiry", "Technical", "Account", "Billing", "Fraud"]
CHANNELS = ["Email", "Chat", "Web Form"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]

st.set_page_config(page_title="Support Integrity Auditor", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading severity classifier…")
def load_model():
    return SeverityClassifier.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_dossier_generator():
    return EvidenceDossierGenerator()


def predict_severity_proba(clf, df):
    embd_X = clf._build_embeddings(df)
    meta_X = clf._build_meta(df)
    meta_X[clf.SCALE_COLS] = clf.scaler.transform(meta_X[clf.SCALE_COLS])
    meta_X = clf._align_meta_columns(meta_X)

    embd_prob = clf.embd_model.predict_proba(embd_X)
    meta_prob = clf.meta_model.predict_proba(meta_X)
    return clf.weights[0] * embd_prob + clf.weights[1] * meta_prob


def audit(clf, gen, df):
    df = df.copy().reset_index(drop=True)
    proba = predict_severity_proba(clf, df)
    pred_idx = np.argmax(proba, axis=1)

    df["Inferred_Severity"] = [SEVERITY_LABELS[i] for i in pred_idx]
    df["Is_Mismatch"] = (
        df["Priority_Level"].str.upper() != df["Inferred_Severity"].str.upper()
    ).astype(int)

    rank = EvidenceDossierGenerator.SEVERITY_RANK
    df["Severity_Delta"] = [
        rank[sev] - rank[str(prio).upper()]
        for sev, prio in zip(df["Inferred_Severity"], df["Priority_Level"])
    ]

    dossiers = []
    for i, row in df.iterrows():
        scores = {SEVERITY_LABELS[j]: float(proba[i, j]) for j in range(4)}
        dossiers.append(
            gen.generate(row, inferred_severity=row["Inferred_Severity"], scores=scores)
        )
    return df, dossiers


def render_dossier(dossier):
    """Render a single Evidence Dossier with its JSON and a readable view."""
    badge = "🔴 Hidden Crisis" if dossier["mismatch_type"] == "Hidden Crisis" else "🟡 False Alarm"
    st.markdown(
        f"**{dossier['ticket_id']}** — {badge} · "
        f"assigned `{dossier['assigned_priority']}` → inferred `{dossier['inferred_severity']}` "
        f"(Δ {dossier['severity_delta']:+d}) · confidence `{dossier['confidence']}`"
    )
    st.markdown(f"> {dossier['constraint_analysis']}")
    st.dataframe(pd.DataFrame(dossier["feature_evidence"]), use_container_width=True, hide_index=True)
    with st.expander("Raw dossier JSON"):
        st.json(dossier)


def render_dashboard(df):
    flagged = df[df["Is_Mismatch"] == 1]
    total, n_flag = len(df), len(flagged)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets audited", total)
    c2.metric("Flagged mismatches", n_flag, f"{(n_flag / total * 100):.1f}%" if total else "0%")

    delta = flagged["Severity_Delta"]
    c3.metric("Hidden Crises", int((delta > 0).sum()))
    c4.metric("False Alarms", int((delta < 0).sum()))

    if n_flag == 0:
        st.success("No priority mismatches detected.")
        return

    left, right = st.columns(2)
    with left:
        st.subheader("Mismatch type distribution")
        types = flagged["Severity_Delta"].apply(
            lambda d: "Hidden Crisis" if d > 0 else "False Alarm"
        )
        st.bar_chart(types.value_counts())
    with right:
        st.subheader("Flagged tickets by category")
        if "Issue_Category" in flagged.columns:
            st.bar_chart(flagged["Issue_Category"].value_counts())

    st.subheader("Top contributing signals (across flagged tickets)")
    gen = get_dossier_generator()
    signal_counts = {}
    for _, row in flagged.iterrows():
        d = gen.generate(row, inferred_severity=row["Inferred_Severity"])
        if d is None:
            continue
        for ev in d["feature_evidence"]:
            key = ev["value"] if ev["signal"] == "keyword" else ev["signal"]
            signal_counts[f"{ev['signal']}: {key}"] = signal_counts.get(f"{ev['signal']}: {key}", 0) + 1
    if signal_counts:
        top = pd.Series(signal_counts).sort_values(ascending=False).head(15)
        st.bar_chart(top)

    st.subheader("Severity-delta heatmap (category × channel)")
    if {"Issue_Category", "Ticket_Channel"}.issubset(df.columns):
        pivot = df.pivot_table(
            index="Issue_Category",
            columns="Ticket_Channel",
            values="Severity_Delta",
            aggfunc="mean",
        )
        st.dataframe(
            pivot.style.format("{:.2f}").background_gradient(cmap="RdBu_r", axis=None),
            use_container_width=True,
        )
        st.caption(
            "Mean severity delta (inferred − assigned). Positive (red) = systematic "
            "under-prioritisation; negative (blue) = inflation."
        )


def single_ticket_page(clf, gen):
    st.header("Single-ticket audit")
    with st.form("ticket_form"):
        col1, col2 = st.columns(2)
        with col1:
            ticket_id = st.text_input("Ticket ID", "TKT-DEMO-001")
            subject = st.text_input("Ticket subject", "Cannot log into my account")
            category = st.selectbox("Issue category", CATEGORIES, index=1)
            channel = st.selectbox("Ticket channel", CHANNELS, index=0)
        with col2:
            priority = st.selectbox("Assigned priority", PRIORITIES, index=0)
            resolution = st.number_input("Resolution time (hours)", 0, 500, 36)
            satisfaction = st.slider("Satisfaction score", 1, 5, 3)
        description = st.text_area(
            "Ticket description",
            "I cannot log in even after resetting my password and the 2FA check keeps failing.",
            height=120,
        )
        submitted = st.form_submit_button("Audit ticket", type="primary")

    if not submitted:
        return

    row = pd.DataFrame([{
        "Ticket_ID": ticket_id,
        "Ticket_Subject": subject,
        "Ticket_Description": description,
        "Issue_Category": category,
        "Priority_Level": priority,
        "Ticket_Channel": channel,
        "Resolution_Time_Hours": resolution,
        "Satisfaction_Score": satisfaction,
    }])

    result, dossiers = audit(clf, gen, row)
    inferred = result.loc[0, "Inferred_Severity"]
    is_mismatch = bool(result.loc[0, "Is_Mismatch"])

    if is_mismatch:
        st.error(f"⚠️ PRIORITY MISMATCH — assigned `{priority}`, inferred severity `{inferred.title()}`.")
        render_dossier(dossiers[0])
    else:
        st.success(f"✅ CONSISTENT — assigned priority matches inferred severity (`{inferred.title()}`).")


def batch_page(clf, gen):
    st.header("Batch CSV audit")
    st.caption(
        "Upload a CSV with columns: Ticket_Subject, Ticket_Description, Issue_Category, "
        "Priority_Level, Ticket_Channel, Resolution_Time_Hours, Satisfaction_Score "
        "(Ticket_ID optional)."
    )
    uploaded = st.file_uploader("Ticket CSV", type=["csv"])
    use_sample = st.checkbox("Use a sample from dataset/preprocessed_data.csv instead")

    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample:
        n = st.slider("Sample size", 20, 1000, 200, step=20)
        df = pd.read_csv("./dataset/preprocessed_data.csv").sample(n, random_state=42)

    if df is None:
        return

    required = {"Ticket_Subject", "Ticket_Description", "Issue_Category",
                "Priority_Level", "Ticket_Channel", "Resolution_Time_Hours",
                "Satisfaction_Score"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing required columns: {sorted(missing)}")
        return
    if "Ticket_ID" not in df.columns:
        df["Ticket_ID"] = [f"TKT-{i}" for i in range(len(df))]

    with st.spinner(f"Auditing {len(df)} tickets…"):
        result, dossiers = audit(clf, gen, df)

    st.divider()
    render_dashboard(result)

    st.divider()
    st.subheader("Results")
    show_cols = ["Ticket_ID", "Priority_Level", "Inferred_Severity",
                 "Is_Mismatch", "Severity_Delta", "Issue_Category", "Ticket_Channel"]
    st.dataframe(result[[c for c in show_cols if c in result.columns]],
                 use_container_width=True, hide_index=True)
    st.download_button(
        "Download audited CSV",
        result.to_csv(index=False).encode("utf-8"),
        "sia_audit_results.csv",
        "text/csv",
    )

    flagged_dossiers = [d for d in dossiers if d is not None]
    st.subheader(f"Evidence Dossiers ({len(flagged_dossiers)} flagged)")
    for d in flagged_dossiers[:50]:
        with st.expander(f"{d['ticket_id']} · {d['mismatch_type']} (Δ {d['severity_delta']:+d})"):
            render_dossier(d)
    if len(flagged_dossiers) > 50:
        st.caption(f"Showing first 50 of {len(flagged_dossiers)} dossiers. Download the CSV for the full set.")


def main():
    st.title("🛡️ Support Integrity Auditor (SIA)")
    st.caption(
        "Detects priority mismatches between a ticket's objective characteristics "
        "and its human-assigned priority, with grounded Evidence Dossiers."
    )

    clf = load_model()
    gen = get_dossier_generator()

    page = st.sidebar.radio("Mode", ["Single ticket", "Batch CSV"])
    if page == "Single ticket":
        single_ticket_page(clf, gen)
    else:
        batch_page(clf, gen)


if __name__ == "__main__":
    main()
