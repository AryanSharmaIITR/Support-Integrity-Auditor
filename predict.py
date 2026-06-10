import json
import argparse
import numpy as np
import pandas as pd

from script.train_classifier import SeverityClassifier
from script.evidence_dossier_generation import EvidenceDossierGenerator

SEVERITY = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
REQUIRED = ["Ticket_Subject", "Ticket_Description", "Issue_Category",
            "Priority_Level", "Ticket_Channel", "Resolution_Time_Hours",
            "Satisfaction_Score"]


def severity_proba(clf, df):
    
    embd_X = clf._build_embeddings(df)
    meta_X = clf._build_meta(df)
    meta_X[clf.SCALE_COLS] = clf.scaler.transform(meta_X[clf.SCALE_COLS])
    meta_X = clf._align_meta_columns(meta_X)
    
    p_embd = clf.embd_model.predict_proba(embd_X)
    p_meta = clf.meta_model.predict_proba(meta_X)
    
    return clf.weights[0] * p_embd + clf.weights[1] * p_meta


def run(input_path, model_path, pred_out, dossier_out):
    
    df = pd.read_csv(input_path)
    
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {missing}")
    
    if "Ticket_ID" not in df.columns:
        df["Ticket_ID"] = [f"TKT-{i}" for i in range(len(df))]

    print(f"Loaded {len(df)} tickets, loading model...")
    
    clf = SeverityClassifier.load(model_path)
    gen = EvidenceDossierGenerator()

    proba = severity_proba(clf, df)
    
    df["Inferred_Severity"] = [SEVERITY[i] for i in np.argmax(proba, axis=1)]
    df["Is_Mismatch"] = (df["Priority_Level"].str.upper() != df["Inferred_Severity"].str.upper()).astype(int)

    dossiers = []
    
    for i, row in df.iterrows():
        
        if not df.loc[i, "Is_Mismatch"]:
            continue
            
        scores = {SEVERITY[j]: float(proba[i, j]) for j in range(4)}
        d = gen.generate(row, inferred_severity=row["Inferred_Severity"], scores=scores)
        
        if d:
            dossiers.append(d)

    df.to_csv(pred_out, index=False)
    
    with open(dossier_out, "w") as f:
        json.dump(dossiers, f, indent=2, ensure_ascii=False)

    print(f"Flagged {df['Is_Mismatch'].sum()}/{len(df)} tickets as mismatches")
    print(f"Predictions -> {pred_out}")
    print(f"Dossiers    -> {dossier_out}")


def main():
    ap = argparse.ArgumentParser(description="Run SIA inference on a ticket CSV.")
    ap.add_argument("--input", required=True, help="path to the ticket CSV")
    ap.add_argument("--model", default="./Artifacts/severity_classifier.joblib")
    ap.add_argument("--pred-out", default="predictions.csv")
    ap.add_argument("--dossier-out", default="dossiers.json")
    args = ap.parse_args()
    run(args.input, args.model, args.pred_out, args.dossier_out)


if __name__ == "__main__":
    main()