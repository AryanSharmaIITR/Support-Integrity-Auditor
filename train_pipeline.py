"""
    python train_pipeline.py
    python train_pipeline.py --data ./dataset/preprocessed_data.csv --out ./Artifacts/severity_classifier.joblib
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

from script.train_classifier import SeverityClassifier

MIN_ACC = 0.83
MIN_F1 = 0.82
MIN_RECALL = 0.78


def train_and_eval(data_path, out_path, weights, test_size, seed):
    clf = SeverityClassifier(weights=weights)
    df = clf.load_data(data_path)
    print(f"Loaded {len(df)} tickets from {data_path}")

    # stratified split on the inferred severity
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=df["Final_severity_label"], random_state=seed
    )

    embd = clf._build_embeddings(df)
    meta = clf._build_meta(df)

    embd_tr, embd_te = embd.loc[train_idx], embd.loc[test_idx]

    meta_tr = meta.loc[train_idx].copy()
    meta_tr[clf.SCALE_COLS] = clf.scaler.fit_transform(meta_tr[clf.SCALE_COLS])
    clf._meta_columns = meta_tr.columns.tolist()

    meta_te = meta.loc[test_idx].copy()
    meta_te[clf.SCALE_COLS] = clf.scaler.transform(meta_te[clf.SCALE_COLS])
    meta_te = clf._align_meta_columns(meta_te)

    y_tr = df["Final_severity_label"].loc[train_idx]

    print("TRAINING")
    clf.embd_model.fit(embd_tr, y_tr)
    clf.meta_model.fit(meta_tr, y_tr)

    # decomposition: predict severity, then mismatch = (priority != predicted severity)
    print("EVALUATING")
    pred_sev = clf._predict_from_features(embd_te, meta_te)
    priority = df["Priority_Level"].loc[test_idx].values          # already int-encoded by load_data
    y_true = df["Is_Mismatch"].loc[test_idx].values
    y_pred = (priority != pred_sev).astype(int)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average=None)

    print("\n" + classification_report(y_true, y_pred, target_names=["Consistent", "Mismatch"]))

    print("=" * 50)
    print(f"Accuracy   : {acc:.4f}   (need >= {MIN_ACC})")
    print(f"Macro F1   : {macro_f1:.4f}   (need >= {MIN_F1})")
    print(f"Recall[0]  : {rec[0]:.4f}   (need >= {MIN_RECALL})")
    print(f"Recall[1]  : {rec[1]:.4f}   (need >= {MIN_RECALL})")
    passed = acc >= MIN_ACC and macro_f1 >= MIN_F1 and rec.min() >= MIN_RECALL
    print("VERDICT    :", "PASS ✅" if passed else "FAIL ❌")
    print("=" * 50)

    clf.save(out_path)
    return clf


def main():
    ap = argparse.ArgumentParser(description="Train the SIA severity classifier.")
    ap.add_argument("--data", default="./dataset/preprocessed_data.csv")
    ap.add_argument("--out", default="./Artifacts/severity_classifier.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w-text", type=float, default=0.6, help="weight on the text/embedding head")
    ap.add_argument("--w-meta", type=float, default=0.4, help="weight on the metadata head")
    args = ap.parse_args()

    train_and_eval(args.data, args.out, (args.w_text, args.w_meta), args.test_size, args.seed)


if __name__ == "__main__":
    main()
