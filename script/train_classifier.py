from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib


class SeverityClassifier:
    SEVERITY_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    META_FEATURES = ["Resolution_Time_Hours", "Issue_Category", "Satisfaction_Score", "Ticket_Channel"]
    SCALE_COLS = ["Resolution_Time_Hours", "Satisfaction_Score"]

    def __init__(self, weights=(0.6, 0.4), encoder_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.weights = weights
        self.encoder = SentenceTransformer(encoder_name)
        self.scaler = StandardScaler()
        self.embd_model = XGBClassifier()
        self.meta_model = XGBClassifier()
        self._meta_columns = None 

    def _get_embedding(self, row):
        return self.encoder.encode(f"{row['Ticket_Subject']}, {row['Ticket_Description']}")

    def _build_embeddings(self, df):
        print("ENCODING...")
        embeddings = [self._get_embedding(row) for _, row in df.iterrows()]
        return pd.DataFrame(embeddings, index=df.index)

    def _build_meta(self, df):
        meta = df[self.META_FEATURES].copy()
        return pd.get_dummies(meta, columns=["Issue_Category", "Ticket_Channel"], drop_first=True, dtype=int)

    def _align_meta_columns(self, meta):
        """Ensure test meta has the same columns as training meta (fills missing with 0)."""
        return meta.reindex(columns=self._meta_columns, fill_value=0)

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df["Final_severity_label"] = df["Final_severity_label"].map(self.SEVERITY_MAP)
        df["Priority_Level"] = df["Priority_Level"].map(self.PRIORITY_MAP)
        return df

    def fit(self, df):
        train_idx, test_idx = train_test_split(
            df.index, test_size=0.2, stratify=df["Final_severity_label"]
        )

        embd_X = self._build_embeddings(df)
        meta_X = self._build_meta(df)

        embd_X_train = embd_X.loc[train_idx]
        meta_X_train = meta_X.loc[train_idx].copy()
        meta_X_train[self.SCALE_COLS] = self.scaler.fit_transform(meta_X_train[self.SCALE_COLS])
        self._meta_columns = meta_X_train.columns.tolist()

        y_train = df["Final_severity_label"].loc[train_idx]

        embd_X_test = embd_X.loc[test_idx]
        meta_X_test = meta_X.loc[test_idx].copy()
        meta_X_test[self.SCALE_COLS] = self.scaler.transform(meta_X_test[self.SCALE_COLS])
        meta_X_test = self._align_meta_columns(meta_X_test)

        y_test = df["Final_severity_label"].loc[test_idx]
        priority_level_test = df["Priority_Level"].loc[test_idx]
        y_true_mismatch = df["Is_Mismatch"].loc[test_idx]

        print("TRAINING")
        self.embd_model.fit(embd_X_train, y_train)
        self.meta_model.fit(meta_X_train, y_train)

        print("EVALUATING")
        y_pred = self._predict_from_features(embd_X_test, meta_X_test)
        y_pred_mismatch = (priority_level_test.values != y_pred).astype(int)

        report = classification_report(y_true_mismatch, y_pred_mismatch)
        print(report)
        return report

    def predict(self, df):
        embd_X = self._build_embeddings(df)
        meta_X = self._build_meta(df)
        meta_X[self.SCALE_COLS] = self.scaler.transform(meta_X[self.SCALE_COLS])
        meta_X = self._align_meta_columns(meta_X)
        return self._predict_from_features(embd_X, meta_X)

    def _predict_from_features(self, embd_X, meta_X):
        embd_prob = self.embd_model.predict_proba(embd_X)
        meta_prob = self.meta_model.predict_proba(meta_X)
        final_prob = self.weights[0] * embd_prob + self.weights[1] * meta_prob
        return np.argmax(final_prob, axis=1)

    def save(self, path="./Artifacts/severity_classifier.joblib"):
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load(path="./Artifacts/severity_classifier.joblib"):
        return joblib.load(path)

if __name__ == "__main__":
    clf = SeverityClassifier(weights=(0.6, 0.4))
    df = clf.load_data("./dataset/preprocessed_data.csv")
    clf.fit(df)
    clf.save()