# Support Integrity Auditor (SIA)

This project finds **priority mismatches** in customer support tickets. A
mismatch is when the priority a human put on a ticket doesn't match how serious
the ticket actually is based on its text, category, channel and how long it took
to resolve.

The tricky part is the dataset has no mismatch labels. So we have to build our own
training signal first (self-supervised), then train a classifier on it.

There are two types of mismatch we care about:

- **Hidden Crisis**: looks low priority but is actually serious. These are the
  dangerous ones (SLA breaches, angry customers).
- **False Alarm**: marked urgent but isn't really.

For every ticket we flag, the system also writes an **Evidence Dossier** that
explains *why* it was flagged, using only facts that are actually in the ticket
(no made up reasons).

## Pipeline

Three stages:

```
Stage 1  ->  Stage 2  ->  Stage 3
labels       classifier   dossier
```

### Stage 1: pseudo-labels (self-supervised)

We figure out a "true" severity for each ticket without ever looking at the
assigned priority. Three independent signals get combined:

```
final_score = 0.30*LLM + 0.35*embedding + 0.35*(category+resolution)
inferred_severity = argmax(final_score)
Is_Mismatch = (priority != inferred_severity)
```

1. **LLM signal** - Gemma-2-2B (4-bit) reads the subject + description and scores
   LOW/MEDIUM/HIGH/CRITICAL. Output is constrained to just those 4 tokens.
2. **Embedding signal** - cosine similarity between the ticket and a few anchor
   sentences per severity, using MiniLM (`all-MiniLM-L6-v2`).
3. **Category + resolution signal** - a lookup table over `Issue_Category` and
   `Resolution_Time_Hours`. Resolution time is an indirect hint about severity.

Code is in `script/pseudo_label_generation.py`.

**Why three signals?** None of them is reliable alone, but they make different
mistakes, so combining them is much steadier. The table below shows how often each
signal on its own agrees with the final fused label, and how much the two
non-LLM signals agree with each other (this is the pairwise agreement metric the
spec asks for). Measured on a 1,500 ticket sample:

| Signal (used alone) | Agreement with fused label |
|---|---|
| Embedding (MiniLM) | 0.776 |
| Category + resolution | 0.783 |
| Embedding vs Category/resolution (pairwise) | 0.626 |

The two signals only agree about 63% of the time, which is the point - they're
independent and each catches things the other misses. One reads the text, the
other reads the metadata. To reproduce the full 3-signal version (with the LLM),
set `REGENERATE_PSEUDO_LABELS = True` in `notebook.ipynb` (needs a GPU).

### Stage 2: classifier

One thing I learned the hard way: don't predict the mismatch directly. If you try
to predict `Is_Mismatch` straight from the features you get stuck around 72%
because the model has to learn a priority-vs-severity interaction. The fix is to
**predict severity instead**, then just compare it to the assigned priority. The
comparison is exact so it adds no error.

`SeverityClassifier` (`script/train_classifier.py`) has two parts:

- text head: MiniLM embedding of subject + description -> XGBoost
- metadata head: one-hot category + channel, scaled resolution time + satisfaction
  -> XGBoost

and combines them as `0.6*text + 0.4*metadata`. So it uses both text and
structured metadata, like the spec requires. The class imbalance mostly takes care
of itself here, because the 4-class severity target is way more balanced than the
binary mismatch target.

### Stage 3: evidence dossier

`EvidenceDossierGenerator` (`script/evidence_dossier_generation.py`) builds the
dossier for each flagged ticket. It's fully rule based, no generative model, so it
literally can't hallucinate. A keyword only shows up as evidence if it actually
appears in the ticket text, and everything else is copied directly from a field.

Example output:

```json
{
  "ticket_id": "TKT-ADV-001",
  "assigned_priority": "Low",
  "inferred_severity": "Critical",
  "mismatch_type": "Hidden Crisis",
  "severity_delta": 3,
  "feature_evidence": [
    { "signal": "keyword", "value": "unauthorized", "weight": "aligns with inferred severity" },
    { "signal": "resolution_time", "value": "6 hours", "interpretation": "resolved in under 24h ..." }
  ],
  "constraint_analysis": "...2-3 grounded sentences...",
  "confidence": 0.71
}
```

## Results

Tested on a held-out 20% split (`random_state=42`), binary mismatch task. The first
few rows are approaches that didn't work, kept here so the comparison is clear.

| Approach | Accuracy | Macro F1 | Recall (Consistent) | Recall (Mismatch) | Pass |
|---|---|---|---|---|---|
| DistilBERT, text only | 0.9667| 0.9633 | 0.9341	 | 0.9845 |  yes But Heavy Model |
| TF-IDF + meta + priority -> LogReg (direct) | 0.713 | 0.697 | 0.696 | 0.722 | no |
| MiniLM + meta + priority -> LogReg (direct) | 0.723 | 0.707 | 0.706 | 0.732 | no |
| Decompose -> severity -> compare (LogReg) | 0.953 | 0.948 | 0.927 | 0.967 | yes |
| **Decompose -> severity -> compare (XGBoost)** | **0.959** | **0.955** | **0.937** | **0.971** | **yes** |

Required thresholds: accuracy >= 83%, macro F1 >= 0.82, recall >= 0.78 on both
classes. The shipped model (last row) clears all of them with room to spare.
`train_pipeline.py` prints these numbers and a PASS/FAIL line when it finishes.

## Files

```
script/
  pseudo_label_generation.py     stage 1
  train_classifier.py            stage 2 (SeverityClassifier)
  evidence_dossier_generation.py stage 3
train_pipeline.py                train + check thresholds
predict.py                       run on a CSV -> predictions + dossiers
main.py                          streamlit app
notebook.ipynb                   whole pipeline start to finish
dataset/                         csv files
Artifacts/                       trained model
```

## How to run

```bash
pip install -r requirements.txt

# train (writes Artifacts/severity_classifier.joblib)
python train_pipeline.py

# predict on a csv -> predictions.csv + dossiers.json
python predict.py --input my_tickets.csv

# web app
streamlit run main.py
```

The streamlit app does single tickets or a batch CSV, shows the judgment + dossier
for each, and has a dashboard with the mismatch breakdown and a severity-delta
heatmap across categories and channels.

Your input CSV needs these columns: `Ticket_Subject`, `Ticket_Description`,
`Issue_Category`, `Priority_Level`, `Ticket_Channel`, `Resolution_Time_Hours`,
`Satisfaction_Score`. `Ticket_ID` is optional.

## Dataset

Customer Support Tickets - CRM Dataset (from Kaggle, ajverse). About 20k tickets.
After stage 1, around 65% of them end up flagged as mismatches.
