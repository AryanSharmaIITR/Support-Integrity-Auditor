import json
import re

class EvidenceDossierGenerator:
    SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    RANK_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
    SEVERITY_KEYWORDS = {
        "CRITICAL": [
            "outage", "data loss", "data breach", "breach", "hacked",
            "security breach", "unauthorized", "unauthorised", "fraud",
            "fraudulent", "stolen", "locked out", "lock my account",
            "system down", "service unavailable", "server error",
            "500 internal server error", "corrupted", "emergency",
            "immediately", "asap", "right now", "lawsuit", "legal action",
            "compromised", "ransom",
        ],
        "HIGH": [
            "cannot log in", "can't log in", "cannot login", "login broken",
            "payment fail", "payment failed", "payment processing",
            "checkout failing", "2fa", "authentication error", "auth error",
            "broken", "not working", "unable to", "crashes", "crash",
            "error", "failing", "declined", "double charged", "overcharged",
        ],
        "MEDIUM": [
            "slow", "delay", "delayed", "intermittent", "sometimes",
            "glitch", "minor bug", "workaround", "not loading", "lag",
        ],
        "LOW": [
            "how do i", "how to", "question", "feature request",
            "documentation", "typo", "cosmetic", "general inquiry",
            "where is", "can you add", "would like", "information about",
        ],
    }
    RESOLUTION_BANDS = [
        (0, 24, "resolved in under 24h — fast turnaround consistent with high urgency / triage escalation"),
        (24, 48, "resolved within 24-48h — moderate handling time"),
        (48, 72, "took 48-72h to resolve — slower handling, weaker urgency signal"),
        (72, float("inf"), "took 72h+ to resolve — protracted handling typical of lower-urgency tickets"),
    ]
    CATEGORY_NOTE = {
        "Fraud": "Fraud-category tickets carry an elevated baseline severity (account/financial risk)",
        "Billing": "Billing-category tickets often map to High severity (payment/charge impact)",
        "Technical": "Technical-category tickets skew toward Medium/High depending on the failure",
        "Account": "Account-category tickets vary; access-loss raises severity",
        "General Inquiry": "General-Inquiry tickets carry a low baseline severity (informational)",
    }


    FIELD_ALIASES = {
        "ticket_id": ["Ticket_ID", "Ticket ID", "ticket_id"],
        "subject": ["Ticket_Subject", "Ticket Subject", "subject"],
        "description": ["Ticket_Description", "Ticket Description", "description"],
        "category": ["Issue_Category", "Ticket_Type", "Ticket Type", "category"],
        "priority": ["Priority_Level", "Ticket_Priority", "Ticket Priority", "priority"],
        "channel": ["Ticket_Channel", "Ticket Channel", "channel"],
        "resolution_time": ["Resolution_Time_Hours", "Resolution Time", "resolution_time"],
        "satisfaction": ["Satisfaction_Score", "Customer Satisfaction Rating", "satisfaction"],
        "inferred_severity": ["Final_severity_label", "inferred_severity"],
    }

    def __init__(self, max_keyword_evidence=4):
        self.max_keyword_evidence = max_keyword_evidence

    def _get(self, ticket, logical_name, default=None):
        """Fetch a logical field from a dict / pandas Series, trying each alias."""
        for key in self.FIELD_ALIASES.get(logical_name, [logical_name]):
            if key in ticket and ticket[key] is not None:
                value = ticket[key]
                if isinstance(value, float) and value != value:
                    continue
                return value
            
        return default

    @staticmethod
    def _norm_severity(label):
        return str(label).strip().upper() if label is not None else None

    def _find_keywords(self, text, inferred_severity):
        if not text:
            return []
        haystack = text.lower()
        matches = []
        for tier, phrases in self.SEVERITY_KEYWORDS.items():
            for phrase in phrases:
                if re.search(r"\b" + re.escape(phrase) + r"\b", haystack):
                    matches.append((tier, phrase))

        if not matches:
            return []

        target = self._norm_severity(inferred_severity)
        matches.sort(key=lambda m: (m[0] != target, -self.SEVERITY_RANK[m[0]]))

        evidence = []
        for tier, phrase in matches[: self.max_keyword_evidence]:
            weight = (
                "aligns with inferred severity"
                if tier == target
                else f"tier {tier.lower()}"
            )
            evidence.append(
                {"signal": "keyword", "value": phrase, "weight": weight}
            )
        return evidence

    def _resolution_interpretation(self, hours):
        try:
            hrs = float(hours)
        except (TypeError, ValueError):
            return None
        for low, high, note in self.RESOLUTION_BANDS:
            if low <= hrs < high:
                return note
        return "resolution time outside expected range"

    def _confidence(self, scores, inferred_severity, delta):
        sev = self._norm_severity(inferred_severity)
        if scores:
            norm = {self._norm_severity(k): float(v) for k, v in scores.items()}
            total = sum(norm.values()) or 1.0
            if sev in norm:
                return round(norm[sev] / total, 3)

        return {1: 0.6, 2: 0.78, 3: 0.9}.get(abs(int(delta)), 0.5)

    def generate(self, ticket, inferred_severity=None, scores=None):
        assigned_priority = self._get(ticket, "priority")
        if inferred_severity is None:
            inferred_severity = self._get(ticket, "inferred_severity")

        prio = self._norm_severity(assigned_priority)
        sev = self._norm_severity(inferred_severity)
        if prio not in self.SEVERITY_RANK or sev not in self.SEVERITY_RANK:
            raise ValueError(
                f"Unknown priority/severity: priority={assigned_priority!r}, "
                f"severity={inferred_severity!r}"
            )

        delta = self.SEVERITY_RANK[sev] - self.SEVERITY_RANK[prio]
        if delta == 0:
            return None  # Consistent ticket — no dossier required.

        mismatch_type = "Hidden Crisis" if delta > 0 else "False Alarm"

        subject = self._get(ticket, "subject", "")
        description = self._get(ticket, "description", "")
        text = f"{subject} {description}".strip()

        feature_evidence = []
        feature_evidence.extend(self._find_keywords(text, sev))

        res_hours = self._get(ticket, "resolution_time")
        res_note = self._resolution_interpretation(res_hours)
        if res_note is not None:
            feature_evidence.append(
                {
                    "signal": "resolution_time",
                    "value": f"{res_hours} hours",
                    "interpretation": res_note,
                }
            )

        category = self._get(ticket, "category")
        if category is not None:
            feature_evidence.append(
                {
                    "signal": "category",
                    "value": str(category),
                    "interpretation": self.CATEGORY_NOTE.get(
                        str(category), f"'{category}' category context"
                    ),
                }
            )

        channel = self._get(ticket, "channel")
        if channel is not None:
            feature_evidence.append(
                {
                    "signal": "channel",
                    "value": str(channel),
                    "interpretation": f"intake channel recorded as {channel}",
                }
            )

        satisfaction = self._get(ticket, "satisfaction")
        if satisfaction is not None:
            feature_evidence.append(
                {
                    "signal": "satisfaction_score",
                    "value": str(satisfaction),
                    "interpretation": "customer-reported satisfaction (lower implies unresolved frustration)",
                }
            )

        confidence = self._confidence(scores, sev, delta)

        kw_phrases = [e["value"] for e in feature_evidence if e["signal"] == "keyword"]
        kw_clause = (
            f"Text evidence such as {', '.join(repr(k) for k in kw_phrases)} "
            if kw_phrases
            else "The ticket text and metadata "
        )
        if mismatch_type == "Hidden Crisis":
            analysis = (
                f"The ticket was assigned '{assigned_priority}' priority but its objective "
                f"characteristics point to a higher '{sev.title()}' severity (delta +{delta}). "
                f"{kw_clause}together with a resolution time of {res_hours}h and the "
                f"'{category}' category indicate the issue is more urgent than its label — a "
                f"potential SLA risk that was under-prioritised."
            )
        else:
            analysis = (
                f"The ticket was assigned '{assigned_priority}' priority yet its objective "
                f"characteristics suggest a lower '{sev.title()}' severity (delta {delta}). "
                f"{kw_clause}together with a resolution time of {res_hours}h and the "
                f"'{category}' category indicate the ticket was inflated above its true urgency."
            )

        return {
            "ticket_id": self._get(ticket, "ticket_id", "UNKNOWN"),
            "assigned_priority": str(assigned_priority),
            "inferred_severity": sev.title(),
            "mismatch_type": mismatch_type,
            "severity_delta": int(delta),
            "feature_evidence": feature_evidence,
            "constraint_analysis": analysis,
            "confidence": confidence,
        }

    def generate_batch(self, df, inferred_col="Final_severity_label", score_cols=None):
        score_cols = score_cols or {
            "LOW": "low_score",
            "MEDIUM": "medium_score",
            "HIGH": "high_score",
            "CRITICAL": "critical_score",
        }
        have_scores = all(c in df.columns for c in score_cols.values())
        dossiers = []
        for _, row in df.iterrows():
            inferred = row[inferred_col] if inferred_col in row.index else None
            scores = {k: row[c] for k, c in score_cols.items()} if have_scores else None
            dossier = self.generate(row, inferred_severity=inferred, scores=scores)
            if dossier is not None:
                dossiers.append(dossier)
        
        return dossiers

    @staticmethod
    def to_json(dossier, indent=2):
        return json.dumps(dossier, indent=indent, ensure_ascii=False)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("./dataset/preprocessed_data.csv")
    flagged = df[df["Is_Mismatch"] == 1].head(3)

    gen = EvidenceDossierGenerator()
    for dossier in gen.generate_batch(flagged):
        print(gen.to_json(dossier))
        print("-" * 60)
