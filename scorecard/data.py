"""Load and clean the UCI 'Default of Credit Card Clients' data (Taiwan, Apr-Sep 2005)."""
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "default"
# Excluded from the model on purpose: under EU anti-discrimination rules (e.g. the
# Gender Directive 2004/113/EC) sex and marital status should not drive credit decisions.
# SEX is kept only to run the fairness check after scoring.
PROTECTED = ["SEX", "MARRIAGE"]

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]   # most recent first
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    return df.rename(columns={"default payment next month": TARGET})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix undocumented codes and add behavioural features. Returns a new frame."""
    out = df.copy()
    # EDUCATION 0/5/6 and MARRIAGE 0 are not in the dataset documentation -> "other"
    out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"].isin([1, 2, 3]), 4)
    out["MARRIAGE"] = out["MARRIAGE"].where(out["MARRIAGE"].isin([1, 2]), 3)

    # Repayment status: -2 no consumption, -1 paid in full, 0 revolving credit, 1..8 months late
    delays = out[PAY_STATUS].clip(lower=0)
    out["max_delay_6m"] = delays.max(axis=1)
    out["months_delayed_6m"] = (delays > 0).sum(axis=1)
    out["recent_delay"] = delays["PAY_0"]

    bills = out[BILLS].clip(lower=0)   # negative bill = credit balance; treat as zero owed
    out["utilisation_last"] = (bills["BILL_AMT1"] / out["LIMIT_BAL"]).clip(upper=2)
    out["utilisation_avg_6m"] = (bills.mean(axis=1) / out["LIMIT_BAL"]).clip(upper=2)

    # Share of the previous month's bill actually paid (PAY_AMTi pays BILL_AMT(i+1))
    paid = out[PAYMENTS[:-1]].to_numpy()
    owed = bills[BILLS[1:]].to_numpy()
    ratio = np.where(owed > 0, np.minimum(paid / np.where(owed > 0, owed, 1), 1.5), np.nan)
    # months with nothing owed are skipped; a client who owed nothing all 5 months counts as fully paid
    out["payment_ratio_avg_5m"] = pd.DataFrame(ratio, index=out.index).mean(axis=1, skipna=True).fillna(1.0)
    return out


# EDUCATION was tested and left out: weak (IV 0.035), and its best-scoring group is the
# "other" bucket built from undocumented codes 0/5/6, i.e. a data artefact, not a risk signal.
CANDIDATE_FEATURES = [
    "LIMIT_BAL", "AGE", "PAY_0", "max_delay_6m", "months_delayed_6m",
    "utilisation_last", "utilisation_avg_6m", "payment_ratio_avg_5m", "PAY_AMT1", "BILL_AMT1",
]
CATEGORICAL = ["PAY_0"]
