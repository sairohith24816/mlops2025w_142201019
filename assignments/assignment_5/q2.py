import wandb
import pandas as pd
import numpy as np
from datasets import load_dataset
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling import LFAnalysis
from sklearn.metrics import accuracy_score

ABSTAIN, DATE_MISC, ORG = -1, 0, 1

@labeling_function()
def lf_year_detection(x):
    try:
        y = int(x.token)
        if 1900 <= y <= 2099:
            return DATE_MISC
    except:
        # pass
        return ABSTAIN
    return ABSTAIN

@labeling_function()
def lf_org_suffix(x):
    return ORG if any(x.token.endswith(s) for s in ["Inc.", "Corp.", "Ltd."]) else ABSTAIN


wandb.init(project="Q2-weak-supervision-ner-conll2003", name="lf_analysis")

ds = load_dataset("eriktks/conll2003", revision="convert/parquet")["train"]
rows = []
for ex in ds:
    for token, tag in zip(ex["tokens"], ex["ner_tags"]):
        rows.append({
            "token": token,
            "true_label": DATE_MISC if tag in [7, 8] else ORG if tag in [3, 4] else ABSTAIN
        })
df = pd.DataFrame(rows)
print(f"Created dataset with {len(df)} tokens")

lfs = [lf_year_detection, lf_org_suffix]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df)
print(f"Applied {len(lfs)} labeling functions")

analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary(df['true_label'].values)

print("\nLabeling Function Analysis:")
print(analysis)

for lf in lfs:
    coverage = analysis.loc[lf.name, "Coverage"]
    accuracy = analysis.loc[lf.name, "Emp. Acc."]
    
    print(f"\n--- {lf.name} ---")
    print(f"Coverage: {coverage:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    wandb.log({
        f"{lf.name}_coverage": coverage,
        f"{lf.name}_accuracy": accuracy
    })

    wandb.run.summary[f"{lf.name}_coverage"] = coverage
    wandb.run.summary[f"{lf.name}_accuracy"] = accuracy

print("\nLabeling function evaluation completed!")
wandb.finish()
