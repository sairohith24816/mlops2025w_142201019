import wandb
import pandas as pd
import numpy as np
from datasets import load_dataset
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model import MajorityLabelVoter
from sklearn.metrics import accuracy_score

ABSTAIN, DATE_MISC, ORG = -1, 0, 1

@labeling_function()
def lf_year_detection(x):
    try:
        y = int(x.token)
        if 1900 <= y <= 2099:
            return DATE_MISC
    except:
        pass
    return ABSTAIN

@labeling_function()
def lf_org_suffix(x):
    return ORG if any(x.token.endswith(s) for s in ["Inc.", "Corp.", "Ltd."]) else ABSTAIN

wandb.init(project="Q3-majority-voter-conll2003", name="label-aggregation")

ds = load_dataset("eriktks/conll2003", revision="convert/parquet")["train"]
rows = []
for ex in ds:
    for token, tag in zip(ex["tokens"], ex["ner_tags"]):
        rows.append({
            "token": token,
            "true_label": DATE_MISC if tag in [7, 8] else ORG if tag in [3, 4] else ABSTAIN
        })
df = pd.DataFrame(rows)

lfs = [lf_year_detection, lf_org_suffix]
L = PandasLFApplier(lfs).apply(df)

aggregated = MajorityLabelVoter().predict(L)
total = len(aggregated)
abstained = np.sum(aggregated == -1)
labeled = total - abstained
coverage = labeled / total
print(f"\nTotal: {total} | Labeled: {labeled} | Abstained: {abstained} | Coverage: {coverage:.4f}")

if labeled == 0:
    print("No samples were labeled by any LF.")
    wandb.log({"total_samples": total, "labeled_samples": 0, "abstained_samples": abstained,
               "coverage": 0.0, "accuracy": 0.0})
    wandb.finish()
else:
    mask = aggregated != -1
    true, pred = df.true_label.values[mask], aggregated[mask]
    valid = true != ABSTAIN

    if np.sum(valid) == 0:
        print("No valid samples with ground truth labels found!")
        wandb.finish()
    else:
        acc = accuracy_score(true[valid], pred[valid])
        print(f"Accuracy on labeled samples: {acc:.4f}")

        wandb.log({"total_samples": total, "labeled_samples": labeled, "abstained_samples": abstained,
                   "coverage": coverage, "accuracy": acc})
        wandb.run.summary["majority_voter_coverage"] = coverage
        wandb.run.summary["majority_voter_accuracy"] = acc
        wandb.finish()
