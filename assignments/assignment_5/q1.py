from datasets import load_dataset
from collections import Counter
import wandb

conll = load_dataset("eriktks/conll2003", revision="convert/parquet")

total_train = len(conll["train"])
total_val = len(conll["validation"])
total_test = len(conll["test"])


def entity_distribution(split):
    # Map integer entity indices to string labels
    entity_map = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    idx_to_label = {v: k for k, v in entity_map.items()}
    all_entities = []
    for example in conll[split]:
        all_entities.extend(example['ner_tags'])
    label_entities = [idx_to_label[idx] for idx in all_entities]
    count_entity = Counter([label.split('-')[-1] for label in label_entities if label != 'O'])
    return count_entity

train_entity_dist = entity_distribution("train")
val_entity_dist = entity_distribution("validation")
test_entity_dist = entity_distribution("test")

wandb.init(project="Q1-weak-supervision-ner")

# Log the dataset statistics 
wandb.summary["total_train_samples"] = total_train
wandb.summary["total_val_samples"] = total_val
wandb.summary["total_test_samples"] = total_test
wandb.summary["train_entity_distribution"] = dict(train_entity_dist)
wandb.summary["val_entity_distribution"] = dict(val_entity_dist)
wandb.summary["test_entity_distribution"] = dict(test_entity_dist)
