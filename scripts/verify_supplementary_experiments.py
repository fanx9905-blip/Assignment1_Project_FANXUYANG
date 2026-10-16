"""Read-only verification: no fitting, downloads, or training."""
import ast
import contextlib
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import nbformat
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

root = Path(__file__).resolve().parents[1]
a = root / "artifacts"
read = lambda name: json.loads((a / name).read_text(encoding="utf-8"))
hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()}
for rel, digest in read("supplementary_original_hashes.json").items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
n = nbformat.read(root / "Assignment_code.ipynb", 4)
nbformat.validate(n)
by_id = {c.id: c for c in n.cells}
assert len(by_id) == len(n.cells)
for c in n.cells:
    if c.cell_type == "code":
        ast.parse(c.source)
        if c.id.startswith("ext-"):
            assert c.execution_count is not None
            assert not any(o.output_type == "error" for o in c.outputs)

data = read("part3_sentiment_data.json")
rows = data["rows"]
train, val, test = [np.array(data["split_indices"][s]) for s in ("train", "validation", "test")]
fit = np.sort(np.concatenate([train, val]))
labels = np.array([r["label"] for r in rows])
texts = [r["tokens"] for r in rows]
selection = read("part3_tfidf_selection.json")
best = min(selection["trials"], key=lambda r: (-r["validation_macro_f1"], r["C"]))
assert selection["selected_C"] == best["C"]

# Independently derive document frequencies and the IDF formula for both fit stages.
def verify_idf(indices, vocabulary, saved_idf):
    counts = Counter()
    for i in indices:
        tokens = texts[i]
        terms = set(tokens) | {" ".join(tokens[j:j+2]) for j in range(len(tokens)-1)}
        counts.update(terms)
    expected_terms = {t for t, count in counts.items() if count >= 2}
    assert len(expected_terms) < 50000
    assert set(vocabulary) == expected_terms
    expected = np.empty(len(vocabulary))
    for term, index in vocabulary.items():
        expected[index] = np.log((1 + len(indices)) / (1 + counts[term])) + 1
    assert np.allclose(expected, saved_idf, atol=1e-12)

verify_idf(train, selection["training_vocabulary"], selection["training_idf"])
state = np.load(a / "part3_tfidf_classifier.npz")
vocabulary = read("part3_tfidf_vocabulary.json")
verify_idf(fit, vocabulary, state["idf"])
vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=None, token_pattern=None,
                             lowercase=False, ngram_range=(1, 2), min_df=2,
                             max_features=50000, sublinear_tf=True, vocabulary=vocabulary)
vectorizer.idf_ = state["idf"]
X = vectorizer.transform([texts[i] for i in test])
prob = expit(np.asarray(X @ state["coefficients"].T).ravel() + state["intercept"][0])
pred = (prob > .5).astype(int)
records = read("part3_tfidf_test_predictions.json")
assert [r["id"] for r in records] == [rows[i]["id"] for i in test]
assert np.allclose(prob, [r["positive_probability"] for r in records], atol=1e-12)
assert np.array_equal(pred, [r["predicted_label"] for r in records])
result = read("part3_tfidf_results.json")
assert result["selected_C"] == best["C"]
assert abs(accuracy_score(labels[test], pred) - result["test_accuracy"]) < 1e-12
assert abs(f1_score(labels[test], pred, average="macro") - result["test_macro_f1"]) < 1e-12
assert confusion_matrix(labels[test], pred).tolist() == result["confusion_matrix"]
print("TF-IDF: training-only selection IDF, train+validation final IDF, numeric restoration and metrics passed.")

all_records = read("part3_test_predictions.json")
all_records["tfidf"] = records
stats = read("part3_uncertainty.json")
preds = {name: np.array([r["predicted_label"] for r in rec]) for name, rec in all_records.items()}
y = labels[test]
for rec in all_records.values():
    assert [r["id"] for r in rec] == [rows[i]["id"] for i in test]
    assert np.array_equal([r["label"] for r in rec], y)
rng = np.random.default_rng(4213)
indices = rng.integers(len(test), size=(10000, len(test)))
boot = {}
for name, p in preds.items():
    yt, pt = y[indices], p[indices]
    # Independently compute per-class precision/recall F1 through confusion counts.
    f1 = np.zeros(10000)
    for label in (0, 1):
        tp = ((yt == label) & (pt == label)).sum(axis=1)
        fp = ((yt != label) & (pt == label)).sum(axis=1)
        fn = ((yt == label) & (pt != label)).sum(axis=1)
        f1 += np.divide(2*tp, 2*tp+fp+fn, out=np.zeros(10000), where=(2*tp+fp+fn) != 0) / 2
    boot[name] = {"accuracy": (yt == pt).mean(axis=1), "macro_f1": f1}
    for metric, values in boot[name].items():
        assert np.allclose(np.quantile(values, [.025, .975]), stats["models"][name][metric+"_ci95"], atol=1e-12)
for pair in stats["pairs"]:
    aa, bb = pair["a"], pair["b"]
    ca, cb = preds[aa] == y, preds[bb] == y
    na, nb = int((ca & ~cb).sum()), int((~ca & cb).sum())
    assert [na, nb] == [pair["a_only_correct"], pair["b_only_correct"]]
    p = binomtest(na, na+nb, .5).pvalue if na+nb else 1.
    assert abs(p-pair["mcnemar_exact_p"]) < 1e-12
    for metric in ("accuracy", "macro_f1"):
        assert np.allclose(np.quantile(boot[bb][metric]-boot[aa][metric], [.025, .975]),
                           pair[metric+"_difference_ci95"], atol=1e-12)
order = sorted(stats["pairs"], key=lambda r: r["mcnemar_exact_p"])
adj = np.minimum(1, np.maximum.accumulate([r["mcnemar_exact_p"]*(6-i) for i, r in enumerate(order)]))
assert np.allclose(adj, [r["mcnemar_holm_p"] for r in order])
print("Statistics: paired 10,000-resample intervals, exact McNemar and six-comparison Holm adjustment passed.")

os.chdir(root)
ns = {"json": json, "hashlib": hashlib, "np": np, "torch": torch}
def execute(source):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, "<supplementary-read-only>", "exec"), ns)
def definition(cid, name):
    source = by_id[cid].source
    node = next(x for x in ast.parse(source).body if isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name == name)
    execute(ast.get_source_segment(source, node))
for cid in ("part1-load-code", "part1-clean-code", "part1-split-code", "part1-vocab-code", "part1-neural-batches-code"):
    execute(by_id[cid].source)
execute('import math, time\nARTIFACT_DIR = DATA_PATH.resolve().parent / "artifacts"')
execute(by_id["part1-training-api-code"].source)
definition("part1-lstm-model-code", "LSTMLanguageModel")
torch.set_num_threads(4)
ns["reset_neural_seed"](42)
model = ns["LSTMLanguageModel"](3620, 100, 128, .2)
initial = model.embedding.weight.detach().clone()
checkpoint = torch.load(a / "lstm_random_frozen_best.pt", map_location="cpu", weights_only=True)
assert checkpoint["id_to_token"] == ns["id_to_token"]
assert checkpoint["corpus_sha256"] == ns["corpus_sha256"]
model.load_state_dict(checkpoint["state_dict"])
model.embedding.weight.requires_grad_(False)
assert torch.equal(initial, model.embedding.weight.detach())
record = read("lstm_random_frozen_training.json")
res = read("lstm_random_frozen_results.json")
assert hashlib.sha256(initial.numpy().tobytes()).hexdigest() == res["initial_embedding_sha256"]
assert hashlib.sha256((a/"lstm_random_frozen_best.pt").read_bytes()).hexdigest() == res["checkpoint_sha256"]
assert record["best_epoch"] == min(record["history"], key=lambda r: r["validation"]["cross_entropy"])["epoch"]
base_config = read("lstm_training.json")["config"]
assert all(record["config"][k] == v for k, v in base_config.items())
loaders = ns["make_lm_loaders"](42, 8)
for split in ("validation", "test"):
    metrics = ns["run_lm_epoch"](model, loaders[split], torch.device("cpu"))
    assert abs(metrics["cross_entropy"] - res["metrics"][split]["cross_entropy"]) < 1e-5
    assert metrics["target_count"] == res["metrics"][split]["target_count"]
for cid, name in (("part1-ngram-generation-code", "sample_next_token"),
                  ("part1-rnn-generation-code", "generate_neural_text")):
    definition(cid, name)
cfg = read("generation_config.json")
ns.update(GENERATION_LENGTH=cfg["length"], GENERATION_TEMPERATURE=cfg["temperature"],
          GENERATION_TOP_K=cfg["top_k"], GENERATION_BLOCKED_IDS=tuple(cfg["blocked_token_ids"]))
model.eval()
for sample in read("lstm_random_frozen_generations.json")["samples"]:
    assert sample["generated_ids"] == ns["generate_neural_text"](model, sample["prompt"], sample["seed"], torch.device("cpu"))
print("Random-frozen LSTM: whole input table, config, checkpoint CE, target counts and generation reproduction passed.")
for p, digest in hashes.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, p
print("Read-only audit passed; no artifacts changed.")
