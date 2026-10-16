"""Audit sequential notebook dependencies in an isolated copy, without LM training.

This does not certify a full from-scratch training run. The only substitutions
restore saved LM/Skip-gram weights; all other notebook cells run in order.
"""
import ast
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
work = root / ".execution_audit" / datetime.now().strftime("%Y%m%d_%H%M%S")
work.mkdir(parents=True, exist_ok=False)
nb = nbformat.read(root / "Assignment_code.ipynb", 4)
original_hash = hashlib.sha256((root / "Assignment_code.ipynb").read_bytes()).hexdigest()
protected = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (root/"artifacts").iterdir() if p.is_file()}
for directory in ("artifacts", "external_embeddings", "downstream_data"):
    shutil.copytree(root/directory, work/directory)
shutil.copy2(root/"pride_and_prejudice_gutenberg_1342.txt", work)
script_refs = []
for c in nb.cells:
    if c.cell_type == "code":
        ast.parse(c.source)
        for node in ast.walk(ast.parse(c.source)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and ".py" in node.value:
                script_refs.append({"cell": c.id, "literal": node.value})
        c.execution_count = None
        c.outputs = []
restore = '''
def train_language_model(model, model_name, config, loaders, device=DEVICE):
    checkpoint = torch.load(ARTIFACT_DIR / f"{model_name}_best.pt", map_location=device, weights_only=True)
    record = json.loads((ARTIFACT_DIR / f"{model_name}_training.json").read_text(encoding="utf-8"))
    assert checkpoint["config"] == config, model_name
    assert checkpoint["id_to_token"] == id_to_token
    assert checkpoint["corpus_sha256"] == corpus_sha256
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    print("AUDIT: restored existing", model_name, "instead of training.")
    return model, record
'''
for c in nb.cells:
    if c.id == "part1-training-api-code":
        c.source += "\n" + restore
    if c.id == "part2-skipgram-training-code":
        c.source = '''
sg_record = json.loads((ARTIFACT_DIR / "skipgram_training.json").read_text(encoding="utf-8"))
sg_input_vectors = torch.load(ARTIFACT_DIR / "skipgram_input_vectors.pt", map_location="cpu", weights_only=True)["input_vectors"]
print("AUDIT: restored Skip-gram vectors instead of training.")
'''
os.environ["IPYTHONDIR"] = str(work / ".ipython")
completed = []
class Client(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        return super().process_message(msg, cell, cell_index)
client = Client(nb, timeout=600, kernel_name="python3", resources={"metadata":{"path":str(work)}})
report = {"scope":"Sequential isolated replay with saved LM/Skip-gram substitution, NOT full retraining",
          "original_notebook_sha256":original_hash, "workspace":str(work),
          "python_script_literals":script_refs, "completed_cells":completed}
try:
    with client.setup_kernel():
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code":
                print("AUDIT executing", c.id, flush=True)
                client.execute_cell(c, i)
                completed.append(c.id)
    report["status"] = "passed"
except Exception as e:
    report.update(status="failed", failed_cell=c.id, exception=str(e)[-5000:])
    print("AUDIT FAILED:", c.id, str(e)[-1600:], flush=True)
finally:
    nbformat.write(nb, work/"audit_executed.ipynb")
    assert hashlib.sha256((root/"Assignment_code.ipynb").read_bytes()).hexdigest()==original_hash
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in protected.items())
    report["original_files_unchanged"] = True
    (work/"notebook_execution_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("AUDIT", report["status"], "completed code cells:", len(completed), flush=True)
