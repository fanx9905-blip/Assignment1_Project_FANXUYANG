"""Read-only audit of saved encoders, features, classifier selection and predictions."""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import numpy as np
import nbformat
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

folder=Path(__file__).resolve().parents[1]
a=folder/'artifacts'
protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()}
n=nbformat.read(folder/'Assignment_code.ipynb',4)
nbformat.validate(n)
by_id={c.id:c for c in n.cells}
assert len(by_id)==len(n.cells)
os.chdir(folder)
namespace={'torch':torch,'json':json,'hashlib':hashlib,'np':np}
def execute(source):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source,'<part3-read-only-check>','exec'),namespace)
def definition(cid,name):
    source=by_id[cid].source
    node=next(node for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
    execute(ast.get_source_segment(source,node))
for cid in ('part1-load-code','part1-clean-code','part1-split-code','part1-vocab-code'):
    execute(by_id[cid].source)
namespace['DEVICE']=torch.device('cpu')
torch.set_num_threads(4)
definition('part1-lstm-model-code','LSTMLanguageModel')
definition('part3-encoders-code','extract_part3_mean_states')
plan=json.loads((a/'part3_plan.json').read_text(encoding='utf-8'))
data=json.loads((a/'part3_sentiment_data.json').read_text(encoding='utf-8'))
results=json.loads((a/'part3_results.json').read_text(encoding='utf-8'))
selection=json.loads((a/'part3_classifier_selection.json').read_text(encoding='utf-8'))
predictions=json.loads((a/'part3_test_predictions.json').read_text(encoding='utf-8'))
checks=json.loads((a/'part3_feature_checks.json').read_text(encoding='utf-8'))
labels=np.array([row['label'] for row in data['rows']],dtype=np.int64)
indices={split:np.array(values) for split,values in data['split_indices'].items()}
trainval=np.sort(np.concatenate([indices['train'],indices['validation']]))
test=indices['test']
assert not set(trainval.tolist()) & set(test.tolist())
assert len(set(np.concatenate(list(indices.values())).tolist()))==len(data['rows'])==2979
for row in data['rows']:
    tokens=namespace['tokenize'](row['text'])
    assert tokens==row['tokens']
    key=json.dumps(tokens,ensure_ascii=False,separators=(',',':'))
    assert hashlib.sha256(key.encode('utf-8')).hexdigest()==row['id']
sequences=[[namespace['BOS_ID']]+[namespace['token_to_id'].get(t,namespace['UNK_ID']) for t in row['tokens']]+[namespace['EOS_ID']] for row in data['rows']]
for spec in plan['representations']:
    name=spec['name']
    assert hashlib.sha256((a/spec['checkpoint']).read_bytes()).hexdigest()==spec['checkpoint_sha256']
    checkpoint=torch.load(a/spec['checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['id_to_token']==namespace['id_to_token'] and checkpoint['corpus_sha256']==namespace['corpus_sha256']
    config=checkpoint['config']
    encoder=namespace['LSTMLanguageModel'](3620,100,128,config['dropout'])
    encoder.load_state_dict(checkpoint['state_dict'])
    encoder.requires_grad_(False).eval()
    saved=torch.load(a/f'part3_{name}_features.pt',map_location='cpu',weights_only=True)
    assert saved['row_ids']==[row['id'] for row in data['rows']]
    assert torch.equal(saved['labels'],torch.tensor(labels))
    matrix=saved['features'].numpy()
    assert matrix.shape==(2979,128) and np.isfinite(matrix).all()
    assert hashlib.sha256(matrix.tobytes()).hexdigest()==checks[name]['features_sha256']
    recomputed=namespace['extract_part3_mean_states'](encoder,sequences,batch_size=32).numpy()
    assert np.allclose(matrix,recomputed,atol=1e-6,rtol=1e-5)
    best=min(selection[name]['trials'],key=lambda row:(-row['validation_macro_f1'],row['C']))
    assert best['C']==selection[name]['selected_C']==results['rows'][name]['selected_C']
    with np.load(a/f'part3_{name}_classifier.npz',allow_pickle=False) as numeric:
        scaler=StandardScaler()
        scaler.mean_=numeric['scaler_mean'];scaler.scale_=numeric['scaler_scale'];scaler.var_=numeric['scaler_var']
        scaler.n_samples_seen_=len(trainval);scaler.n_features_in_=128
        assert np.allclose(scaler.mean_,matrix[trainval].mean(0,dtype=np.float64))
        assert np.allclose(scaler.var_,matrix[trainval].var(0,dtype=np.float64))
        classifier=LogisticRegression(C=best['C'],solver='lbfgs',max_iter=2000)
        classifier.coef_=numeric['coefficient'];classifier.intercept_=numeric['intercept'];classifier.classes_=numeric['classes']
        classifier.n_features_in_=128
        pipeline=Pipeline([('scaler',scaler),('classifier',classifier)])
    predicted=pipeline.predict(matrix[test])
    probabilities=pipeline.predict_proba(matrix[test])[:,1]
    assert np.array_equal(predicted,[row['predicted_label'] for row in predictions[name]])
    assert np.allclose(probabilities,[row['positive_probability'] for row in predictions[name]],atol=1e-10)
    assert [row['id'] for row in predictions[name]]==[data['rows'][i]['id'] for i in test]
    metrics=results['rows'][name]
    assert math.isclose(accuracy_score(labels[test],predicted),metrics['test_accuracy'],abs_tol=1e-12)
    assert math.isclose(f1_score(labels[test],predicted,average='macro'),metrics['test_macro_f1'],abs_tol=1e-12)
    assert confusion_matrix(labels[test],predicted,labels=[0,1]).tolist()==metrics['confusion_matrix']
    assert metrics['refit_samples']==2383 and metrics['test_samples']==596
    print(name,': encoder features, validation selection, train+val scaling, numeric classifier and test metrics verified.')
for cid in ('part3-encoders-code','part3-features-code','part3-selection-code','part3-evaluation-code','part3-errors-code','part3-comparison-code'):
    cell=by_id[cid]
    assert cell.execution_count is not None and cell.outputs
    assert not any(o.output_type=='error' for o in cell.outputs)
assert any('image/png' in o.get('data',{}) for o in by_id['part3-errors-code'].outputs)
for p,digest in protected.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,p
print('Part III read-only audit passed; no encoder/classifier training or artifact changes.')
