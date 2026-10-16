"""Read-only verification of the completed three-way LSTM embedding experiment."""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
import nbformat

folder=Path(__file__).resolve().parents[1]
artifacts=folder/'artifacts'
protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}
n=nbformat.read(folder/'Assignment_code.ipynb',4)
nbformat.validate(n)
by_id={c.id:c for c in n.cells}
assert len(by_id)==len(n.cells)
namespace={'json':json,'math':math,'time':time,'hashlib':hashlib}
os.chdir(folder)
def execute(source):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source,'<part2-read-only-audit>','exec'),namespace)
def definition(cid,name):
    src=by_id[cid].source
    node=next(node for node in ast.parse(src).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
    execute(ast.get_source_segment(src,node))
for cid in ('part1-load-code','part1-clean-code','part1-split-code','part1-vocab-code','part1-neural-batches-code'):
    execute(by_id[cid].source)
namespace['ARTIFACT_DIR']=artifacts
execute(by_id['part1-training-api-code'].source)
definition('part1-lstm-model-code','LSTMLanguageModel')
plan=json.loads((artifacts/'part2_plan.json').read_text(encoding='utf-8'))
namespace['PART2_LM_CONFIG']=plan['common_lm_config']
torch=namespace['torch']
torch.set_num_threads(4)
definition('part2-freeze-code','make_part2_fallback_table')
definition('part2-freeze-code','make_part2_lstm')
definition('part1-rnn-generation-code','generate_neural_text')
definition('part1-ngram-generation-code','sample_next_token')
generation_config=json.loads((artifacts/'generation_config.json').read_text(encoding='utf-8'))
namespace.update({'GENERATION_LENGTH':generation_config['length'],'GENERATION_TEMPERATURE':generation_config['temperature'],
                  'GENERATION_TOP_K':generation_config['top_k'],'GENERATION_BLOCKED_IDS':tuple(generation_config['blocked_token_ids'])})
assert plan['corpus_sha256']==namespace['corpus_sha256']
assert plan['vocabulary_sha256']==hashlib.sha256(json.dumps(namespace['id_to_token'],ensure_ascii=False).encode('utf-8')).hexdigest()
tables={}
for label in ('self','glove'):
    snapshot=torch.load(artifacts/f'part2_{label}_table.pt',map_location='cpu',weights_only=True)
    assert snapshot['id_to_token']==namespace['id_to_token'] and snapshot['corpus_sha256']==namespace['corpus_sha256']
    table=snapshot['table']
    assert tuple(table.shape)==(3620,100) and torch.isfinite(table).all()
    assert not torch.count_nonzero(table[0])
    matched=set(snapshot['matched_ids'])
    missing=sorted(set(range(3620))-matched)
    assert torch.equal(table[missing],namespace['make_part2_fallback_table']()[missing])
    coverage=json.loads((artifacts/f'part2_{label}_coverage.json').read_text(encoding='utf-8'))
    assert len(matched)==coverage['matched_types']
    words={namespace['id_to_token'][i] for i in matched}
    for split,paragraphs in namespace['split_paragraphs'].items():
        count=sum(word in words for p in paragraphs for word in p)
        assert count==coverage['token_coverage'][split]['matched_body_occurrences']
    tables[label]=table
assert torch.equal(tables['self'][:4],tables['glove'][:4])
init_self=namespace['make_part2_lstm'](tables['self'],True)
init_glove=namespace['make_part2_lstm'](tables['glove'],True)
for name,value in init_self.named_parameters():
    if name!='embedding.weight':
        assert torch.equal(value,dict(init_glove.named_parameters())[name])
del init_self,init_glove
loaders=namespace['make_lm_loaders'](42,8)
for kind,label in (('lstm','random'),('lstm_self_frozen','self'),('lstm_glove_frozen','glove')):
    result=json.loads((artifacts/f'{kind}_results.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(artifacts/f'{kind}_best.pt',map_location=namespace['DEVICE'],weights_only=True)
    assert checkpoint['id_to_token']==namespace['id_to_token'] and checkpoint['corpus_sha256']==namespace['corpus_sha256']
    assert checkpoint['config']==result['config']
    for key,value in plan['common_lm_config'].items():
        assert checkpoint['config'][key]==value
    assert checkpoint['epoch']==result['best_epoch']==min(result['history'],key=lambda row:row['validation']['cross_entropy'])['epoch']
    model=namespace['make_part2_lstm']() if label=='random' else namespace['make_part2_lstm'](tables[label],True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if label!='random':
        assert not model.embedding.weight.requires_grad
        assert torch.equal(model.embedding.weight.detach().cpu(),tables[label])
        assert result['trainable_parameters']==584740 and result['full_input_table_unchanged_after_training']
        assert checkpoint['config']['aligned_embedding_sha256']==hashlib.sha256(tables[label].numpy().tobytes()).hexdigest()
    for split in ('validation','test'):
        metrics=namespace['run_lm_epoch'](model,loaders[split],namespace['DEVICE'])
        assert metrics['target_count']==result['metrics'][split]['target_count']
        assert math.isclose(metrics['cross_entropy'],result['metrics'][split]['cross_entropy'],abs_tol=1e-5)
    saved=json.loads((artifacts/f'{kind}_generations.json').read_text(encoding='utf-8'))
    assert saved['config']==generation_config
    for index,sample in enumerate(saved['samples']):
        assert sample['prompt']==generation_config['prompts'][index] and sample['seed']==generation_config['seeds'][index]
        ids=namespace['generate_neural_text'](model,sample['prompt'],sample['seed'],namespace['DEVICE'])
        assert ids==sample['generated_ids'] and len(ids)==50
        assert sample['generated_tokens']==[namespace['id_to_token'][i] for i in ids]
        assert not any(i in generation_config['blocked_token_ids'] for i in ids)
    print(kind,': restored validation/test scores, selected epoch, and three sample-ID checks passed.')
sg=torch.load(artifacts/'skipgram_input_vectors.pt',map_location='cpu',weights_only=True)
assert sg['config']==plan['self_embedding_training']
assert sg['corpus_sha256']==namespace['corpus_sha256'] and sg['id_to_token']==namespace['id_to_token']
assert torch.equal(tables['self'][4:],sg['input_vectors'][4:])
for cid in ('part2-skipgram-pairs-code','part2-skipgram-training-code','part2-self-alignment-code','part2-self-training-code',
            'part2-glove-alignment-code','part2-glove-training-code','part2-evaluation-code','part2-generation-code','part2-comparison-code'):
    c=by_id[cid]
    assert c.execution_count is not None and c.outputs
    assert not any(o.output_type=='error' for o in c.outputs)
assert any('image/png' in o.get('data',{}) for o in by_id['part2-evaluation-code'].outputs)
for p,digest in protected.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,p
print('Part II read-only audit passed; complete frozen input tables verified; artifacts unchanged.')
