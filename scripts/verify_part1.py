"""Read-only Part I audit: restore all models, score test targets, reproduce samples."""
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import nbformat

folder = Path(__file__).resolve().parents[1]
artifacts = folder / 'artifacts'
protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts.iterdir() if p.is_file()}
notebook = nbformat.read(folder / 'Assignment_code.ipynb', 4)
nbformat.validate(notebook)
assert len({c.id for c in notebook.cells}) == len(notebook.cells)
by_id = {c.id: c for c in notebook.cells}
namespace = {'json': json, 'math': math, 'time': time, 'Counter': Counter, 'defaultdict': defaultdict}
os.chdir(folder)

def execute(source):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, '<part1-read-only-check>', 'exec'), namespace)

def definition(cid, name):
    source = by_id[cid].source
    node = next(node for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    execute(ast.get_source_segment(source, node))

for cid in ('part1-load-code', 'part1-clean-code', 'part1-split-code', 'part1-vocab-code', 'part1-neural-batches-code'):
    execute(by_id[cid].source)
namespace['ARTIFACT_DIR'] = artifacts
execute(by_id['part1-training-api-code'].source)
execute(by_id['part1-rnn-model-code'].source)
namespace['rnn_result'] = json.loads((artifacts / 'rnn_results.json').read_text(encoding='utf-8'))
execute(by_id['part1-lstm-model-code'].source)
transformer_source = by_id['part1-transformer-model-code'].source
transformer_tree = ast.parse(transformer_source)
transformer_tree.body = [node for node in transformer_tree.body
                        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                                and isinstance(node.value.func, ast.Attribute)
                                and node.value.func.attr == 'write_text')]
execute(ast.unparse(transformer_tree))
for cid, name in [('part1-ngram-code', 'InterpolatedNGramLM'),
                  ('part1-ngram-selection-code', 'evaluate_ngram'),
                  ('part1-ngram-generation-code', 'sample_next_token'),
                  ('part1-ngram-generation-code', 'generate_ngram'),
                  ('part1-rnn-generation-code', 'generate_neural_text')]:
    definition(cid, name)

config = json.loads((artifacts / 'generation_config.json').read_text(encoding='utf-8'))
namespace.update({
    'GENERATION_PROMPTS': config['prompts'], 'GENERATION_LENGTH': config['length'],
    'GENERATION_TEMPERATURE': config['temperature'], 'GENERATION_TOP_K': config['top_k'],
    'GENERATION_BLOCKED_IDS': tuple(config['blocked_token_ids']),
})
snapshot = json.loads((artifacts / 'ngram_model.json').read_text(encoding='utf-8'))
assert snapshot['id_to_token'] == namespace['id_to_token']
assert snapshot['corpus_sha256'] == namespace['corpus_sha256']
ngram = namespace['InterpolatedNGramLM'](snapshot['vocab_size'], snapshot['order'], snapshot['alpha'], snapshot['weights'])
for context_length, rows in enumerate(snapshot['counts']):
    for row in rows:
        context = tuple(row['context'])
        successors = Counter({int(target): count for target, count in row['successors']})
        ngram.counts[context_length][context] = successors
        ngram.totals[context_length][context] = sum(successors.values())
ngram.target_count = snapshot['target_count']
models = {'ngram': ngram}
torch = namespace['torch']
for name, config_name in [('rnn', 'RNN_CONFIG'), ('lstm', 'LSTM_CONFIG'), ('transformer', 'TRANSFORMER_CONFIG')]:
    checkpoint = torch.load(artifacts / f'{name}_best.pt', map_location=namespace['DEVICE'], weights_only=True)
    assert checkpoint['config'] == namespace[config_name]
    assert checkpoint['id_to_token'] == namespace['id_to_token']
    assert checkpoint['corpus_sha256'] == namespace['corpus_sha256']
    model = namespace[f'{name}_model']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(namespace['DEVICE']).eval()
    training = json.loads((artifacts / f'{name}_training.json').read_text(encoding='utf-8'))
    assert training['best_epoch'] == checkpoint['epoch']
    assert training['best_epoch'] == min(training['history'], key=lambda row: row['validation']['cross_entropy'])['epoch']
    for row in training['history']:
        for split in ('train', 'validation'):
            assert row[split]['target_count'] == sum(len(s)-1 for s in namespace['encoded_paragraphs'][split])
    models[name] = model

loaders = namespace['make_lm_loaders'](42, 8)
for name, model in models.items():
    result = json.loads((artifacts / f'{name}_results.json').read_text(encoding='utf-8'))
    metrics = (namespace['evaluate_ngram'](model, namespace['encoded_paragraphs']['test']) if name == 'ngram'
               else namespace['run_lm_epoch'](model, loaders['test'], namespace['DEVICE']))
    assert metrics['target_count'] == result['metrics']['test']['target_count'] == 14811
    assert math.isclose(metrics['cross_entropy'], result['metrics']['test']['cross_entropy'], abs_tol=1e-5)
    saved = json.loads((artifacts / f'{name}_generations.json').read_text(encoding='utf-8'))
    assert saved['config'] == config
    assert len(saved['samples']) == 3
    for index, sample in enumerate(saved['samples']):
        assert sample['prompt'] == config['prompts'][index] and sample['seed'] == config['seeds'][index]
        ids = (namespace['generate_ngram'](model, sample['prompt'], sample['seed']) if name == 'ngram'
               else namespace['generate_neural_text'](model, sample['prompt'], sample['seed'], namespace['DEVICE']))
        assert ids == sample['generated_ids'] and len(ids) == 50
        assert not any(i in config['blocked_token_ids'] for i in ids)
        assert sample['generated_tokens'] == [namespace['id_to_token'][i] for i in ids]
        assert sample['continuation'] == ' '.join(sample['generated_tokens'])
    print(f'{name}: restored test CE={metrics["cross_entropy"]:.4f}, targets=14811, three sample-ID checks passed.')

comparison = json.loads((artifacts / 'part1_comparison.json').read_text(encoding='utf-8'))
assert len(comparison['rows']) == 4
assert len([c for c in notebook.cells if c.cell_type == 'code' and c.id.startswith('part1-')]) == 27
for cid in ['part1-transformer-training-code', 'part1-transformer-results-code', 'part1-transformer-cost-code',
            'part1-transformer-generation-code', 'part1-comparison-code']:
    cell = by_id[cid]
    assert cell.execution_count is not None and cell.outputs
    assert not any(o.output_type == 'error' for o in cell.outputs)
assert any('image/png' in o.get('data', {}) for o in by_id['part1-transformer-results-code'].outputs)
for p, digest in protected.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, p
print('Part I read-only audit passed; artifact hashes unchanged; 27 Part I code cells with real new outputs.')
