"""Download the Gensim redistribution of GloVe 6B 100d; verify release checksum."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

folder = Path(__file__).resolve().parents[1]
destination = folder / 'external_embeddings'
destination.mkdir(exist_ok=True)
metadata_url = 'https://raw.githubusercontent.com/piskvorky/gensim-data/master/list.json'
metadata = json.loads(urllib.request.urlopen(metadata_url, timeout=60).read())['models']['glove-wiki-gigaword-100']
assert metadata['parameters']['dimension'] == 100 and metadata['num_records'] == 400000
url = 'https://github.com/piskvorky/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz'
path = destination / metadata['file_name']
start = time.perf_counter()
if not path.exists():
    temporary = path.with_suffix('.gz.partial')
    print('Downloading GloVe 6B 100d redistribution:', metadata['file_size'], 'bytes', flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
        received = 0
        previous_report = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            received += len(chunk)
            if received - previous_report >= 16 * 1024 * 1024:
                print(f'Downloaded {received / 1024**2:.1f} MiB', flush=True)
                previous_report = received
    assert temporary.stat().st_size == metadata['file_size']
    assert hashlib.file_digest(temporary.open('rb'), 'md5').hexdigest() == metadata['checksum']
    temporary.replace(path)
with path.open('rb') as handle:
    md5 = hashlib.file_digest(handle, 'md5').hexdigest()
assert md5 == metadata['checksum']
with path.open('rb') as handle:
    sha = hashlib.file_digest(handle, 'sha256').hexdigest()
record = {'name': 'GloVe 6B 100d', 'distribution': 'Gensim-data word2vec-text gzip redistribution',
          'url': url, 'original_source': 'https://nlp.stanford.edu/projects/glove/',
          'metadata_url': metadata_url, 'file': path.name, 'sha256': sha, 'md5': md5,
          'download_or_verification_seconds': time.perf_counter()-start, 'metadata': metadata}
(destination / 'glove_source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print('GloVe verified:', path.name, 'MD5', md5, 'SHA256', sha, flush=True)
