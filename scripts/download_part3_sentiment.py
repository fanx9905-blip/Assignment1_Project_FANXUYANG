"""Acquire the small UCI labelled sentence archive; preserve provenance."""
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

folder=Path(__file__).resolve().parents[1]
destination=folder/'downstream_data'
destination.mkdir(exist_ok=True)
url='https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
archive=destination/'uci_sentiment_labelled_sentences.zip'
if not archive.exists():
    data=urllib.request.urlopen(url,timeout=60).read()
    temporary=archive.with_suffix('.zip.partial')
    temporary.write_bytes(data)
    with zipfile.ZipFile(temporary) as reader:
        assert reader.testzip() is None
        for name in ('amazon_cells_labelled.txt','imdb_labelled.txt','yelp_labelled.txt'):
            assert any(member.endswith('/'+name) for member in reader.namelist())
    temporary.replace(archive)
record={'dataset':'UCI Sentiment Labelled Sentences','doi':'10.24432/C57604',
        'source_url':'https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences',
        'download_url':url,'license':'CC BY 4.0','citation':'Kotzias, D. (2015). Sentiment Labelled Sentences. UCI Machine Learning Repository.',
        'archive_file':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
(destination/'uci_source.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('UCI archive acquired:',record['bytes'],'bytes; SHA256',record['sha256'])
