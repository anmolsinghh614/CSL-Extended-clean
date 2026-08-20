import pathlib

path = pathlib.Path('LARGE_DATASET_RUNBOOK.md')
text = path.read_text(encoding='utf-8')
text = text.replace('/data/inat', '$HOME/data/inat')
text = text.replace('/data/imagenet', '$HOME/data/imagenet')
path.write_text(text, encoding='utf-8')

for number, line in enumerate(text.splitlines(), 1):
    if 'HOME/data' in line:
        print(f"{number}: {line.strip()}")
