"""Make a GitHub-ready archive without site identity, credentials, or caches."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
excluded={'.git','.openai','.venv','.pytest_cache','__pycache__','reports','node_modules'}
with ZipFile(root/'dist/source.zip','w',ZIP_DEFLATED) as out:
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        if not p.is_file() or any(x in excluded or x.endswith('.egg-info') for x in rel.parts):continue
        if p.name in {'.env','source.zip'} or (p.name.startswith('.env.') and p.name!='.env.example'):continue
        if rel.parts[:2] in {('data','raw'),('data','live')}:continue
        out.write(p,Path('credit-atlas')/rel)
print(root/'dist/source.zip')
