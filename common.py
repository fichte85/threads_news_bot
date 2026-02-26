import os, json, datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('/home/ubuntu/threads-bot-news2/.env')

BASE = Path('/home/ubuntu/threads-bot-news2')
DATA = BASE / 'data'
DATA.mkdir(parents=True, exist_ok=True)


def now_iso():
    return datetime.datetime.now().isoformat()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def append_jsonl(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def read_json(path, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return default


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def keywords_from_env(name, default=''):
    val = os.getenv(name, default)
    return [x.strip() for x in val.split(',') if x.strip()]
