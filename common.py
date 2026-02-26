import os, json, datetime
import hashlib
import fcntl
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




def _atomic_lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(path) + '.lock'
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_lock(fd):
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except Exception:
        pass

def append_jsonl(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = _atomic_lock(p)
    try:
        with p.open('a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    finally:
        _release_lock(fd)


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
    fd = _atomic_lock(p)
    try:
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(p)
    finally:
        _release_lock(fd)


def keywords_from_env(name, default=''):
    val = os.getenv(name, default)
    return [x.strip() for x in val.split(',') if x.strip()]
