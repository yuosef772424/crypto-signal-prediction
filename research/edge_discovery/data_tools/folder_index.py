"""
PURPOSE:  Builds a title->fileId index of a Drive folder from search_files results captured in session transcripts and
          lists missing local files.
TAGS:     drive index, fileId, missing_ids, futures_metrics folder, data rebuild
PITFALLS: Reads /root/.claude/projects transcripts and hard-coded /home/user paths: only works inside the original
          session environment.

Build a title->fileId index for a Drive folder from search_files results
(parentId = '<folder>') captured in the session transcripts, then list which
pending symbols are still missing locally.

Usage: python3 folder_index.py [FOLDER_ID]
Writes drive_index_<folder>.csv and missing_ids.txt (one 'TITLE ID' per line).
"""
import glob, json, os, re, sys

FOLDER = sys.argv[1] if len(sys.argv) > 1 else '1LmmrTpBo-62nao_0FdqkZrohlMgfGTQT'
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = '/home/user/research/metrics_new'
PAT = re.compile(r'"id":"([\w-]+)"[^{}]*?"parentId":"%s"[^{}]*?"title":"([^"]+)"' % re.escape(FOLDER))

idx = {}
srcs = glob.glob('/root/.claude/projects/*/*.jsonl') + glob.glob('/root/.claude/projects/*/*/tool-results/*.txt')
for f in srcs:
    with open(f, errors='ignore') as fh:
        txt = fh.read().replace('\\"', '"')
    for fid, title in PAT.findall(txt):
        idx[title] = fid

out = os.path.join(HERE, f'drive_index_{FOLDER}.csv')
with open(out, 'w') as fh:
    fh.write('title,id\n')
    for t in sorted(idx):
        fh.write(f'{t},{idx[t]}\n')

pending = [l.strip() for l in open(os.path.join(HERE, 'newlist_pending.txt')) if l.strip()]
have = set(os.listdir(LOCAL)) if os.path.isdir(LOCAL) else set()
miss, unknown = [], []
for p in pending:
    t = p if p.endswith('.csv.gz') else f'{p}USDT.csv.gz' if not p.endswith('USDT') else f'{p}.csv.gz'
    if t in have:
        continue
    (miss if t in idx else unknown).append((t, idx.get(t)))
with open(os.path.join(HERE, 'missing_ids.txt'), 'w') as fh:
    for t, i in miss:
        fh.write(f'{t} {i}\n')
print(f'indexed {len(idx)} files; pending {len(pending)}; local {len(have)}; '
      f'missing-with-id {len(miss)}; missing-not-indexed {len(unknown)}')
print('not indexed yet:', ' '.join(t for t, _ in unknown[:80]))
