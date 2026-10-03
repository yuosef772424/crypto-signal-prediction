"""
PURPOSE:  Decodes saved Drive download tool-results (base64 gzip) into metrics/, metrics_new/, funding/ etc., removing
          the source file.
TAGS:     sweep, decode drive downloads, metrics_new, data rebuild, gzip
PITFALLS: Deletes the tool-result files it decodes (os.remove); hard-coded session paths, so only works in the
          original environment.
"""
import json,base64,gzip,io,glob,os
D='/root/.claude/projects/-home-user/eb453cbe-317e-569f-b6d5-da8e67e4ab0d/tool-results/'
n=0
for f in sorted(glob.glob(D+'mcp-Google_Drive-download_file_content-*.txt')):
    d=json.load(open(f)); b=base64.b64decode(d['content'])
    if b[:2]==b'\x1f\x8b':
        head=gzip.decompress(b)[:200].decode(errors='ignore').split('\n')[0]
        new=set(open('/home/user/crypto-signal-prediction/research/edge_discovery/data_tools/newlist_pending.txt').read().split())
        sub=('metrics_new' if d['title'].replace('.csv.gz','') in new else 'metrics') if 'sum_open_interest' in head else ('funding' if 'funding' in head.lower() else ('oi' if 'open_interest' in head.lower() else 'raw15m'))
    else: sub='meta'
    os.makedirs(sub,exist_ok=True); open(f'{sub}/{d["title"]}','wb').write(b); os.remove(f); n+=1
print('decoded',n,{s:len(glob.glob(s+'/*')) for s in ['metrics','metrics_new','raw15m']})
