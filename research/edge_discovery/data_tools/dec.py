"""
PURPOSE:  Decodes one Drive download_file_content tool-result (base64 JSON) into a file under an output dir.
TAGS:     decode, base64, drive download, tool result, data rebuild
PITFALLS: Usage: dec.py <toolresult.txt> <outdir>.
"""
import json,base64,sys,os
# usage: dec.py <toolresult.txt> <outdir>
d=json.load(open(sys.argv[1])); b=base64.b64decode(d['content'])
os.makedirs(sys.argv[2],exist_ok=True); p=os.path.join(sys.argv[2],d['title']); open(p,'wb').write(b); print(p,len(b))
