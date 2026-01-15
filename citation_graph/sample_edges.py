import ijson, re
from collections import Counter

KEEP = {"alg-geom", "plasm-ph", "hep-th"} 
cat_re = re.compile(r"^([^/]+)/")

edge_sample = 5000
edges = 0
deg_out = []

with open("/home/ubuntu/internal-references-pdftotext.json", "rb") as f:
    for pid, refs in ijson.kvitems(f, ""):
        cat = cat_re.match(pid).group(1) if cat_re.match(pid) else None
        if cat not in KEEP:
            continue
        if not isinstance(refs, list):
            continue
        deg_out.append(len(refs))
        edges += len(refs)
        if edges >= edge_sample:
            break

print("sample avg out-degree:", sum(deg_out)/max(1,len(deg_out)))
print("sample max out-degree:", max(deg_out) if deg_out else 0)
print("sample total edges:", edges)
