# pairs.py
import re
import ijson

IN_PATH = "/home/ubuntu/internal-references-pdftotext.json"
OUT_PATH = "/home/ubuntu/arxiv_edges.csv"

cat_re = re.compile(r"^([^/]+)/")

def cat_of(pid: str):
    m = cat_re.match(pid)
    return m.group(1) if m else None


KEEP_CATS = None             

MAX_EDGES = None  

def include(pid: str) -> bool:
    if KEEP_CATS is None:
        return True
    c = cat_of(pid)
    return c in KEEP_CATS

edges_written = 0
cats_seen = set()

with open(IN_PATH, "rb") as f, open(OUT_PATH, "w", encoding="utf-8") as out:
    out.write("src,dst\n")
    for src, refs in ijson.kvitems(f, ""):
        c = cat_of(src)
        if c:
            cats_seen.add(c)

        if not include(src):
            continue
        if not isinstance(refs, list) or len(refs) == 0:
            continue

        for dst in refs:
            if isinstance(dst, str):
                out.write(f"{src},{dst}\n")
                edges_written += 1
                if MAX_EDGES is not None and edges_written >= MAX_EDGES:
                    break

        if MAX_EDGES is not None and edges_written >= MAX_EDGES:
            break

print("edges_written:", edges_written)
print("saved to:", OUT_PATH)
print("num categories seen:", len(cats_seen))
print("categories:", ", ".join(sorted(cats_seen)))
