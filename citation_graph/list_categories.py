import re
import ijson

IN_PATH = "/home/ubuntu/internal-references-pdftotext.json"
cat_re = re.compile(r"^([^/]+)/")

cats = set()

with open(IN_PATH, "rb") as f:
    for paper_id, _ in ijson.kvitems(f, ""):
        m = cat_re.match(paper_id)
        if m:
            cats.add(m.group(1))

for c in sorted(cats):
    print(c)

print("num categories:", len(cats))
