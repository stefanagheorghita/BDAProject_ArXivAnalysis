import re
import csv
import sys
import subprocess
from pathlib import Path

import requests
from bs4 import BeautifulSoup

TAXONOMY_URL = "https://arxiv.org/category_taxonomy"
OUT_LOCAL = Path("arxiv_category_map.csv")
HDFS_DEST = "hdfs:///user/ubuntu/arxiv_data/arxiv_category_map_1.csv"

RE_H4 = re.compile(r"^\s*([A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)?)\s*\(([^)]+)\)\s*$")
RE_PARENS = re.compile(r"^\s*\(([^)]+)\)\s*$")

def fetch_html() -> str:
    r = requests.get(
        TAXONOMY_URL,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0 (category-map-scraper)"}
    )
    r.raise_for_status()
    return r.text

def extract_categories_and_archives(html: str):
    soup = BeautifulSoup(html, "html.parser")

    rows = []  # (code, pretty_name, group_name)
    group = None
    current_archive_name = None


    tags = soup.find_all(["h2", "h3", "h4", "p"])

    for i, tag in enumerate(tags):
        txt = tag.get_text(" ", strip=True)

        if tag.name == "h2":
            group = txt
            current_archive_name = None
            continue

        if tag.name == "h3":
            current_archive_name = txt

            archive_id = None
            for j in range(i + 1, min(i + 6, len(tags))):
                t2 = tags[j].get_text(" ", strip=True)
                m = RE_PARENS.match(t2)
                if m:
                    archive_id = m.group(1).strip()
                    break

            if archive_id:
                rows.append((archive_id, current_archive_name, group or ""))
            continue

        if tag.name == "h4":
            m = RE_H4.match(txt)
            if m:
                cat_id = m.group(1).strip()
                pretty = m.group(2).strip()
                rows.append((cat_id, pretty, group or ""))
            continue

    return rows

def write_csv(rows, out_path: Path) -> int:
    seen = set()
    dedup = []
    for code, pretty, group in rows:
        if code in seen:
            continue
        seen.add(code)
        dedup.append((code, pretty, group))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category_id", "pretty_name", "group_name"])
        w.writerows(dedup)

    return len(dedup)

def put_to_hdfs(local_path: Path, hdfs_dest: str):
    subprocess.check_call(["hdfs", "dfs", "-put", "-f", str(local_path), hdfs_dest])

def main():
    html = fetch_html()
    rows = extract_categories_and_archives(html)

    if not rows:
        print("ERROR: No categories extracted. Page structure may have changed.", file=sys.stderr)
        sys.exit(2)

    manual_archives = {
        "astro-ph": ("Astrophysics", "Physics"),
        "cond-mat": ("Condensed Matter", "Physics"),
        "hep-th": ("High Energy Physics - Theory", "Physics"),
    }

    existing_codes = {code for (code, _, _) in rows}
    for code, (pretty, group_name) in manual_archives.items():
        if code not in existing_codes:
            rows.append((code, pretty, group_name))

    n = write_csv(rows, OUT_LOCAL)

    codes = {r[0] for r in rows}
    missing = [c for c in ["astro-ph", "cond-mat"] if c not in codes]
    if missing:
        print(f"WARNING: Still missing expected archive codes: {', '.join(missing)}", file=sys.stderr)

    print(f"Wrote {n} codes to {OUT_LOCAL.resolve()}")

    try:
        put_to_hdfs(OUT_LOCAL, HDFS_DEST)
        print(f"Uploaded to HDFS: {HDFS_DEST}")
    except FileNotFoundError:
        print("WARNING: 'hdfs' command not found; upload manually:")
        print(f"  hdfs dfs -put -f {OUT_LOCAL} {HDFS_DEST}")
    except subprocess.CalledProcessError as e:
        print("WARNING: HDFS upload failed; upload manually:")
        print(f"  hdfs dfs -put -f {OUT_LOCAL} {HDFS_DEST}")
        print(f"Details: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
