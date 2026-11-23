# check_chunk_labels.py  (Step 4 check)
# Run:   python scripts/check_chunk_labels.py
# Checks that every chunk which STARTS with a heading (like "7. Public Holidays")
# carries that same heading as its "section" label, and the right page number.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.ingestion.cleaner import is_heading

config = load_config()
raw_folder = str(resolve_path(config["paths"]["raw_dir"]))
processed_folder = str(resolve_path(config["paths"]["processed_dir"]))

pages_of = {}                                   # file name -> list of (page number, text)
for name in sorted(os.listdir(raw_folder)):
    if name.endswith(".json"):
        with open(os.path.join(raw_folder, name), encoding="utf-8") as f:
            doc = json.load(f)
        pages_of[doc["file_name"]] = [(p["page_number"], p["text"]) for p in doc["pages"]]

total_wrong = 0
for size in config["ingestion"]["chunk_sizes_words"]:
    with open(os.path.join(processed_folder, "chunks_" + str(size) + ".json"), encoding="utf-8") as f:
        chunks = json.load(f)
    checked = wrong = 0
    for chunk in chunks:
        first_line = chunk["text"].split("\n")[0].strip()
        if not is_heading(first_line):
            continue
        checked = checked + 1
        expected_pages = [number for number, text in pages_of[chunk["file_name"]] if first_line in text]
        bad_section = chunk["section"] != first_line
        bad_page = len(expected_pages) > 0 and chunk["page"] != expected_pages[0]
        if bad_section or bad_page:
            wrong = wrong + 1
            if wrong <= 3:
                print("  WRONG:", chunk["file_name"], "| starts with:", first_line,
                      "| labelled:", chunk["section"], "page", chunk["page"])
    print("chunk size", size, "| chunks starting with a heading:", checked, "| wrong labels:", wrong)
    total_wrong = total_wrong + wrong

print("\nRESULT:", "all labels correct" if total_wrong == 0 else str(total_wrong) + " wrong labels - re-run Step 4 and Step 5 (see instructions)")
sys.exit(0 if total_wrong == 0 else 1)