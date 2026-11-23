# run_extract.py  (Step 3 - simple version)
# Run from the project folder:   python scripts/run_extract.py

import os
import sys

# This line lets Python find our "src" folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.ingestion.pdf_reader import read_all_pdfs, save_as_json

config = load_config()
pdf_folder = resolve_path(config["paths"]["pdf_dir"])
raw_folder = resolve_path(config["paths"]["raw_dir"])

documents = read_all_pdfs(str(pdf_folder))

total_words = 0
for doc in documents:
    save_as_json(doc, str(raw_folder))
    words = 0
    for page in doc["pages"]:
        words = words + len(page["text"].split())
    total_words = total_words + words
    print(doc["file_name"], "|", doc["doc_id"], "|", doc["department"],
          "| version", doc["version"], "|", doc["num_pages"], "page(s) |", words, "words")

print()
print("Total PDFs read :", len(documents))
print("Total words     :", total_words)
print("JSON files saved in:", raw_folder)