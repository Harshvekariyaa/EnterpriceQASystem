# run_chunking.py  (Step 4)
# Run from the project folder:   python scripts/run_chunking.py
# Reads data/raw/*.json  ->  writes data/processed/chunks_80.json and chunks_160.json

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, resolve_path
from src.ingestion.chunker import make_chunks
from src.ingestion.cleaner import build_text_blocks
from src.ingestion.dedup import remove_duplicates

config = load_config()
raw_folder = str(resolve_path(config["paths"]["raw_dir"]))
processed_folder = str(resolve_path(config["paths"]["processed_dir"]))
chunk_sizes = config["ingestion"]["chunk_sizes_words"]      # [80, 160]
overlap = config["ingestion"]["chunk_overlap_words"]        # 15
os.makedirs(processed_folder, exist_ok=True)

# 1. Load the JSON files made in Step 3
documents = []
for file_name in sorted(os.listdir(raw_folder)):
    if file_name.endswith(".json"):
        with open(os.path.join(raw_folder, file_name), encoding="utf-8") as f:
            documents.append(json.load(f))
print("Documents loaded:", len(documents))

# 2. Clean each document once
all_blocks = []
for doc in documents:
    all_blocks.append(build_text_blocks(doc))

# 3. Make chunks for each chunk size
for size in chunk_sizes:
    all_chunks = []
    for doc, blocks in zip(documents, all_blocks):
        all_chunks = all_chunks + make_chunks(doc, blocks, size, overlap)

    unique_chunks, removed = remove_duplicates(all_chunks)

    output_path = os.path.join(processed_folder, "chunks_" + str(size) + ".json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(unique_chunks, f, indent=2)

    print()
    print("Chunk size", size, "words (overlap", overlap, ")")
    print("  chunks created   :", len(all_chunks))
    print("  duplicates removed:", removed)
    print("  unique chunks    :", len(unique_chunks))
    print("  saved to         :", output_path)

# 4. Show one example chunk so you can see what it looks like
example = unique_chunks[1]
print()
print("Example chunk (size", chunk_sizes[-1], "):")
for key in ["chunk_id", "file_name", "section", "page", "word_count", "text"]:
    print(" ", key, "->", example[key])