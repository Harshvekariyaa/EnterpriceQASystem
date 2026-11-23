# ask.py  (Step 8)
# Ask one question:       python scripts/ask.py "How many sick leave days do I get?"
# Or chat in the terminal: python scripts/ask.py     (press Enter on an empty line to stop)

import os
import sys
from src.cache.cache_manager import CachedRAG

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.generator import GenerationError
from src.rag.pipeline import build_pipeline


def show(result):
    print("\nANSWER :", result["answer"])
    if result["refused"]:
        print("REFUSED: yes  (reason:", result["refusal_reason"] + ")")
    else:
        print("SOURCES:")
        for c in result["citations"]:
            print("   [" + str(c["number"]) + "]", c["file_name"], "|", c["section"], "| page", c["page"])
        for warning in result["citation_warnings"]:
            print("WARNING:", warning)
    print("top similarity:", result["top_similarity"], "| time:", result["latency_seconds"], "seconds\n")
    print("CACHE  :", result.get("cache", "none"), result.get("cache_similarity") or "")


def ask_safely(pipeline, question):
    try:
        show(pipeline.ask(question))
    except ValueError as error:
        print("Invalid question:", error)
    except GenerationError as error:
        print("LLM problem:", error)


pipeline = CachedRAG(build_pipeline())
if len(sys.argv) > 1:
    ask_safely(pipeline, " ".join(sys.argv[1:]))
else:
    while True:
        text = input("Your question (empty to stop): ").strip()
        if text == "":
            break
        ask_safely(pipeline, text)