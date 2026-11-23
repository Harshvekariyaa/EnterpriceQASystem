"""
check_setup.py
Checks that libraries, config, PDFs and Ollama are all ready.
Run from the project root:   python scripts/check_setup.py
"""
import importlib
import sys
from pathlib import Path

# Allow "from src..." imports when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from src.config import load_config, resolve_path
from src.logger import get_logger

log = get_logger("check_setup")

# import name -> pip package name
PACKAGES = {
    "pypdf": "pypdf", "reportlab": "reportlab", "numpy": "numpy",
    "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib","torch": "torch",
    "transformers": "transformers", "sentence_transformers": "sentence-transformers",
    "chromadb": "chromadb","rank_bm25": "rank_bm25", "requests": "requests", "fastapi": "fastapi",
    "uvicorn": "uvicorn", "streamlit": "streamlit", "httpx": "httpx",
    "yaml": "pyyaml", "pytest": "pytest","langchain_community": "langchain-community",
    "langchain_text_splitters": "langchain-text-splitters","langchain_huggingface": "langchain-huggingface",
    "langchain_chroma": "langchain-chroma","langchain_ollama": "langchain-ollama",
}


def check_packages() -> bool:
    ok = True
    for import_name, pip_name in PACKAGES.items():
        try:
            importlib.import_module(import_name)
            print(f"  [OK]      {pip_name}")
        except ImportError:
            print(f"  [MISSING] {pip_name}  ->  pip install {pip_name}")
            ok = False
    return ok


def check_config() -> bool:
    try:
        cfg = load_config()
        print(f"  [OK]      config.yaml loaded (project: {cfg['project']['name']})")
        return True
    except Exception as exc:  # show any problem clearly
        print(f"  [FAILED]  config.yaml: {exc}")
        return False


def check_pdfs() -> bool:
    pdf_dir = resolve_path(load_config()["paths"]["pdf_dir"])
    count = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
    if count == 10:
        print(f"  [OK]      found {count} PDFs in {pdf_dir}")
        return True
    print(f"  [FAILED]  expected 10 PDFs in {pdf_dir}, found {count}")
    return False


def check_ollama() -> bool:
    llm = load_config()["llm"]
    try:
        resp = requests.get(f"{llm['base_url']}/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
    except Exception as exc:
        print(f"  [FAILED]  Ollama not reachable ({exc}). Is the Ollama app running?")
        return False

    if any(n.startswith(llm["model"]) for n in names):
        print(f"  [OK]      Ollama running and model '{llm['model']}' is available")
        return True
    print(f"  [FAILED]  model '{llm['model']}' not found. Installed: {names}")
    print(f"            Run: ollama pull {llm['model']}")
    return False


if __name__ == "__main__":
    print("\n1. Python libraries")
    r1 = check_packages()
    print("\n2. Configuration")
    r2 = check_config()
    print("\n3. Knowledge-base PDFs")
    r3 = check_pdfs() if r2 else False
    print("\n4. Ollama (local LLM)")
    r4 = check_ollama() if r2 else False

    if all([r1, r2, r3, r4]):
        log.info("Setup check passed. Ready for Step 3!")
        print("\nALL CHECKS PASSED. You are ready for the next step.")
    else:
        log.warning("Setup check found problems.")
        print("\nSome checks FAILED. Fix them using the messages above, then run again.")
        sys.exit(1)