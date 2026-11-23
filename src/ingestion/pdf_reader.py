# pdf_reader.py  (Step 3 - LangChain version)
# Job: load all PDFs with LangChain, then collect the details of each document.
#
# DirectoryLoader  = opens every file in a folder
# PyPDFLoader      = reads a PDF using pypdf (one LangChain "Document" per PAGE)

import json
import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


def get_value_after(lines, label):
    """Find the line that says `label` (like 'Version') and return the NEXT line.
    Example: lines = ["Version", "3.2"]  ->  get_value_after(lines, "Version") gives "3.2"
    """
    for i in range(len(lines)):
        if lines[i] == label:
            return lines[i + 1]
    return "UNKNOWN"  # label not found


def load_pages(pdf_folder):
    """Load every PDF in the folder. Returns a list of page-Documents."""
    loader = DirectoryLoader(
        pdf_folder,
        glob="*.pdf",              # only PDF files
        loader_cls=PyPDFLoader,    # read them with pypdf
        silent_errors=True,        # a broken PDF is skipped instead of crashing
    )
    pages = loader.load()
    if len(pages) == 0:
        raise FileNotFoundError("No PDF pages could be loaded from: " + pdf_folder)
    return pages


def group_pages_by_file(pages):
    """LangChain gives one item per page. Here we group the pages of each file together."""
    grouped = {}
    for page in pages:
        file_name = os.path.basename(page.metadata["source"])
        if file_name not in grouped:
            grouped[file_name] = []
        grouped[file_name].append({
            "page_number": page.metadata["page"] + 1,   # LangChain counts pages from 0
            "text": page.page_content,
        })

    # make sure pages are in order: 1, 2, 3...
    for file_name in grouped:
        grouped[file_name].sort(key=lambda p: p["page_number"])
    return grouped


def build_document(file_name, pages):
    """Find the document details from the top of page 1 and return one dictionary."""
    # Split page 1 into lines, remove extra spaces, ignore empty lines
    lines = []
    for line in pages[0]["text"].split("\n"):
        line = line.strip()
        if line != "":
            lines.append(line)

    # Line 1 is the company name, line 2 is the document title
    title = lines[1] if len(lines) > 1 else file_name

    return {
        "file_name": file_name,
        "title": title,
        "doc_id": get_value_after(lines, "Document ID"),
        "department": get_value_after(lines, "Department"),
        "version": get_value_after(lines, "Version"),
        "effective_date": get_value_after(lines, "Effective Date"),
        "num_pages": len(pages),
        "pages": pages,
    }


def read_all_pdfs(pdf_folder):
    """Load all PDFs and return a list of document dictionaries (sorted by file name)."""
    pages = load_pages(pdf_folder)
    grouped = group_pages_by_file(pages)

    documents = []
    for file_name in sorted(grouped.keys()):
        documents.append(build_document(file_name, grouped[file_name]))
    return documents


def save_as_json(document, output_folder):
    """Save one document dictionary as a .json file."""
    os.makedirs(output_folder, exist_ok=True)
    name_without_pdf = document["file_name"].replace(".pdf", "")
    json_path = os.path.join(output_folder, name_without_pdf + ".json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(document, file, indent=2)