# chunker.py  (Step 4 - LangChain version)
# Job: cut text into chunks using LangChain's RecursiveCharacterTextSplitter.
#
# "Recursive" means it tries the BEST place to cut first, then a worse place:
#   1st try: cut at a new line  (= between sections)
#   2nd try: cut after a sentence  (". ")
#   3rd try: cut at a space  (between words)
#   last   : cut anywhere
# So chunks end at natural places instead of in the middle of a sentence.

from langchain_text_splitters import RecursiveCharacterTextSplitter


def count_words(text):
    """We measure chunk size in WORDS (not characters)."""
    return len(text.split())


def find_section_and_page(markers, position):
    """Look up which section/page the text at `position` belongs to."""
    section, page = markers[0][1], markers[0][2]
    for marker_position, marker_section, marker_page in markers:
        if marker_position <= position:
            section, page = marker_section, marker_page
        else:
            break
    return section, page


def make_chunks(document, blocks, chunk_size, overlap):
    """Return a list of chunks (dictionaries) for one document."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # max words in a chunk
        chunk_overlap=overlap,          # words shared between neighbours
        length_function=count_words,    # count words, not characters
        separators=["\n", ". ", " ", ""],
        add_start_index=True,           # tells us where each chunk starts
    )

    chunks = []
    chunk_number = 0
    for block in blocks:
        pieces = splitter.create_documents([block["text"]])
        search_from = 0                       # chunks come in order, so we search forward only
        for piece in pieces:
            text = piece.page_content.lstrip(". ").strip()   # tidy the start
            if text == "":
                continue

            # Where does this chunk start inside the block text?
            # We find it OURSELVES. (LangChain's start_index can be -1 when we count
            # chunk size in words, and a wrong start gives a wrong section and page.)
            start = block["text"].find(text, search_from)
            if start < 0:
                start = block["text"].find(text)
            if start < 0:
                start = search_from           # last resort: stay at the previous position
            search_from = start + 1
            section, page = find_section_and_page(block["markers"], start)

            chunks.append({
                "doc_id": document["doc_id"],
                "file_name": document["file_name"],
                "title": document["title"],
                "section": section,
                "page": page,
                "chunk_index": chunk_number,
                "chunk_size": chunk_size,
                "word_count": count_words(text),
                "text": text,
            })
            chunk_number = chunk_number + 1
    return chunks