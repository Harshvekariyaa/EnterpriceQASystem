# cleaner.py  (Step 4 - part 1)
# Job: clean the page text and build flowing text for the splitter.
# We also remember which SECTION and PAGE every part of the text belongs to.


def is_heading(line):
    """A heading looks like '3. Sick Leave' (a number, a dot, a few words).
    'Document Notice' is the only heading without a number."""
    if line == "Document Notice":
        return True
    words = line.split()
    if len(words) < 2 or len(words) > 8:
        return False
    first_word = words[0]                      # example: "3."
    return first_word.endswith(".") and first_word[:-1].isdigit()


def get_clean_lines(page_text, page_number):
    """Split page text into lines, remove spaces and empty lines.
    On page 1 we also remove the header table (Document ID, Version...),
    because it is not policy content."""
    lines = []
    for line in page_text.split("\n"):
        line = line.strip()
        if line != "":
            lines.append(line)

    if page_number == 1 and "Effective Date" in lines:
        position = lines.index("Effective Date")
        lines = lines[position + 2:]           # skip the label and its value
    return lines


def new_text_block():
    # "text"    = the flowing text of the block
    # "markers" = list of (position_in_text, section, page) so we can look up
    #             the section and page of any chunk later
    return {"text": "", "markers": []}


def build_text_blocks(document):
    """Turn a document into blocks of FLOWING text.
    - Normal lines are joined with a space (so sentences are not broken by line ends).
    - Every heading starts on a NEW line (so the splitter can cut at headings first).
    - 'Document Notice' becomes its own block."""
    blocks = []
    block = new_text_block()
    section = "Introduction"

    for page in document["pages"]:
        page_number = page["page_number"]
        for line in get_clean_lines(page["text"], page_number):
            if is_heading(line):
                section = line
                if line == "Document Notice" and block["text"] != "":
                    blocks.append(block)
                    block = new_text_block()
                if block["text"] != "" and not block["text"].endswith("\n"):
                    block["text"] = block["text"] + "\n"
                block["markers"].append((len(block["text"]), section, page_number))
                block["text"] = block["text"] + line + "\n"
            else:
                block["markers"].append((len(block["text"]), section, page_number))
                block["text"] = block["text"] + line + " "

    if block["text"] != "":
        blocks.append(block)
    return blocks