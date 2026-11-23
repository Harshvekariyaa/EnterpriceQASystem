# prompt.py  (Step 8 - part 1)
# Job: build the instructions we send to the LLM.

# The LLM must reply with EXACTLY this sentence when the answer is not in the context.
# We also use it to DETECT a refusal, so it lives in one place.
NOT_FOUND_MESSAGE = "I could not find this in the company documents."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions for employees of NovaTech Solutions.\n"
    "Answer the question using ONLY the numbered context passages given below.\n"
    "Rules:\n"
    "1. Do not use outside knowledge. Do not guess.\n"
    "2. After every fact, write the number of the passage it came from, like [1] or [2].\n"
    "3. If the context does not contain the answer, reply with exactly: " + NOT_FOUND_MESSAGE + "\n"
    "4. Keep the answer short: one to three sentences."
)


def build_context(chunks):
    """Turn the retrieved chunks into numbered passages.
    Example:
        [1] Source: 01_Leave_Policy.pdf | Section: 3. Sick Leave | Page: 1
        Employees are entitled to 12 days of paid sick leave...
    """
    parts = []
    number = 1
    for chunk in chunks:
        header = ("[" + str(number) + "] Source: " + str(chunk["file_name"]) +
                  " | Section: " + str(chunk["section"]) + " | Page: " + str(chunk["page"]))
        parts.append(header + "\n" + chunk["text"])
        number = number + 1
    return "\n\n".join(parts)


def build_messages(question, chunks):
    """Return the messages for the LLM: a system message and the user's message."""
    user_text = "Context:\n" + build_context(chunks) + "\n\nQuestion: " + question + "\n\nAnswer:"
    return [("system", SYSTEM_PROMPT), ("human", user_text)]