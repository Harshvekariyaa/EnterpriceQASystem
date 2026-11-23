# app.py  (Step 12 - part 2)
# Job: a simple web page for employees. It does NOT contain any RAG logic: it only talks to the API.
# Run (second terminal):  streamlit run src/ui/app.py

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="NovaTech Knowledge Q&A", page_icon="📄")
st.title("📄 NovaTech Knowledge Q&A")
st.caption("Ask about leave, remote work, IT security, expenses, travel, conduct, "
           "onboarding, performance, privacy or benefits. Answers come only from company documents.")

# ---- sidebar: system status ----
with st.sidebar:
    st.header("System status")
    try:
        health = requests.get(API_URL + "/health", timeout=5).json()
        if health["status"] == "ok":
            st.success("All systems running")
        else:
            st.warning("Degraded")
            if not health["ollama"]["reachable"]:
                st.write("Ollama is not reachable.")
            elif not health["ollama"]["model_available"]:
                st.write("Model not downloaded: " + health["ollama"]["model"])
        st.write("Knowledge base version: **" + health["kb_version"] + "**")
    except Exception:
        st.error("The API is not running. Start it with: uvicorn src.api.main:app")

# ---- example questions ----
EXAMPLES = [
    "How many days of annual leave are credited each month?",
    "What is the daily limit for meal claims on business duty?",
    "Who won the cricket world cup?",
]
if "question" not in st.session_state:
    st.session_state["question"] = ""

st.write("Try an example:")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example, use_container_width=True):
        st.session_state["question"] = example

question = st.text_input("Your question", key="question", max_chars=500)
ask_clicked = st.button("Ask", type="primary")

# ---- ask and show the result ----
if ask_clicked:
    if question.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching the documents ..."):
            try:
                reply = requests.post(API_URL + "/ask", json={"question": question}, timeout=180)
            except requests.exceptions.RequestException:
                reply = None

        if reply is None:
            st.error("Could not reach the API. Is it running?")
        elif reply.status_code != 200:
            st.error(reply.json().get("detail", "Something went wrong."))
        else:
            data = reply.json()
            if data["refused"]:
                st.warning("**Not found.** " + data["answer"])
                st.caption("Reason: " + str(data["refusal_reason"]))
            else:
                st.markdown("### Answer")
                st.write(data["answer"])

                st.markdown("### Sources")
                if len(data["citations"]) == 0:
                    st.info("No source was cited for this answer. Please verify it.")
                for c in data["citations"]:
                    title = "[%d] %s | %s | page %s" % (c["number"], c["file_name"], c["section"], c["page"])
                    with st.expander(title):
                        st.write(c["snippet"])
                for warning in data["citation_warnings"]:
                    st.caption("⚠️ " + warning)

            # badge: where did this answer come from?
            if data["cache"] == "exact":
                st.success("⚡ From cache (exact match)")
            elif data["cache"] == "semantic":
                st.success("⚡ From cache (similar question, similarity %.2f)" % data["cache_similarity"])
            else:
                st.info("🔎 Freshly generated")
            st.caption("Response time: %.2f seconds" % data["response_time_seconds"])