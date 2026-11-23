# llm_judge.py  (Step 10 - part 1)
# Job: let an LLM grade the answers (LLM-as-a-judge). We ask ONE small question at a time
# and demand ONE word as the reply. Small models are poor at long, structured replies.

import re

SYSTEM_PROMPT = ("You are a strict evaluator. Reply with exactly one word and nothing else. "
                 "Do not explain.")

FAITHFULNESS_SCORES = {"YES": 1.0, "PARTIAL": 0.5, "NO": 0.0}
YES_NO_SCORES = {"YES": 1.0, "NO": 0.0}


def parse_verdict(reply, allowed_words):
    """Find the first allowed word (YES / NO / PARTIAL) in the reply. None if there is none."""
    for word in re.findall(r"[A-Za-z]+", reply.upper()):
        if word in allowed_words:
            return word
    return None


class LLMJudge:
    def __init__(self, generator):
        self.generator = generator          # any object with generate(messages) -> text

    def _ask(self, prompt, scores):
        reply = self.generator.generate([("system", SYSTEM_PROMPT), ("human", prompt)])
        verdict = parse_verdict(reply, list(scores.keys()))
        score = scores[verdict] if verdict is not None else None     # None = could not be parsed
        return score, reply

    def faithfulness(self, context, answer):
        prompt = ("Context:\n" + context + "\n\nAnswer:\n" + answer + "\n\n"
                  "Is every factual claim in the Answer supported by the Context?\n"
                  "Reply with one word: YES (all claims supported), PARTIAL (some claims "
                  "supported, some not) or NO (not supported).")
        return self._ask(prompt, FAITHFULNESS_SCORES)

    def relevance(self, question, answer):
        prompt = ("Question: " + question + "\n\nAnswer:\n" + answer + "\n\n"
                  "Does the Answer directly address the Question? Reply with one word: YES or NO.")
        return self._ask(prompt, YES_NO_SCORES)

    def correctness(self, question, expected_answer, answer):
        prompt = ("Question: " + question + "\nExpected answer: " + expected_answer +
                  "\n\nAnswer:\n" + answer + "\n\n"
                  "Does the Answer state the same key facts as the Expected answer? "
                  "Numbers must match. Reply with one word: YES or NO.")
        return self._ask(prompt, YES_NO_SCORES)

    def citation(self, answer, cited_text):
        prompt = ("Answer:\n" + answer + "\n\nCited passages:\n" + cited_text + "\n\n"
                  "Do the cited passages support the Answer? Reply with one word: YES or NO.")
        return self._ask(prompt, YES_NO_SCORES)