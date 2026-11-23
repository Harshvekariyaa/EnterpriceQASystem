# generator.py  (Step 8 - part 2)
# Job: send the prompt to the local LLM (Ollama) and return the answer text.
# Uses LangChain's ChatOllama.

from langchain_ollama import ChatOllama

from src.config import load_config
from src.logger import get_logger

log = get_logger("generator")


class GenerationError(Exception):
    """Raised when the LLM cannot give an answer (for example Ollama is not running)."""


class OllamaGenerator:
    def __init__(self, model=None, max_tokens=None):
        settings = load_config()["llm"]
        self.model_name = model or settings["model"]
        self.model = ChatOllama(
            model=self.model_name,
            base_url=settings["base_url"],
            temperature=settings["temperature"],                         # 0 = most stable answers
            num_predict=max_tokens or settings.get("max_tokens", 300),   # maximum answer length
        )
        
    def generate(self, messages):
        """Send messages to the LLM. Returns the answer as plain text."""
        try:
            reply = self.model.invoke(messages)
        except Exception as error:
            log.error("Ollama call failed: %s", error)
            raise GenerationError(
                "Could not get an answer from Ollama. Is the Ollama app running, "
                "and did you run: ollama pull " + self.model_name
            ) from error
        return reply.content.strip()