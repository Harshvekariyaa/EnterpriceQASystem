# test_generator.py  (Step 13)
# The Ollama integration is tested with ChatOllama REPLACED by a fake,
# so these tests need no running LLM.
import pytest

from src.config import load_config
from src.rag import generator as generator_module
from src.rag.generator import GenerationError, OllamaGenerator


class FakeReply:
    def __init__(self, content):
        self.content = content


class FakeChatOllama:
    """Records how it was created and returns a fixed reply."""
    last_kwargs = None
    reply_text = "  Employees get 12 days [1].  \n"
    should_fail = False

    def __init__(self, **kwargs):
        FakeChatOllama.last_kwargs = kwargs

    def invoke(self, messages):
        if FakeChatOllama.should_fail:
            raise ConnectionError("connection refused")
        FakeChatOllama.last_messages = messages
        return FakeReply(FakeChatOllama.reply_text)


def install_fake(monkeypatch, should_fail=False):
    FakeChatOllama.should_fail = should_fail
    FakeChatOllama.last_kwargs = None
    monkeypatch.setattr(generator_module, "ChatOllama", FakeChatOllama)


def test_generate_returns_the_stripped_answer_text(monkeypatch):
    install_fake(monkeypatch)
    answer = OllamaGenerator().generate([("system", "rules"), ("human", "question")])
    assert answer == "Employees get 12 days [1]."


def test_messages_are_passed_to_the_model_unchanged(monkeypatch):
    install_fake(monkeypatch)
    messages = [("system", "rules"), ("human", "question")]
    OllamaGenerator().generate(messages)
    assert FakeChatOllama.last_messages == messages


def test_the_model_is_created_with_the_settings_from_config_yaml(monkeypatch):
    install_fake(monkeypatch)
    OllamaGenerator()
    settings = load_config()["llm"]
    assert FakeChatOllama.last_kwargs["model"] == settings["model"]
    assert FakeChatOllama.last_kwargs["base_url"] == settings["base_url"]
    assert FakeChatOllama.last_kwargs["temperature"] == settings["temperature"]
    assert FakeChatOllama.last_kwargs["temperature"] == 0.0             # stable answers


def test_model_and_length_can_be_overridden(monkeypatch):
    install_fake(monkeypatch)
    OllamaGenerator(model="llama3.2:3b", max_tokens=8)
    assert FakeChatOllama.last_kwargs["model"] == "llama3.2:3b"
    assert FakeChatOllama.last_kwargs["num_predict"] == 8


def test_a_failing_ollama_becomes_a_clear_generation_error(monkeypatch):
    install_fake(monkeypatch, should_fail=True)
    with pytest.raises(GenerationError) as error:
        OllamaGenerator().generate([("human", "hi")])
    assert "Ollama" in str(error.value)