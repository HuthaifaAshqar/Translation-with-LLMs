import inspect
from enum import Enum


class SupportedModel(Enum):
    GEMINI = "gemini/gemini-1.5-pro"
    GPT4O = "openai/gpt-4o-mini"
    LLAMA3 = "groq/llama-3.1-70b-versatile"

    @classmethod
    def as_list(cls, only_values=False):
        _list = []
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not inspect.isroutine(value):
                _list.append(value.value if only_values else value)
        return _list
