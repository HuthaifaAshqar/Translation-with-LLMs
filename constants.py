import inspect
from enum import Enum


class SupportedModel(Enum):
    GEMINI = "gemini/gemini-1.5-pro"
    GPT4O = "openai/gpt-4o-mini"
    LLAMA3 = "groq/llama3-8b-8192"

    @classmethod
    def as_list(cls, only_values=False):
        _list = []
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not inspect.isroutine(value):
                _list.append(value.value if only_values else value)
        return _list


# Define your target language pairs
LANGUAGE_PAIRS = [
    ("en", "ar"),  # English to Arabic
    ("en", "sw"),  # English to Swahili
    ("en", "hi"),  # English to Hindi
    ("en", "yo"),  # English to Yoruba
]
