import inspect
from enum import Enum


class SupportedModel(Enum):
    GEMINI = "gemini/gemini-1.5-flash"
    GPT4O = "openai/gpt-4o-mini"
    LLAMA3 = "groq/llama-3.1-70b-versatile"

    @classmethod
    def as_list(cls, only_values=False):
        _list = []
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not inspect.isroutine(value):
                _list.append(value.value if only_values else value)
        return _list


class Flores200Topic(Enum):
    LAW = ("crime and law", "law", "policy", "united_states_charters_of_freedom/constitution")
    HEALTH = ("health", "disease, research, canada", "science/disease", "science/sexual health", "science/first aid")
    NEWS = ("politics", "accident", "disasters and accidents", "politics and conflicts", "international",
            "internatoinal")