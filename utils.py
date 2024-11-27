import os

from datasets import load_dataset
from huggingface_hub import login
from litellm import completion
from loguru import logger

from constants import SupportedModel


def make_model_rhyme(model: SupportedModel = SupportedModel.GEMINI):
    if model not in SupportedModel.as_list():
        raise ValueError(f"{model} not supported")

    messages = [{"role": "user", "content": "Write a short poem about the sea."}]

    response = completion(model=model.value, messages=messages)

    logger.info(response["choices"][0]["message"]["content"])


def inspect_flores200():
    login(os.environ.get("HF_API_KEY"))

    dataset = load_dataset("openlanguagedata/flores_plus")
    logger.info(dataset)
    train_split = dataset["dev"]
    logger.info(train_split[0])


def generate_context_paragraph(sentence, model: SupportedModel = SupportedModel.GEMINI):
    if model not in SupportedModel.as_list():
        raise ValueError(f"{model} not supported")

    prompt = f"Write a paragraph containing the following sentence:\n[English]: {sentence}"
    logger.info(f"{prompt=}")

    messages = [{"role": "user", "content": prompt}]

    response = completion(model=model.value, messages=messages, retry_strategy="exponential_backoff_retry")
    context_sentence = response["choices"][0]["message"]["content"]
    logger.info(f"{context_sentence=}")
    return context_sentence


def invoke_llm(prompt: str, model: SupportedModel = SupportedModel.GEMINI):
    messages = [{"role": "user", "content": prompt}]
    response = completion(model=model.value, messages=messages, retry_strategy="exponential_backoff_retry")
    content = response["choices"][0]["message"]["content"]
    logger.info(f"{content=}")
    return content
