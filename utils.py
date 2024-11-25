import os

from datasets import load_dataset
from huggingface_hub import login
from litellm import completion
from loguru import logger

from constants import SupportedModel


def test_api_call(model: SupportedModel = SupportedModel.GEMINI):
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
