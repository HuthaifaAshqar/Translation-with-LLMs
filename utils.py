import json
import os
import time

import numpy as np
from bert_score import score
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
from huggingface_hub import login
from litellm import completion
from loguru import logger
from sacrebleu import corpus_bleu, corpus_chrf

from constants import SupportedModel


def make_model_rhyme(model: SupportedModel = SupportedModel.GEMINI):
    if model not in SupportedModel.as_list():
        raise ValueError(f"{model} not supported")

    messages = [{"role": "user", "content": "Write a short poem about the sea."}]

    response = completion(model=model.value, messages=messages)

    logger.info(response["choices"][0]["message"]["content"])


def extract_iso_mappings(output_file="data/iso_mappings.json"):
    """Extract distinct iso_639_3: iso_15924 pairs and save to a JSON file."""
    # Load the dataset
    dataset = load_dataset("openlanguagedata/flores_plus")

    # Select the relevant split (e.g., 'devtest')
    train_split = dataset["devtest"]

    # Extract distinct pairs
    iso_mappings = {}
    for entry in train_split:
        iso_639_3 = entry["iso_639_3"]
        iso_15924 = entry["iso_15924"]

        if iso_639_3 and iso_15924:  # Ensure valid values
            iso_mappings[iso_639_3] = iso_15924

    # Save to the output file
    with open(output_file, "w") as file:
        json.dump(iso_mappings, file, indent=4)

    print(f"Extracted mappings saved to {output_file}")


def inspect_flores200():
    login(os.environ.get("HF_API_KEY"))

    dataset = load_dataset("openlanguagedata/flores_plus")
    logger.info(dataset)
    train_split = dataset["devtest"]
    logger.info(train_split[0])

    # Extract the 'topic' column values from the 'dev' split
    # topics = train_split['topic']
    #
    # # Get the distinct values of 'topic'
    # distinct_topics = set(topics)
    #
    # # Log the distinct topics
    # logger.info(f"Distinct topics: {distinct_topics}")


def invoke_llm(prompt: str, model: SupportedModel = SupportedModel.GEMINI, system_message: str = None, temperature=0):
    # print(f"\nPrompt: \n{prompt}")

    if system_message:
        messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]

    try:
        response = completion(
            model=model.value, messages=messages, temperature=temperature, retry_strategy="exponential_backoff_retry"
        )
        llm_response = response["choices"][0]["message"]["content"]
        # print(f"\nLLM Response: \n{llm_response}")
    except Exception as e:
        logger.error(f"An error occurred during LLM invocation: {e}")
        time.sleep(10)
        try:
            response = completion(
                model=model.value,
                messages=messages,
                temperature=temperature,
                retry_strategy="exponential_backoff_retry",
            )
            llm_response = response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"An error occurred during LLM invocation: {e}")
            return "An error occurred while processing your request. Please try again later."

    return llm_response


def calculate_bleu(translations, reference_translations):
    return corpus_bleu(translations, [reference_translations]).score


def calculate_chrf(translations, references):
    return corpus_chrf(translations, [references]).score


def load_comet_model():
    comet_model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_model_path)
    return comet_model


def calculate_comet(comet_model, source, hypothesis, reference):
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(source, hypothesis, reference)]
    results = comet_model.predict(data, batch_size=8, gpus=0)
    comet_score = np.mean(results["scores"])

    print(f"{comet_score=}")

    return comet_score * 100


def calculate_bertscore(baseline_translations, reference_translations, lang_code):
    P, R, F = score(baseline_translations, reference_translations, lang=lang_code)
    bertscore = np.mean(F.numpy())
    return bertscore * 100
