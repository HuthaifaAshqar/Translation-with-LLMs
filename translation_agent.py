import os
import time
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from loguru import logger
from tqdm import tqdm

from constants import SupportedModel
from utils import invoke_llm


class TranslationAgent:
    def __init__(self):
        login(os.environ.get("HF_API_KEY"))
        self.dataset = load_dataset("openlanguagedata/flores_plus")
        self.sampled_df = pd.DataFrame([])
        self.extra_sentences = pd.DataFrame([])
        self.model = SupportedModel.GEMINI
        self.languages_code = {"French": "fra", "Arabic": "arb", "Thai": "tha"}

    def select_model(self, model: SupportedModel):
        logger.info(f"Selecting {model.name} model")
        self.model = model
        return self

    def sample(self, n=1):
        devtest_split = self.dataset["devtest"]

        english_sentences = devtest_split.filter(lambda x: x["iso_639_3"] == "eng")

        sampled_eng = english_sentences.shuffle(seed=42).select(range(n))
        sampled_ids = [s["id"] for s in sampled_eng]
        relevant_translations = devtest_split.filter(
            lambda x: x["id"] in sampled_ids and x["iso_639_3"] in self.languages_code.values()
        )

        cols = (
            ["eng", "context"]
            + list(self.languages_code.values())
            + [f"{c}_translation" for c in self.languages_code.values()]
        )
        df = pd.DataFrame(columns=cols)

        for sample in sampled_eng:
            _id = sample["id"]
            eng_text = sample["text"]

            # Get French, Arabic, and Thai translations using the same 'id'
            fra_text = relevant_translations.filter(lambda x: x["id"] == _id and x["iso_639_3"] == "fra")["text"][0]
            arb_text = relevant_translations.filter(
                lambda x: x["id"] == _id and x["iso_639_3"] == "arb" and x["iso_15924"] == "Arab"
            )["text"][0]
            tha_text = relevant_translations.filter(lambda x: x["id"] == _id and x["iso_639_3"] == "tha")["text"][0]

            df = pd.concat(
                [df, pd.DataFrame({"eng": [eng_text], "fra": [fra_text], "arb": [arb_text], "tha": [tha_text]})],
                ignore_index=True,
            )

        # Sample additional sentences for each language
        extra_sentences = defaultdict(list)
        for lang, langCode in self.languages_code.items():
            # Filter out sentences not in the sampled English IDs and sample 3 unique ones
            lang_sentences = (
                devtest_split.filter(lambda x: x["iso_639_3"] == langCode and x["id"] not in sampled_ids)
                .shuffle(seed=42)
                .select(range(3))
            )
            extra_sentences[lang] = [s["text"] for s in lang_sentences]

        self.sampled_df = df
        self.sampled_df.to_csv(f"sample_{n}.csv", index=False)

        self.extra_sentences = extra_sentences
        return self

    def generate_context(self):
        for index, row in tqdm(self.sampled_df.iterrows(), desc="Generate context"):
            prompt = f"Write a paragraph containing the following sentence:\n[English]: {row['eng']}"
            context_sentence = invoke_llm(prompt, self.model)
            self.sampled_df.at[index, "context"] = context_sentence
            time.sleep(2)

        return self

    def translate_with_context(self):
        for index, row in tqdm(self.sampled_df.iterrows(), desc="Translate with context"):
            for lang, langCode in self.languages_code.items():
                prompt = f"""
                    Context: {row['context']}.\n\n
                    [{lang}]: {self.extra_sentences[lang][0]}.\n\n
                    [{lang}]: {self.extra_sentences[lang][1]}.\n\n
                    [{lang}]: {self.extra_sentences[lang][2]}.\n\n
                    Translate this sentence into [{lang}]: {row['eng']}
                """
                translation = invoke_llm(prompt, self.model)
                self.sampled_df.at[index, f"{langCode}_translation"] = translation
                # prevent llm throttling
                time.sleep(2)

        self.sampled_df.to_csv(f"{self.model.name}_data.csv", index=False)

        return self
