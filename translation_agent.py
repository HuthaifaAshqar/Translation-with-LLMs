import json
import os
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from loguru import logger
from tqdm import tqdm

from constants import SupportedModel
from utils import invoke_llm


class TranslationAgent:
    def __init__(
        self,
        model: SupportedModel = SupportedModel.GEMINI,
        base_language: Optional[Tuple[str, str]] = ("English", "eng"),
        languages_code: Optional[Dict[str, str]] = None,
        dataset_name: str = "openlanguagedata/flores_plus",
        api_key_env_var: str = "HF_API_KEY",
        sample_data_path: Optional[str] = None,
        extra_sentences_path: Optional[str] = None,
    ):
        """
        Initializes the TranslationAgent with a base language and target languages.

        :param model: model to use for translation. default GEMINI.
        :param base_language: tuple of (base language, ISO 639-3 code) for the base language.
        :param languages_code: Dictionary mapping language names to their ISO 639-3 codes.
        :param dataset_name: Name of the dataset to load.
        :param api_key_env_var: Environment variable name for the API key.
        :param sample_data_path: Path to an existing sample_data.csv file.
        :param extra_sentences_path: Path to an existing extra_sentences JSON file.
        """
        # Authenticate using the provided API key
        login(os.environ.get(api_key_env_var))

        # Load the dataset
        self.dataset = load_dataset(dataset_name)

        # Initialize or load sampled DataFrame
        if sample_data_path and os.path.isfile(sample_data_path):
            self.sampled_df = pd.read_csv(sample_data_path)
            logger.info(f"Loaded sampled data from {sample_data_path}")
        else:
            self.sampled_df = pd.DataFrame([])

        # Initialize or load extra sentences
        if extra_sentences_path and os.path.isfile(extra_sentences_path):
            with open(extra_sentences_path, encoding="utf-8") as f:
                self.extra_sentences = json.load(f)
            logger.info(f"Loaded extra sentences from {extra_sentences_path}")
        else:
            self.extra_sentences = defaultdict(list)

        self.model = model

        # Set language codes
        if languages_code is None:
            self.languages_code = {"French": "fra", "Arabic": "arb", "Azerbaijani": "azj"}
        else:
            self.languages_code = languages_code

        # Set the base language code
        self.base_language_name = base_language[0]
        self.base_language_code = base_language[1]

    def select_model(self, model: SupportedModel):
        """
        Selects the model to be used for translation.

        :param model: An instance of SupportedModel.
        :return: self
        """
        logger.info(f"Selecting {model.name} model")
        self.model = model
        return self

    def run(self):
        (self.baseline_translation().generate_context().translate_context().translate_qa().translate_with_context())
        return self

    def sample(self, n: int = 1):
        """
        Samples sentences from the dataset based on the base language and prepares translations.

        :param n: Number of samples to draw.
        :return: self
        """
        devtest_split = self.dataset["devtest"]

        # Filter sentences in the base language
        base_sentences = devtest_split.filter(lambda x: x["iso_639_3"] == self.base_language_code)
        logger.info(f"Filtered base language sentences: {self.base_language_name}")

        # Sample n sentences
        sampled_base = base_sentences.shuffle(seed=42).select(range(n))
        sampled_ids = [s["id"] for s in sampled_base]
        logger.info(f"Sampled {n} sentences.")

        # Filter relevant translations for the sampled IDs and target languages
        relevant_translations = devtest_split.filter(
            lambda x: x["id"] in sampled_ids and x["iso_639_3"] in self.languages_code.values()
        )

        # Define columns: base language, context, target languages, and their translations
        target_lang_codes = list(self.languages_code.values())
        cols = [self.base_language_code, "context"] + target_lang_codes
        df = pd.DataFrame(columns=cols)
        logger.info(f"Initialized DataFrame with columns: {cols}")

        for sample in sampled_base:
            _id = sample["id"]
            base_text = sample["text"]

            row_data = {self.base_language_code: base_text}

            for lang_name, lang_code in self.languages_code.items():
                # Fetch the translation for each target language
                if lang_name.lower() == "arabic":
                    # Ensure the script is Arabic
                    translations = relevant_translations.filter(
                        lambda x: x["id"] == _id and x["iso_639_3"] == lang_code and x.get("iso_15924") == "Arab"
                    )
                else:
                    translations = relevant_translations.filter(
                        lambda x: x["id"] == _id and x["iso_639_3"] == lang_code
                    )

                if len(translations) > 0:
                    translated_text = translations["text"][0]
                else:
                    translated_text = ""
                    logger.warning(f"No translation found for language: {lang_name} (Code: {lang_code}) and ID: {_id}")

                row_data[lang_code] = translated_text

            df = pd.concat([df, pd.DataFrame(row_data, index=[0])], ignore_index=True)

        # Sample additional sentences for each target language
        extra_sentences = defaultdict(list)
        for lang_name, lang_code in self.languages_code.items():
            # Define additional filters for specific languages
            if lang_name.lower() == "arabic":
                # Ensure the script is Arabic
                def lang_filter(x):
                    return x["iso_639_3"] == lang_code and x["id"] not in sampled_ids and x.get("iso_15924") == "Arab"

            else:

                def lang_filter(x):
                    return x["iso_639_3"] == lang_code and x["id"] not in sampled_ids

            # Filter sentences based on the defined filter
            lang_sentences = devtest_split.filter(lang_filter).shuffle(seed=42).select(range(3))
            extra_sentences[lang_name] = [s["text"] for s in lang_sentences]
            logger.info(f"Sampled 3 extra sentences for language: {lang_name}")

        self.sampled_df = df
        self.sampled_df.to_csv(f"sample_{n}.csv", index=False)
        logger.info(f"Saved sampled data to sample_{n}.csv")

        self.extra_sentences = extra_sentences

        # Save the extra_sentences dictionary to a file if a path is provided
        extra_sentences_save_path = "extra_sentences.json"
        with open(extra_sentences_save_path, "w", encoding="utf-8") as f:
            json.dump(self.extra_sentences, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved extra sentences to {extra_sentences_save_path}")

        return self

    def baseline_translation(self):
        for index, row in tqdm(self.sampled_df.iterrows(), desc="baseline translation", total=self.sampled_df.shape[0]):
            for lang_name, lang_code in self.languages_code.items():
                prompt = f"""
                        [{self.base_language_name}] {row[self.base_language_code]}.\n
                        [{lang_name}]
                        """
                translation = invoke_llm(prompt, self.model)
                self.sampled_df.at[index, f"{lang_code}_base_translation"] = translation
                time.sleep(2)  # To prevent throttling

        self.sampled_df.to_csv(f"{self.model.name}_data.csv", index=False)

        return self

    def generate_context(self):
        """
        Generates context sentences for each sampled base sentence using the selected model.

        :return: self
        """
        for index, row in tqdm(self.sampled_df.iterrows(), desc="Generate context", total=self.sampled_df.shape[0]):
            prompt = (
                f"Write a paragraph containing the following sentence:"
                f"\n[{self.base_language_name}]: {row[self.base_language_code]}"
            )
            context_sentence = invoke_llm(prompt, self.model)
            self.sampled_df.at[index, "context"] = context_sentence
            time.sleep(2)  # To prevent throttling

        self.sampled_df.to_csv(f"{self.model.name}_data.csv", index=False)
        return self

    def translate_context(self):
        """
        :return: self
        """
        for index, row in tqdm(self.sampled_df.iterrows(), desc="Translate context", total=self.sampled_df.shape[0]):
            for lang_name, lang_code in self.languages_code.items():
                extra_texts = self.extra_sentences.get(lang_name, [""] * 3)
                prompt = f"""
                        [{lang_name}]: {extra_texts[0]}.\n
                        [{lang_name}]: {extra_texts[1]}.\n
                        [{lang_name}]: {extra_texts[2]}.\n
                        Translate this paragraph from {self.base_language_name} to {lang_name}:\n
                        [{self.base_language_name}] {row['context']}
                        """
                translation = invoke_llm(prompt, self.model)
                self.sampled_df.at[index, f"{lang_code}_context"] = translation
                time.sleep(2)  # To prevent throttling

        self.sampled_df.to_csv(f"{self.model.name}_data.csv", index=False)

        return self

    def translate_qa(self):
        for index, row in tqdm(self.sampled_df.iterrows(), desc="LLM MCQA", total=self.sampled_df.shape[0]):
            for lang_name, lang_code in self.languages_code.items():
                prompt = f"""
                        Identify the sentences that are translations of the following
                        {self.base_language_name} sentence.
                        You may select multiple answers.
                        Respond only with choice numbers separated by commas, without any extra text.
                        [{self.base_language_name}] {row[self.base_language_code]}.
                        """
                translated_context = row[f"{lang_code}_context"]
                choices = translated_context.lstrip().rstrip().split(".")
                for idx, choice in enumerate(choices):
                    prompt += f"\n{idx+1}. {choice}."

                translation = invoke_llm(prompt, self.model)
                self.sampled_df.at[index, f"{lang_code}_mcqa"] = translation
                time.sleep(2)  # To prevent throttling

        self.sampled_df.to_csv(f"{self.model.name}_data.csv", index=False)

        return self

    def translate_with_context(self):
        """
        Translates each base sentence into the target languages using the generated context.

        :return: self
        """
        for index, row in tqdm(
            self.sampled_df.iterrows(), desc="Translate with context", total=self.sampled_df.shape[0]
        ):
            for lang_name, lang_code in self.languages_code.items():
                extra_texts = self.extra_sentences.get(lang_name, [""] * 3)
                prompt = f"""
                        Context: {row['context']}.\n
                        [{lang_name}]: {extra_texts[0]}.\n
                        [{lang_name}]: {extra_texts[1]}.\n
                        [{lang_name}]: {extra_texts[2]}.\n
                        Translate this sentence into [{lang_name}]: {row[self.base_language_code]}
                        """
                translation = invoke_llm(prompt, self.model)
                self.sampled_df.at[index, f"{lang_code}_translate_with_context"] = translation
                time.sleep(2)  # To prevent throttling

        self.sampled_df.to_csv(f"{self.model.name}_data.csv", index=False)

        return self
