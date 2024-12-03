from dotenv import load_dotenv

from constants import SupportedModel
from translation_agent import TranslationAgent

if __name__ == "__main__":
    # Load .env file
    load_dotenv()

    # make_model_rhyme(SupportedModel.GEMINI)
    # inspect_flores200()

    translator = TranslationAgent(sample_data_path="sample_100.csv", extra_sentences_path="extra_sentences.json")
    (
        translator
        # .sample(n=100)
        .select_model(SupportedModel.GEMINI)
        .run()
        .select_model(SupportedModel.GPT4O)
        .run()
        .select_model(SupportedModel.LLAMA3)
        .run()
    )
