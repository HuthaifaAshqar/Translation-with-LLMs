from dotenv import load_dotenv

from constants import SupportedModel
from translation_agent import TranslationAgent

if __name__ == "__main__":
    # Load .env file
    load_dotenv()

    # make_model_rhyme(SupportedModel.GEMINI)
    # inspect_flores200()

    translator = TranslationAgent()
    (
        translator.sample(n=1)
        .select_model(SupportedModel.GEMINI)
        .generate_context()
        .translate_with_context()
        .select_model(SupportedModel.GPT4O)
        .generate_context()
        .translate_with_context()
        .select_model(SupportedModel.LLAMA3)
        .generate_context()
        .translate_with_context()
    )
