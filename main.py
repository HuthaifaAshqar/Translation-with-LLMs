import argparse
import sys

from dotenv import load_dotenv
from loguru import logger

from constants import SupportedModel
from utils import inspect_flores200, test_api_call

if __name__ == "__main__":
    # Load .env file
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[m.name for m in SupportedModel],
        default=SupportedModel.GEMINI,
        help="API model to use",
    )
    args = parser.parse_args()

    try:
        selected_model = SupportedModel[args.model]
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    test_api_call(selected_model)
    inspect_flores200()
