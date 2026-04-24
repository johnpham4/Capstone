import json
from loguru import logger

from src.config.settings import settings

from src.services.generators import DSLGeneratorFactory
from src.prompts import DSL_INFERENCE_INSTRUCTION


def main():
    print(f"Provider: {settings.LLM_PROVIDER}")

    generator = DSLGeneratorFactory.create()

    user_input = "Cho tam giác ABC vuông tại A, AB = AC"


    try:
        dsl = generator.generate_dsl(
            user_input=user_input,
            dsl_prompt=DSL_INFERENCE_INSTRUCTION,
            clean_problem=True,
        )

        print("\n--- RESULT ---\n")

        if dsl:
            print(dsl)

    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    main()