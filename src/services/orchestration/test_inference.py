import json
from loguru import logger

from src.config.settings import settings

from src.services.generators import DSLGeneratorFactory
from src.prompts import DSL_INFERENCE_INSTRUCTION


def main():
    print(f"Provider: {settings.LLM_PROVIDER}")

    generator = DSLGeneratorFactory.create()

    user_input = "Cho tam giác ABC với góc ABC = 90°, O là tâm đường tròn ngoại tiếp tam giác ABC, N là điểm thuộc đoạn thẳng AB, AC là đường kính của đường tròn ngoại tiếp. Tính độ dài đoạn thẳng ON."


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