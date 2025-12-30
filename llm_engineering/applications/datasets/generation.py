from abc import ABC
from huggingface_hub import autotokenizer

from llm_engineering.domains.prompt import Prompt

class DatasetGeneration(ABC):
    tokenizer = autotokenizer()

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
        Provide your response in JSON format.
    """

    promp_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        dataset_format = (
            "instruction-answer pairs"
        )
        input_variables = {
            "dataset_format": dataset_format
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt
        )

    @classmethod
    def get_prompts(cls, documents: list[Document])