from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import torch
from loguru import logger

from models.modeling_geomagvit import GeoMAGVIT
from llm_engineering.domains.training_config import TrainingConfig
from llm_engineering.applications.training.data_collator import T2DDataCollatorCached


class ModelTrainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.vq_model = None
        self.prompter = None

    def load_model(self):
        logger.info(f"Loading LLM: {self.config.base_llm_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_llm_path,
            load_in_8bit=self.config.use_8bit,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_llm_path,
            trust_remote_code=True,
        )

        logger.info(f"Loading VQ-VAE: {self.config.vq_model_path}")

        from models.prompting_utils import UniversalPrompting
        self.prompter = UniversalPrompting(
            self.tokenizer,
            max_len=4096,
            special_tokens=("<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>",
                          "<formalization>", "</formalization>", "<answer>", "</answer>"),
            ignore_id=-100
        )

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        logger.success("Model loaded successfully")
        self.model.print_trainable_parameters()

    def load_dataset(self, train_path: str, images_dir: str, eval_path: str = None):
        logger.info(f"Loading dataset from {train_path}")

        with open(train_path, encoding='utf-8') as f:
            train_data = json.load(f)

        for sample in train_data:
            sample['images_dir'] = images_dir

        train_dataset = Dataset.from_list(train_data)
        logger.success(f"Loaded {len(train_dataset)} training samples")

        eval_dataset = None
        if eval_path:
            with open(eval_path, encoding='utf-8') as f:
                eval_data = json.load(f)
            for sample in eval_data:
                sample['images_dir'] = images_dir
            eval_dataset = Dataset.from_list(eval_data)
            logger.info(f"Loaded {len(eval_dataset)} eval samples")

        return train_dataset, eval_dataset

    def train(self, train_dataset, eval_dataset=None):
        logger.info("Starting training with cached tokens")

        data_collator = T2DDataCollatorCached(prompter=self.prompter)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        logger.success("Training completed")

        return trainer
