from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loguru import logger
import json

from llm_engineering.applications.training.callbacks.logger import LossLoggingCallback
from llm_engineering.domains.training_config import TrainingConfig


class ModelTrainer:
    def __init__(self, config: TrainingConfig):

        self.config = config
        self.model = None
        self.tokenizer = None


    def load_model(self):
        logger.info(f"Loading LLM from: {self.config.base_llm_path}")

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

        # Prepare model for k-bit training
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

        logger.success("Model loaded and LoRA applied successfully")
        self.model.print_trainable_parameters()

    def _preprocess_dataset(self, raw_data: list[dict], images_dir: str):
        """
        Add image paths and tokenize if needed.
        """
        for sample in raw_data:
            sample['images_dir'] = images_dir
        return Dataset.from_list(raw_data)

    def load_dataset(self, train_path: str, images_dir: str, eval_path: str = None):
        logger.info(f"Loading training dataset from: {train_path}")
        with open(train_path, encoding='utf-8') as f:
            train_data = json.load(f)
        train_dataset = self._preprocess_dataset(train_data, images_dir)
        logger.success(f"Loaded {len(train_dataset)} training samples")

        eval_dataset = None
        if eval_path:
            logger.info(f"Loading evaluation dataset from: {eval_path}")
            with open(eval_path, encoding='utf-8') as f:
                eval_data = json.load(f)
            eval_dataset = self._preprocess_dataset(eval_data, images_dir)
            logger.success(f"Loaded {len(eval_dataset)} evaluation samples")

        return train_dataset, eval_dataset


    def train(self, train_dataset, eval_dataset=None, resume_from_checkpoint=None):
        logger.info("Initializing Trainer")

        trainer = Trainer(
            model=self.model,
            args=self._get_training_args(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator="",
            callbacks=[LossLoggingCallback()],
        )

        logger.info("Starting training")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        logger.success("Training completed")

        return trainer

    def save_model(self, path=None):
        path = path or self.config.output_dir
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.success(f"Model and tokenizer saved to {path}")

    def _get_training_args(self):
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            weight_decay=0.01,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fp16=self.config.fp16,
            logging_steps=getattr(self.config, "logging_steps", 10),
            logging_strategy="steps",
            save_steps=getattr(self.config, "save_steps", 500),
            save_total_limit=getattr(self.config, "save_total_limit", 2),
            remove_unused_columns=False,
            report_to="none",
        )