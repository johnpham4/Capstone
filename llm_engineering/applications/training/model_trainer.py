from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import json
from pathlib import Path
from loguru import logger


class UnslothGeoTrainer:
    """Trainer tối ưu cho Geometry formalization với Unsloth"""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
    ):
        self.base_model = base_model
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model với Unsloth optimization"""
        logger.info(f"Loading {self.base_model} với Unsloth...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto detect
            load_in_4bit=self.load_in_4bit,
        )

        logger.info("Applying LoRA adapter...")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.lora_alpha,
            lora_dropout=0,  # Unsloth optimizes for 0 dropout
            bias="none",
            use_gradient_checkpointing="unsloth",  # Tiết kiệm 30% VRAM
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        logger.success("✅ Model loaded with Unsloth optimizations!")
        return self.model, self.tokenizer

    def prepare_dataset(self, json_path: str) -> Dataset:

        logger.info(f"Loading dataset from {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        formatted_data = []
        for item in data:
            # Format theo Qwen chat template
            text = (
                f"<|im_start|>system\n"
                f"Bạn là một trợ lý AI chuyên về hình học. "
                f"Nhiệm vụ của bạn là chuyển đổi mô tả hình học bằng tiếng Việt "
                f"sang định dạng formal geometry code.<|im_end|>\n"
                f"<|im_start|>user\n"
                f"<formalization>{item['instruction']}</formalization><|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"<answer>{item['answer']}</answer><|im_end|>"
            )
            formatted_data.append({"text": text})

        dataset = Dataset.from_list(formatted_data)
        logger.success(f"✅ Loaded {len(dataset)} samples")
        logger.info(f"📝 Sample:\n{dataset[0]['text'][:300]}...")

        return dataset

    def train(
        self,
        train_dataset: Dataset,
        output_dir: str = "outputs/qwen-geo-lora",
        num_epochs: int = 3,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        save_steps: int = 500,
    ):
        """Train model với optimized settings"""

        logger.info("🚀 Initializing SFTTrainer với Unsloth...")

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Set True nếu samples ngắn và đồng đều

            args=TrainingArguments(
                # Paths
                output_dir=output_dir,

                # Training hyperparams
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_epochs,

                # Learning rate
                learning_rate=learning_rate,
                lr_scheduler_type="cosine",
                warmup_steps=100,

                # Optimizer - 8bit Adam tiết kiệm VRAM
                optim="adamw_8bit",
                weight_decay=0.01,

                # Logging
                logging_steps=10,
                logging_strategy="steps",

                # Saving
                save_strategy="steps",
                save_steps=save_steps,
                save_total_limit=2,

                # Mixed precision
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),

                # Performance
                gradient_checkpointing=True,

                # Misc
                seed=42,
                report_to="none",  # Change to "wandb" for tracking
            ),
        )

        logger.info("🏋️ Starting training...")
        trainer_stats = trainer.train()

        logger.success("✅ Training completed!")
        logger.info(f"📊 Final loss: {trainer_stats.training_loss:.4f}")

        return trainer

    def save_model(
        self,
        output_dir: str = "outputs/qwen-geo-lora",
        merge: bool = True,
        merged_dir: str = "outputs/qwen-geo-merged",
    ):
        """Save LoRA adapter và optionally merge với base model"""

        # Save LoRA adapter
        logger.info(f"💾 Saving LoRA adapter to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.success(f"✅ LoRA adapter saved!")

        # Merge và save full model
        if merge:
            logger.info(f"🔄 Merging LoRA với base model...")
            self.model.save_pretrained_merged(
                merged_dir,
                self.tokenizer,
                save_method="merged_16bit",  # hoặc "merged_4bit"
            )
            logger.success(f"✅ Merged model saved to {merged_dir}")

    def inference(self, instruction: str, max_new_tokens: int = 256):
        """Test inference với trained model"""

        # Enable fast inference mode
        FastLanguageModel.for_inference(self.model)

        # Format prompt
        prompt = (
            f"<|im_start|>system\n"
            f"Bạn là một trợ lý AI chuyên về hình học.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<formalization>{instruction}</formalization><|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Generate
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )

        result = self.tokenizer.batch_decode(outputs)[0]
        return result


def main():
    """Main training script"""

    # =============== CONFIGURATION ===============
    CONFIG = {
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "train_data": "dataset/generated_gmbl/train.json",
        "output_dir": "outputs/qwen-geo-lora",
        "merged_dir": "outputs/qwen-geo-merged",

        # Model config
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "lora_r": 32,
        "lora_alpha": 64,

        # Training config
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "save_steps": 500,

        # Merge after training?
        "merge_model": True,
    }

    # =============== INITIALIZE TRAINER ===============
    trainer = UnslothGeoTrainer(
        base_model=CONFIG["base_model"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=CONFIG["load_in_4bit"],
        lora_r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
    )

    # =============== LOAD MODEL ===============
    trainer.load_model()

    # =============== PREPARE DATASET ===============
    train_dataset = trainer.prepare_dataset(CONFIG["train_data"])

    # =============== TRAIN ===============
    trainer.train(
        train_dataset=train_dataset,
        output_dir=CONFIG["output_dir"],
        num_epochs=CONFIG["num_epochs"],
        batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        save_steps=CONFIG["save_steps"],
    )

    # =============== SAVE MODEL ===============
    trainer.save_model(
        output_dir=CONFIG["output_dir"],
        merge=CONFIG["merge_model"],
        merged_dir=CONFIG["merged_dir"],
    )

    # =============== TEST INFERENCE ===============
    logger.info("🧪 Testing inference...")
    test_instruction = "Tam giác ABC, AB = AC, đường tròn O với đường kính BC"
    result = trainer.inference(test_instruction)
    logger.info(f"📝 Result:\n{result}")

    logger.success("🎉 All done!")


if __name__ == "__main__":
    main()
