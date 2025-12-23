from transformers import TrainerCallback
from loguru import logger

class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            logger.info(
                f"step={state.global_step} "
                f"loss={logs['loss']:.4f} "
                f"lr={logs.get('learning_rate', 0):.2e}"
            )