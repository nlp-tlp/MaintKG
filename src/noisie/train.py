"""NoisIE training script."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import jsonlines
import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)
from transformers.optimization import AdamW

from noisie.config import ModelConfig
from noisie.scheduler import get_inverse_square_root_schedule_with_warmup
from noisie.utils import shift_tokens_left

config = ModelConfig()
random.seed(config.seed)

arg_to_scheduler = {
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup,
}


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    model_name: str = "google/byt5-small"
    max_length: int = 512
    batch_size: int = 8
    num_epochs: int = 600
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    val_check_interval: int = 5
    grad_clip_val: float = 10
    grad_accum_steps: int = 1
    train_test_split: float = 0.95
    seed: int = 42
    num_beams: int = 1
    pad_token_id: int = 0
    experiment_name: str = "noisie_training"


class TranslationDataset(Dataset):
    """Dataset for sequence-to-sequence translation tasks."""

    def __init__(self, encodings: Dict[str, List]) -> None:
        """Initialise the dataset with a dictionary of encodings."""
        self.encodings = encodings

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.encodings["input_ids"])


class ByT5Model(LightningModule):
    """PyTorch Lightning module for ByT5 model."""

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
    ) -> None:
        """Initialise the model with the pretrained T5 model and tokenizer."""
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        # self.output_dir = output_dir

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

    def generate(self, input_ids, attention_mask, **generate_kwargs):
        return self.model.generate(
            input_ids, attention_mask=attention_mask, **generate_kwargs
        )

    def forward(self, inputs, labels) -> dict:
        outputs = self.model(input_ids=inputs["input_ids"], labels=labels)
        logits = outputs["logits"]
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        output_dict = {"loss": loss, "logits": logits}

        return output_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        labels_original = labels.clone()

        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        # labels = shift_tokens_left(labels, -100)
        labels[labels == self.config.pad_token_id] = -100

        forward_output = self.forward(batch, labels)
        self.log("loss", forward_output["loss"])
        batch["labels"] = labels_original
        # self.train_step_outputs.append(forward_output["loss"])

        # print(f'Train step loss: {forward_output["loss"]}')

        return forward_output["loss"]

    def validation_step(self, batch: dict):
        gen_kwargs = {
            "max_length": self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            # "length_penalty": 0,
            "num_beams": 1,
        }
        generated_tokens = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **gen_kwargs,
        )
        decoded_preds = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        print("\n Predictions:")
        for pred in decoded_preds:
            print(pred, "\n")

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(
            labels != -100, labels, self.config.pad_token_id
        )
        labels = shift_tokens_left(labels, -100)

        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output["loss"] = forward_output["loss"].mean().detach()

        self.log("val_loss", float(forward_output["loss"]))

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,  # self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (0.9, 0.999),
            "eps": 0.00000001,
            # "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
            # "eps": self.hparams.adam_epsilon,
        }

        optimizer_kwargs["lr"] = 0.00005  # self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(
            1200000, optimizer  # self.hparams.max_steps,
        )

        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[
            "inverse_square_root"
            # self.hparams.lr_scheduler
        ]
        scheduler = schedule_func(
            optimizer,
            num_warmup_steps=1000,
            #   num_warmup_steps=self.hparams.warmup_steps
        )
        return scheduler


def load_dataset(
    file_path: Union[str, Path], tokenizer: PreTrainedTokenizer, config: TrainingConfig
) -> Tuple[TranslationDataset, TranslationDataset]:
    """Load and prepare datasets for training and validation."""
    logger.info(f"Loading dataset from {file_path}")

    with jsonlines.open(file_path, "r") as reader:
        data = list(reader)

    random.shuffle(data)

    inputs = [item["input"] for item in data]
    targets = [item["output"] for item in data] if "output" in data[0] else None

    # Split the data
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs, targets, train_size=config.train_test_split, random_state=config.seed
    )

    def tokenize_data(texts: List[str]) -> Dict[str, List]:
        """Tokenize a list of texts."""
        return tokenizer(
            texts,
            max_length=config.max_length,
            padding=True,
            truncation=True,
        )

    train_encodings = tokenize_data(inputs_train)
    val_encodings = tokenize_data(inputs_val)

    # Tokenize targets if available
    if targets:
        with tokenizer.as_target_tokenizer():
            train_labels = tokenize_data(targets_train)
            val_labels = tokenize_data(targets_val)

        train_encodings["labels"] = train_labels["input_ids"]
        val_encodings["labels"] = val_labels["input_ids"]

    logger.info(
        f"Created dataset with {len(train_encodings['input_ids'])} training and "
        f"{len(val_encodings['input_ids'])} validation examples"
    )

    return TranslationDataset(train_encodings), TranslationDataset(val_encodings)


def setup_tb_logging_dir(base_dir: Path) -> Path:
    """Set up tensorboard directory structure."""
    log_dir = base_dir / "tb_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging directory set up at {log_dir}")
    return log_dir


def main() -> None:
    """Train the ByT5 model."""
    # Initialise config
    config = TrainingConfig()
    pl.seed_everything(config.seed)

    # Set up directory structure
    base_dir = Path(__file__).resolve().parent
    log_dir = setup_tb_logging_dir(base_dir)

    # Initialise tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(
        config.model_name,
        config=AutoConfig.from_pretrained(
            config.model_name,
            decoder_start_token_id=0,
            early_stopping=False,
            no_repeat_ngram_size=0,
            dropout=0.1,
            forced_bos_token_id=None,
        ),
    )

    # Load datasets
    data_path = base_dir / "data" / "maintnormie.jsonl"  # Update with your data path
    train_dataset, val_dataset = load_dataset(data_path, tokenizer, config)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialise model
    pl_model = ByT5Model(model, tokenizer, config)

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            filename="{epoch}-{val_loss:.2f}",
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Initialise logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config.experiment_name,
        default_hp_metric=False,
    )

    # Initialise trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        logger=tb_logger,
        check_val_every_n_epoch=config.val_check_interval,
        gradient_clip_val=config.grad_clip_val,
        accumulate_grad_batches=config.grad_accum_steps,
        deterministic=True,
        default_root_dir=base_dir,
    )

    # Log some initial information
    # logger.info(f"Training logs will be stored in: {log_dir}")
    # logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"TensorBoard logs directory: {tb_logger.log_dir}")

    # Train model
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
