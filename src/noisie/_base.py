"""NoisIE model."""

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
)
from typing_extensions import Self

from noisie.config import ModelConfig
from noisie.model_data import ByT5Model, TranslationDataset, structure_noisie_string

base_dir = Path(__file__).resolve().parent

# Type aliases for clarity
BatchType = Dict[str, torch.Tensor]
InferenceOutput = Dict[str, Any]


class NoisIE:
    """NoisIE model for text processing and inference."""

    def __init__(
        self: "NoisIE", batch_size: int = 32, device: Optional[str] = None
    ) -> None:
        """Initialize the NoisIE model.

        Parameters
        ----------
        batch_size: Number of samples per batch
        device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        self.batch_size = batch_size
        self.model: ByT5Model
        self.tokenizer: PreTrainedTokenizer
        self.config = ModelConfig(
            ckpt_path=base_dir  # type: ignore
            / "./lightning_logs/version_22_512_final_maintnormie/checkpoints/epoch=599-step=498600.ckpt",
        )
        self.model_config = self._create_model_config()
        self.device = self._setup_device(device)
        self.load()

    def _create_model_config(self) -> AutoConfig:
        """Create the model configuration."""
        return AutoConfig.from_pretrained(
            self.config.model_name_or_path,
            decoder_start_token_id=0,
            early_stopping=False,
            no_repeat_ngram_size=0,
            dropout=0.1,
            forced_bos_token_id=None,
            max_length=self.config.max_length,
        )

    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Set up the compute device for the model.

        Args:
            device: Requested device ('cuda', 'cpu', or None for auto-detection)

        Returns:
            torch.device: The selected compute device
        """
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            logger.debug("CUDA is available. Using GPU.")
            return torch.device("cuda")

        logger.debug("CUDA not available. Using CPU.")
        return torch.device("cpu")

    def load(self) -> None:
        """Load the model and tokenizer."""
        try:
            base_model = T5ForConditionalGeneration.from_pretrained(
                self.config.model_name_or_path
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name_or_path, use_fast=True
            )

            self.model = ByT5Model.load_from_checkpoint(
                model=base_model,
                config=self.model_config,
                tokenizer=self.tokenizer,
                checkpoint_path=self.config.ckpt_path,
                output_dir="./",
            )

            self.model.eval()
            self.model.to(self.device)
            self.model.freeze()

            logger.debug("Model loaded and ready for inference.")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError("Model initialization failed") from e

    def prepare_dataset(self, data: List[str]) -> DataLoader:
        """Prepare the dataset for inference.

        Parameters
        ----------
        data: List of input strings to process

        Returns
        -------
        DataLoader: Prepared data loader for batch processing
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized. Call load() first.")

        encoded_data = self.tokenizer(
            data,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
        )

        dataset = TranslationDataset(encodings=encoded_data)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )

    def _process_single_output(
        self, input_text: str, prediction: str, batch_idx: int, item_idx: int
    ) -> InferenceOutput:
        """Process a single model output.

        Parameters
        ----------
        input_text: Original input text
        prediction: Model's prediction
        batch_idx: Batch index for error reporting
        item_idx: Item index for error reporting

        Returns
        -------
        Dict containing processed output and metadata
        """
        output: InferenceOutput = {
            "input": input_text,
            "gen_output": prediction,
            "issues": {},
            "norms": [],
            "entities": [],
            "relations": [],
            "passed": True,
        }

        try:
            structured_pred = structure_noisie_string(
                input_str=input_text, noisie_str=prediction
            )

            output["issues"] = structured_pred["issues"]
            for field in ["norms", "entities", "relations"]:
                output[field] = structured_pred[field]

        except Exception as e:
            logger.error(
                f"Failed to structure prediction for batch {batch_idx}, item {item_idx}: {str(e)}"
            )
            output["passed"] = False
            output["error"] = str(e)

        return output

    def inference(self: Self, data: List[str]) -> List[InferenceOutput]:
        """Run inference on the input data.

        Parameters
        ----------
        data: List of input strings to process

        Returns
        -------
        List of processed outputs with predictions and metadata
        """
        if not self.model:
            raise RuntimeError("Model not initialized. Call load() first.")

        dataloader = self.prepare_dataset(data)
        outputs: List[InferenceOutput] = []

        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                inputs, preds = self.model.generate_samples(batch)

                if len(inputs) != len(preds):
                    logger.error(
                        f"Batch {batch_idx}: Misaligned inputs and predictions"
                    )
                    logger.error(f"Inputs: {len(inputs)}, Predictions: {len(preds)}")
                    raise ValueError(
                        f"Input length {len(inputs)} doesn't match predictions length {len(preds)}"
                    )

                for item_idx, (input_text, pred) in enumerate(zip(inputs, preds)):
                    output = self._process_single_output(
                        input_text=input_text,
                        prediction=pred,
                        batch_idx=batch_idx,
                        item_idx=item_idx,
                    )
                    outputs.append(output)

            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                logger.debug(traceback.format_exc())
                # Add a failed output entry to maintain data alignment
                outputs.append(
                    {"passed": False, "error": str(e), "batch_idx": batch_idx}
                )

        return outputs


if __name__ == "__main__":
    import json

    noisie = NoisIE()

    data = ["rplcaa eng oil$$$$$"]

    preds = noisie.inference(data)

    print(json.dumps(preds, indent=2))
