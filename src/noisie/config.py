"""NoisIE configuration."""

from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the NoisIE model."""

    data_path: str
    model_name: str
    max_length: int = Field(
        512, ge=1, description="The maximum length of the input sequence."
    )
    ckpt_path: Optional[str] = None
    seed: int = 1337
