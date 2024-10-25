"""NoisIE configuration."""

from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field

base_dir = Path(__file__).resolve().parent


class ModelConfig(BaseModel):
    """Configuration for the NoisIE model."""

    data_path: Path = Field(
        default=base_dir / "data/maintnormie.jsonl",
        description="The path to the MaintNormIE dataset.",
    )
    model_name_or_path: str = Field(
        default="google/byt5-small", description="The T5 variants model name."
    )
    max_length: int = Field(
        default=512, ge=1, description="The maximum length of the input sequence."
    )
    ckpt_path: Optional[Union[str, Path]] = None
    seed: int = 1337

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())
