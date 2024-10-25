"""Settings."""

import random
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from maintkg.utils.path_utils import get_cache_file, get_input_dir, get_output_dir


def skewed_random_number(min_val: int, max_val: int) -> int:
    """
    Generate a skewed random number within a specified range.

    This function generates a random integer between `min_val` and `max_val`
    using a triangular distribution. The mode of the distribution is set to
    be at the 5% mark between `min_val` and `max_val`, creating a skew towards
    the lower end of the range.
    """
    mode_val = min_val + 0.05 * (max_val - min_val)
    return int(random.triangular(min_val, max_val, mode_val))


def generate_random_date(min_year: int, max_year: int) -> str:
    """Generate a random date.

    Generate a random date between `min_year` and `max_year` in the format
    "DD/MM/YYYY".
    """
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(min_year, max_year)
    random_date = datetime(year, month, day)
    return random_date.strftime("%d/%m/%Y")


class InputConfig(BaseModel):
    """Input configuration model."""

    csv_filename: str = Field(
        ...,
        description="Filename of the input CSV file containing maintenance work order data.",
    )
    id_col: str = Field(..., description="Column name for the unique identifier")
    type_col: str = Field(..., description="Column name for the order type")
    text_col: str = Field(
        ...,
        description="Column name for the order description (maintenance short text)",
    )
    floc_col: str = Field(
        ..., description="Column name for the functional location (asset) code"
    )
    start_date_col: Optional[str] = Field(
        default=None, description="Column name for the order start date"
    )
    cost_col: Optional[str] = Field(
        default=None, description="Column name for the order cost"
    )
    time_col: Optional[str] = Field(
        default=None, description="Column name for the order duration"
    )
    unplanned_type_codes: List[str] = Field(
        default_factory=list,
        description="List of order type codes that indicate unplanned work",
    )
    planned_type_codes: List[str] = Field(
        default_factory=list,
        description="List of order type codes that indicate planned work",
    )
    date_format: str = Field(
        ...,
        description="Date format in the input CSV file",
    )
    non_semantic_codes: List[str] = Field(
        default_factory=list,
        description="Codes that are site/organisation specific",
    )


class ProcessingConfig(BaseModel):
    """Processing configuration model."""

    id_col: str = Field(
        default="id",
        description="Column name that is used to map to the unique identifier",
    )
    type_col: str = Field(
        default="type", description="Column name that is used to map to the order type"
    )
    text_col: str = Field(
        default="text",
        description="Column name that is used to map to the order description",
    )
    floc_col: str = Field(
        default="floc",
        description="Column name that is used to map to the functional location code",
    )
    start_date_col: str = Field(
        default="start_date",
        description="Column name that is used to map to the order start date",
    )
    cost_col: str = Field(
        default="cost", description="Column name that is used to map to the order cost"
    )
    time_col: str = Field(
        default="time",
        description="Column name that is used to map to the order duration",
    )
    date_format: str = Field(
        default="%Y-%m-%d", description="Standard date format for working with dates"
    )
    add_dummy_cols: bool = Field(
        default=False,
        description="Adds dummy columns if the user does not supply start_date, cost or time.",
    )
    merge_nary_assertions: bool = Field(
        default=False, description="Merge n-ary assertions/triples"
    )
    systems: Optional[List[str]] = Field(
        default=None, description="List of system names to filter records by"
    )
    unplanned_only: bool = Field(
        default=False, description="Only include unplanned work orders"
    )
    use_gazetteer: bool = Field(
        default=True, description="Use entity gazetteer for fine-grained typing"
    )
    remove_subsumption: bool = Field(
        default=False, description="Remove subsumption (is a) relations"
    )
    add_system_nodes: bool = Field(
        default=True,
        description="Add `System` nodes that link to `Records` nodes via the `HAS_RECORDS` relationship.",
    )


class Neo4jConfig(BaseModel):
    """Neo4j configuration model."""

    uri: str = Field("bolt://localhost:7687", description="Neo4j URI")
    username: str = Field("neo4j", description="Neo4j username")
    password: str = Field("password", description="Neo4j password")
    database: str = Field("neo4j", description="Neo4j database")


class NoisieConfig(BaseModel):
    """NoisIE configuration model."""

    batch_size: int = Field(
        default=256, description="Batch size for the IE model inference API"
    )
    cache_dir: str = Field(
        default="./cache", description="Directory to cache IE model predictions"
    )
    cache_filename: str = Field(
        default="cache.db", description="Filename for the IE model cache"
    )


class DevConfig(BaseModel):
    """Development configuration model."""

    dev: bool = Field(default=False, description="Run MaintKG in development mode")
    limit: Optional[int] = Field(
        default=None, description="Limit the number of records (used for testing)"
    )


class Settings(BaseSettings):
    """Main settings class that combines all configuration models."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="",
        env_ignore_empty=True,
    )

    input: InputConfig
    processing: ProcessingConfig
    neo4j: Neo4jConfig
    noisie: NoisieConfig
    dev: DevConfig

    @property
    def cache_file(self: Self) -> str:
        """Get the cache file."""
        return get_cache_file(filename=self.noisie.cache_filename)

    @property
    def output_dir(self: Self) -> Path:
        """Get the output directory path."""
        return get_output_dir()

    @property
    def input_dir(self: Self) -> Path:
        """Get the input directory path."""
        return get_input_dir()

    def get_output_subdir(self: Self, subdir: Optional[str] = None) -> Path:
        """Get an output subdirectory path."""
        path = self.output_dir
        if subdir:
            path = path / subdir
            path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def cols_to_keep(self: Self) -> List[str]:
        """Create a list of columns to keep from the input CSV file after its mapped."""
        cols = [
            self.processing.id_col,
            self.processing.type_col,
            self.processing.text_col,
            self.processing.floc_col,
        ]

        if self.input.start_date_col is not None:
            cols.append(self.processing.start_date_col)
        if self.input.cost_col is not None:
            cols.append(self.processing.cost_col)
        if self.input.time_col is not None:
            cols.append(self.processing.time_col)

        return cols

    @property
    def col_mapping(self: Self) -> Dict[str, str]:
        """Create a dictionary mapping input columns to output columns."""
        mapping = {
            self.input.id_col: self.processing.id_col,
            self.input.type_col: self.processing.type_col,
            self.input.text_col: self.processing.text_col,
            self.input.floc_col: self.processing.floc_col,
        }

        if self.input.start_date_col is not None:
            mapping[self.input.start_date_col] = self.processing.start_date_col
        if self.input.cost_col is not None:
            mapping[self.input.cost_col] = self.processing.cost_col
        if self.input.time_col is not None:
            mapping[self.input.time_col] = self.processing.time_col

        return mapping

    @property
    def dummy_cols(self: Self) -> Dict[str, Callable[[], Union[str, int]]]:
        """Returns a dictionary mapping column names to callables that generate random data."""
        cols: Dict[str, Callable[[], Union[str, int]]] = {}

        if self.input.start_date_col is None:
            cols[self.processing.start_date_col] = lambda: generate_random_date(
                2000, 2020
            )
        if self.input.cost_col is None:
            cols[self.processing.cost_col] = lambda: skewed_random_number(0, 50000)
        if self.input.time_col is None:
            cols[self.processing.time_col] = lambda: skewed_random_number(0, 12 * 7)
        return cols
