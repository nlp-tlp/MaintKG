"""MaintKG construction system for maintenance work orders.

This module provides the entry point for building a MaintKG from
maintenance work order data. It integrates with Neo4j for graph storage and
uses configurable settings for data processing.

Example:
    $ python ./src/maintkg/main.py
"""

from maintkg.builder import GraphBuilder
from maintkg.settings import Settings


def main() -> None:
    """Construct a MaintKG from maintenance work order data.

    This function initializes the graph builder with predefined settings and
    executes the graph construction process. It supports both development and
    production configurations through environment-specific settings.
    """
    settings = Settings(
        csv_filename="case_study_pdl.csv",
        input_id_col="id",
        input_type_col="PMType",
        input_text_col="OriginalShorttext",
        input_floc_col="Asset",
        input_start_date_col="BscStartDate",
        input_cost_col="Cost",
        unplanned_type_codes=["PM01", "PM03"],
        planned_type_codes=["PM02"],
        non_semantic_codes=["WARR", "warranty", "CHUBB", "Chubb", "chubb"],
        input_date_format="%Y-%m-%d",
        dev=True,
        # limit=64,
        add_dummy_cols=False,
        # systems=["O&K RH170 Excavator"],
    )

    if settings.dev:
        settings.NEO4J_DATABASE = "github"

    builder = GraphBuilder(settings=settings, cleanup=False)
    builder.create()


if __name__ == "__main__":
    main()
