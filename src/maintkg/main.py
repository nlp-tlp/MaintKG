"""MaintKG construction system for maintenance work orders.

This module provides the entry point for building a MaintKG from
maintenance work order data. It integrates with Neo4j for graph storage and
uses configurable settings for data processing.

Example:
    $ python ./src/maintkg/main.py
"""

from maintkg.builder import GraphBuilder


def main() -> None:
    """Construct a MaintKG from maintenance work order data.

    This function initializes the graph builder with predefined settings and executes the graph construction process. It supports both development and production configurations through environment-specific settings.
    """
    builder = GraphBuilder(cleanup=False)
    builder.create()


if __name__ == "__main__":
    main()
