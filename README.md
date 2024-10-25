# Maintenance Knowledge Graph (MaintKG)
**Maint**enance **K**nowledge **G**raph (MaintKG) is a graph-based knowledge representation and method for automatically constructing knowledge graphs from maintenance work order records and their short texts. This repository contains code for creating MaintKG from CMMS (Computerised Maintenance Management System) records.

## Table of Contents
- Getting started
- Prerequisites
- Installation
- Usage
- Contributing
- License
- Attribution

## Project Structure
```
maintkg/
├── cache/                  # Cached information extraction results
│   └── .gitkeep
├── input/                  # Input data directory
├── notebooks/             # Jupyter notebooks for analysis and examples
│   ├── assets/           # Notebook resources
│   └── example_queries.ipynb   # Queries that supplement the MaintKG chapter/paper competency questions
├── output/                # Generated outputs
│   ├── .gitkeep
│   └── YYYY-MM-DD_HH_MM-SS-MM/   # Generated output folders for MaintKG construction runs
├── src/
│   ├── maintkg/         # Core MaintKG package
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── settings.py
│   │   └── utils/
│   ├── noisie/          # Simultaneous normalisation and information extraction package
│   └── rebel/           # Relation extraction package
├── .git/
├── .gitignore
├── .pre-commit-config.yaml
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Getting Started
These instructions will help you get a copy of the project up and running on your local machine.

### Prerequisites

What things you need to install:

- Neo4J
- Python 3.9
- venv
- pytorch (cuda preferably)

### Installation

Need to install with `pip install -e .`

1. Clone the repository
```bash
git clone https://github.com/nl-tlp/maintkg.git
```

2. Create a virtual environment (optional)
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

To run the MaintKG pipeline, first ensure your CMMS data is in the `./input` directory, and you have specified your columns, etc., in the `Settings` within `./src/maintkg/main.py`. After this, run the following command:

```bash
python ./src/maintkg/main.py
```

## Contributing
We welcome contributions!

### Semantic Commit Messages
Format: `<type>(<scope>): <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Pull Request Process
1. Fork the repository
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to the branch (`git push origin feat/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
