# MaintKG: Automated Maintenance Knowledge Graph Construction
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


**MaintKG** (Maintenance Knowledge Graph) is a framework for automatically constructing knowledge graphs from maintenance work order records. It processes CMMS (Computerized Maintenance Management System) records to create structured, graph-based knowledge representations.

## 🚀 Features

- Automated knowledge graph construction from maintenance records
- Built-in normalization and information extraction (NoisIE)
- Neo4j integration for graph storage and querying
- Comprehensive data processing pipeline

## 📋 Table of Contents

- [Installation](#-installation)
- [Prerequisites](#-prerequisites)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [NoisIE Model](#-noisie-model)
- [Contributing](#-contributing)
- [License](#-license)
- [Attribution](#-attribution)
- [Acknowledgements](#-acknowledgments)
- [Contact](#-contact)

## 🔧 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nl-tlp/maintkg.git
   cd maintkg
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv env
   # On Unix/macOS:
   source env/bin/activate
   # On Windows:
   .\env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

## 📦 Prerequisites

- **Python 3.9+**
- **Neo4j** Database Server
- **PyTorch** (CUDA-enabled recommended)
- **Virtual Environment** (recommended)

## 🔑 Environment Variables

Update the `.env` file in the project root with your own configuration if you wish to create MaintKG from your own data. Otherwise the default will create the default graph.

```plaintext
# Input Settings
INPUT__CSV_FILENAME='your_file.csv'
INPUT__ID_COL='id'
INPUT__TYPE_COL='type'
# ... other settings

# Full configuration example available in `.env`
```
## 💻 Usage

1. **Prepare Your Data**
   - Place your CMMS data in the `./input` directory
   - Configure column mappings by updating the `.env` file.

2. **Run the Pipeline**
   ```bash
   python ./src/maintkg/main.py
   ```

3. **View Results**
   - Generated knowledge graphs are stored in Neo4j
   - Output files are saved in `./output/YYYY-MM-DD_HH_MM-SS-MM/`

##  📁Project Structure
```plaintext
maintkg/
├── cache/                          # Cache directory
│   └── .gitkeep                    # Placeholder for git
├── input/                          # Input data directory
│   └── README.md                   # Input data specifications
├── notebooks/                      # Jupyter notebooks
│   ├── assets/                     # Notebook resources
│   │   ├── images/                 # Visualization images
│   │   └── data/                   # Sample datasets
│   └── example_queries.ipynb       # MaintKG competency queries
├── output/                         # Generated artifacts
│   ├── .gitkeep
│   └── YYYY-MM-DD_HH_MM-SS-MM/    # Timestamped outputs
├── src/                           # Source code
│   ├── maintkg/                   # Core MaintKG package
│   │   ├── __init__.py           # Package initialization
│   │   ├── builder.py            # Graph construction logic
│   │   ├── main.py              # Entry point script
│   │   ├── models.py            # Data models and schemas
│   │   ├── settings.py          # Configuration management
│   │   └── utils/               # Utility functions
│   ├── noisie/                   # NoisIE package
│   │   ├── __init__.py
│   │   ├── download_checkpoint.py  # Model checkpoint downloader
│   │   ├── lightning_logs/      # Model checkpoints
│   │   │   └── .gitkeep
│   │   ├── data/                # MaintNormIE corpus
│   │   │   └── README.md        # Data documentation
│   └── rebel/                    # REBEL package
│       ├── __init__.py
│       ├── extractor.py         # Relation extraction
│       └── utils/               # REBEL utilities
├── .git/                        # Git repository
├── .gitignore                   # Git ignore patterns
├── .pre-commit-config.yaml      # Pre-commit hooks
├── requirements.txt             # Project dependencies
├── pyproject.toml              # Project configuration
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

## 🤖 NoisIE Model

Download the pretrained NoisIE model:

```bash
python ./src/noisie/download_checkpoint.py
```

This will download the model checkpoint to `./src/noisie/lightning_logs/`, making it available when you run the MaintKG process.


## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork & Clone**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Follow Commit Convention**
   ```
   <type>(<scope>): <subject>

   Types:
   - feat: New feature
   - fix: Bug fix
   - docs: Documentation
   - style: Formatting
   - refactor: Code restructuring
   - test: Testing
   - chore: Maintenance
   ```

4. **Submit PR**
   - Ensure tests pass
   - Update documentation
   - Follow code style guidelines


## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🔍 Attribution

If you use MaintKG in your research, please cite:
```bibtex
COMING SOON
```

## 🙏 Acknowledgments

This work was made possible by the [Australian Research Centre for Transforming Maintenance through Data Science](https://www.maintenance.org.au/display/PUBLIC).

## 📧 Contact

For questions, support, or collaboration:
- **Email**: tyler.bikaun@research.uwa.edu.au
