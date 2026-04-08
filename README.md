# Sorry for the late reply: Response times and reciprocation in WhatsApp and Instagram chats

Code accompanying the [arxiv paper](placeholder). Please cite as:

> [CITATION / LINK PLACEHOLDER]


## Repository structure

```
.                                                                                      
├── run_analyses.ipynb        # Main file: Setup, filtering, statistical analysis, and plots
├── src/                      # Python modules
│   ├── config.py             # Pydantic experiment configuration
│   ├── etl.py                # Data transformation and filtering functions
│   ├── models.py             # Linear mixed model wrapper (pymer4/lmer)
│   └── util.py               # Utility functions (I/O, descriptive statistics)
├── env.yml                   # Conda environment specification
├── data/                     # Generated summary statistics (.dat files) for LaTeX
└── figures/                  # Generated figures
```

## Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)

### Installation

The conda environment includes Python 3.12, R 4.3, and all required packages (including R libraries such as `lme4` and `lmerTest` via `rpy2`). No separate R installation is needed.

```bash
conda env create -f env.yml -n myenv
conda activate myenv
```

## Usage

Run `run_analyses.ipynb`. Starting from `data.parquet` that can be requested on [Zenodo](https://doi.org/10.5281/zenodo.19369010), this notebook:

1. **Filters** conversations (system message removal, group chat removal, unidirectional chat removal, minimum 100 messages per conversation, minimum 5 conversations per donor).
2. **Transforms** data by computing response times, merging consecutive same-sender messages, and converting to minute-level resolution.
3. **Fits linear mixed models** separately for WhatsApp and Instagram:
   `rt_ego ~ rt_prev_alter + (1 | donor_id)`
4. **Computes JSD-based similarity** between ego and alter response time distributions per conversation.
5. **Generates all figures and summary statistics** reported in the paper.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE](LICENSE) for the full text.
