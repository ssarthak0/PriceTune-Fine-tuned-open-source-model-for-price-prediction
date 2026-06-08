# PriceTune

Fine-tune an open-source LLM (Llama 3.2 3B) to predict product prices from text descriptions. This project uses QLoRA for efficient training and evaluates predictions against baseline models and human performance.

## Overview

PriceTune predicts the price of a product from its description. The pipeline:

1. **Data preparation** — Load product data, analyze token distributions, build training prompts, and publish datasets to Hugging Face Hub.
2. **Fine-tuning** — Train a QLoRA adapter on Llama 3.2 3B using free GPU resources on Google Colab.
3. **Evaluation** — Measure prediction error on a held-out test set and compare against baselines.

**Base model:** [meta-llama/Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B)  
**Method:** QLoRA (4-bit quantization + LoRA adapters)  
**Task:** Regression-style price prediction via prompted completion

## Project structure

```
PriceTune-Fine-tuned-open-source-model-for-price-prediction/
├── 01_data_preparation.ipynb          # Local — data prep & prompt creation
├── 02_fine_tune_colab.ipynb           # Colab — QLoRA fine-tuning
├── 03_evaluate_fine_tuned_model_colab.ipynb  # Colab — model evaluation
├── 04_results_visualization.ipynb     # Benchmark comparison visualization
├── pricer/
│   ├── items.py                       # Item data model & Hugging Face Hub helpers
│   └── evaluator.py                   # Local evaluation utilities
├── .env.example                       # Environment variable template
├── pyproject.toml                     # uv project configuration
└── LICENSE
```

## Requirements

### Local development (data preparation)

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Hugging Face account and API token

### Google Colab (training & evaluation)

- Google Colab account with GPU runtime (T4 or A100)
- Hugging Face token (Colab secret: `HF_TOKEN`)
- Weights & Biases API key for training (Colab secret: `WANDB_API_KEY`)

## Getting started

### 1. Local setup with uv

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/<your-username>/PriceTune-Fine-tuned-open-source-model-for-price-prediction.git
cd PriceTune-Fine-tuned-open-source-model-for-price-prediction

# Create virtual environment and install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Then edit .env and add your Hugging Face token
```

### 2. Data preparation (local)

Open and run `01_data_preparation.ipynb` in Jupyter or VS Code. This notebook:

- Loads product data from Hugging Face Hub
- Analyzes token length distributions
- Creates training prompts with a token cutoff
- Pushes the prepared dataset back to Hugging Face Hub

### 3. Fine-tuning (Google Colab)

1. Upload `02_fine_tune_colab.ipynb` to Google Colab (or open from GitHub).
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
3. Add Colab secrets:
   - `HF_TOKEN` — your Hugging Face API token
   - `WANDB_API_KEY` — your Weights & Biases API key
4. Run all cells. The fine-tuned adapter is pushed to Hugging Face Hub when training completes.

### 4. Evaluation (Google Colab)

1. Upload `03_evaluate_fine_tuned_model_colab.ipynb` to Google Colab.
2. Set runtime to **GPU**.
3. Add the `HF_TOKEN` Colab secret.
4. Update `RUN_NAME` in the notebook to match your trained model.
5. Run all cells to evaluate on the test set.

Evaluation utilities (`Tester` class and `evaluate` function) are included directly in the notebook — no external `util.py` file is required.

## Environment variables

| Variable | Where used | Required |
|---|---|---|
| `HF_TOKEN` | Local (`.env`) and Colab secrets | Yes |
| `WANDB_API_KEY` | Colab training notebook only | Yes (for training) |

Copy `.env.example` to `.env` for local development:

```bash
cp .env.example .env
```

For Colab notebooks, add the same variables under **Secrets** (key icon in the left sidebar) instead of using a `.env` file.

## Configuration

| Setting | Lite mode | Full mode |
|---|---|---|
| Epochs | 1 | 3 |
| Batch size | 32 | 256 |
| LoRA rank | 32 | 256 |
| Max sequence length | 128 | 128 |
| Token cutoff | 110 | 110 |

Toggle `LITE_MODE` in the notebooks for faster experimentation vs. full training.

## Results

Fine-tuned model performance (lower error is better):

| Model | Mean absolute error |
|---|---|
| Base Llama 3.2 (4-bit) | 110.72 |
| Fine-tuned (lite) | 65.40 |
| Fine-tuned (full) | **39.85** |
| GPT 5.1 (reference) | 44.74 |

See `04_results_visualization.ipynb` for the full benchmark comparison chart.

## Acknowledgments

Adapted from the [LLM Engineering course](https://github.com/ed-donner/llm_engineering) (Week 7). Modified for independent use with custom datasets and Hugging Face Hub integration.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
