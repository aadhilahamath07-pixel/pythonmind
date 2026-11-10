# PYTHONMIND

> **An experimental Python-specialized Small Language Model (SLM) and AI programming assistant.**

PYTHONMIND is a learning-driven AI project focused on understanding, building, adapting, and evaluating compact language models for Python development.

The project combines two complementary tracks:

- **PythonMind Core** — a compact decoder-only Transformer implemented from the ground up to demonstrate tokenization, embeddings, causal self-attention, training, and text generation.
- **PythonMind Pro** — an intended Python coding assistant built by adapting a suitable compact pretrained code model. Any pretrained components and their licenses will be documented clearly.

This repository is under active development. Features described as planned are goals, not claims of completed functionality.

## Project goals

- Learn and implement the fundamentals of language models and Transformer architecture.
- Build a small language model pipeline with Python and PyTorch.
- Explore Python-focused data preparation and instruction tuning.
- Generate Python code from natural-language instructions.
- Explain and help debug Python code.
- Provide a practical interface for interacting with the model.
- Evaluate model quality, speed, size, and execution reliability.

## Architecture overview

The intended end-to-end flow is:

`Prompt → Tokenizer → Token IDs → Embeddings → Transformer → Logits → Decoding → Response`

The product architecture is planned around these components:

| Component | Responsibility |
|---|---|
| Tokenizer | Convert text into tokens and token IDs |
| PythonMind Core | Custom decoder-only Transformer for learning and experiments |
| PythonMind Pro | Compact pretrained code model, potentially adapted for Python tasks |
| Inference layer | Prompt formatting, generation, and decoding |
| Backend | API layer planned with FastAPI |
| Frontend | Interactive development interface planned with React |
| Evaluation | Syntax checks, test execution, latency, throughput, and model-size measurements |

See [the architecture notes](docs/architecture.md) for more detail.

## Planned capabilities

- [ ] Natural-language-to-Python code generation
- [ ] Python code explanation
- [ ] Debugging assistance
- [ ] Interactive Python playground with isolated execution
- [ ] Model and benchmark dashboard
- [ ] Documented comparison between the custom Core model and adapted pretrained model

**Security note:** Untrusted Python code must never be executed directly on the application host. Any code-execution feature will require an appropriately isolated environment with resource and time limits.

## Technology direction

- Python
- PyTorch
- NumPy
- Hugging Face Tokenizers / Transformers / Datasets
- PEFT for parameter-efficient fine-tuning experiments
- FastAPI for the backend
- React and Vite for the frontend
- Git and GitHub for version control

The exact model, dataset, and deployment choices will be recorded as the implementation develops. Model and dataset licenses will be reviewed before use.

## Development roadmap

1. Mathematical foundations and probability
2. NLP fundamentals and tokenizer implementation
3. Neural-network building blocks
4. Transformer architecture and attention
5. Implement and train PythonMind Core
6. Prepare Python-focused datasets and explore model adaptation
7. Optimize inference and evaluate performance
8. Build the API and user interface
9. Add safe execution and end-to-end testing
10. Prepare documentation and exhibition demonstration

Track progress in [ROADMAP.md](docs/ROADMAP.md).

## Repository structure

```text
pythonmind/
├── core/           # Custom Transformer implementation
├── tokenizer/      # Tokenizer training and encode/decode pipeline
├── data/           # Dataset preparation documentation and scripts
├── finetuning/     # Fine-tuning experiments
├── backend/        # FastAPI application
├── frontend/       # React application
├── evaluation/     # Model evaluation and benchmarks
├── docs/           # Architecture and project roadmap
├── tests/          # Automated tests
├── requirements.txt
├── .gitignore
└── README.md
```

Directories will be populated as their corresponding components are implemented. Large model weights, generated datasets, secrets, and local environment files should not be committed to Git.

## Current status

**Stage:** Project setup and foundational learning.

The repository currently contains project documentation and configuration. Model implementation, training, fine-tuning, application features, and benchmarks are planned work until supported by code and recorded results.

## Responsible use and attribution

This project is intended for educational and experimental use. Outputs from language models can be incorrect or insecure; generated code should be reviewed and tested. Third-party models and datasets will retain their required attribution and license notices.

## Author

Developed as an independent AI/ML learning project.

---

If you find an issue or have a suggestion, open a GitHub issue with steps to reproduce or a clear description.
