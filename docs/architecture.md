# PYTHONMIND — Architecture Notes

## 1. Purpose

PYTHONMIND explores how a compact language model can be built and adapted for Python programming tasks. The design separates an educational model implementation from a practical assistant built around a suitable pretrained code model.

## 2. Two model tracks

### PythonMind Core

A decoder-only Transformer implemented with PyTorch. Its purpose is to make the internal mechanics understandable and measurable:

1. Text normalization and tokenization
2. Token IDs and embedding lookup
3. Positional information
4. Causal self-attention
5. Multi-head attention
6. Feed-forward network
7. Residual connections and layer normalization
8. Output projection, loss, and autoregressive generation

The initial model size and training configuration will be selected after the tokenizer and baseline implementation are tested. No performance target is guaranteed in advance.

### PythonMind Pro

A proposed assistant using a compact pretrained code model. The project may evaluate instruction tuning or LoRA adaptation for Python-specific tasks. The selected model, dataset provenance, license, and changes will be documented before integration.

A pretrained model is not the same as a model trained from scratch; the project will report these tracks separately.

## 3. Inference flow

`User prompt → Prompt formatting → Tokenizer → Model → Logits → Sampling/decoding → Output validation → Response`

The backend is planned to expose a stable interface so that Core and Pro can be evaluated independently.

## 4. Application layers

- **Frontend:** React interface for prompts, generated code, explanations, and evaluation information.
- **API:** FastAPI endpoints for model selection and inference.
- **Model layer:** Tokenizer, model loading, generation settings, and output handling.
- **Evaluation:** Syntax validation, unit-test execution in a controlled environment, latency, tokens per second, and model footprint.
- **Execution isolation:** Any user-provided code execution must run in a restricted sandbox, not in the API process or on the host without isolation.

## 5. Evaluation principles

Report measurements with the model name/version, hardware, dataset or test set, and settings. Useful metrics include:

- Parameter count and on-disk model size
- Time to first token and total generation latency
- Tokens generated per second
- Python syntax-validity rate
- Unit-test pass rate on a documented test set
- Reproducibility of evaluation runs

Do not describe a model as accurate or production-ready without reproducible evidence.
