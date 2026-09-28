# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

This is an **overlay**, not a runnable project on its own. It is the `LLMOps/Evaluation/Long_Context` subdirectory of [ML-TANGO/TANGO2](https://github.com/ML-TANGO/TANGO2). It adds a local Hugging Face Transformers provider, configs, and eval scripts to the upstream NIAH harness ([gkamradt/needle-in-a-haystack](https://github.com/gkamradt/needle-in-a-haystack)). The upstream repo is deliberately not vendored. `scripts/setup.sh` clones it into `./needle-in-a-haystack`, and `NIAH_DIR` overrides the location.

- `files/X` is copied to `<niah>/X`, with the same relative path.
- `patches/*.patch` are `git apply`'d to upstream files.
- `scripts/` run from here and operate on the clone.

Keep upstream edits minimal: the only upstream-file changes are the two patches. If you edit an overlay file, re-run `scripts/setup.sh` (or `cp` it) so the clone picks it up. Regenerate a patch with `git -C needle-in-a-haystack diff <file> > patches/<name>.patch`.

## Commands

```bash
scripts/setup.sh                      # clone NIAH @ pinned commit, copy files/, apply patches, build .venv, install pinned deps
scripts/run_eval.sh --model google/gemma-4-E4B-it --context_length 4 8 16 [--tokenizer gpt|model|<hf id>] [--adapter DIR] [--depths 25 75] [--gpus 0,1] [--dry_run]
needle-in-a-haystack/.venv/bin/python -m pytest -p no:cacheprovider          # upstream tests (from inside the clone)
needle-in-a-haystack/.venv/bin/python -m pytest tests/test_x.py::test_name   # single test
```

- The scripts call the venv binaries directly, not `uv run`. A plain `uv run` would re-sync and revert the pinned `tokenizers==0.22.2`. If you use uv manually, use `uv run --no-sync`.
- On this host `UV_CACHE_DIR=/opt/uv/cache` is root-owned. `setup.sh` falls back to `$HOME/.cache/uv`, and manual `uv` calls need `UV_CACHE_DIR=$HOME/.cache/uv`.
- `.venv` is not portable: it hard-codes its path and uses a uv-managed Python 3.12. On a new server, run `setup.sh`, which detects a copied or broken venv and rebuilds it. `/home` is NFS, where uv's own venv removal can fail, so `setup.sh` runs `rm -rf .venv` itself.

## How the pieces connect

- **Provider registration:** `transformers_local.py` calls `register_provider("transformers", "generate", ...)` at import time. It is only discovered because `register_transformers_provider.patch` imports it in upstream `providers/__init__.py`.
- **Config chain:** a run YAML's `model:` is either a model-config id (looked up in `configs/models/`) or a path. `run_eval.sh` → `niah_helper.py plan` writes `run.yaml`, `model.yaml`, and `plan.json` into `results/<run_name>/`, and the run YAML points to that `model.yaml` by path. NIAH has no CLI overrides, so generated YAML is the only way to parametrize a run.
- **Extra `request` fields** (`dtype`, `device`, `adapter_path`, `tokenizer_path`, `do_sample`, ...) reach the provider builder because upstream `ModelRequest` has `extra="allow"`. `device` is passed as `device_map` (`auto` = spread over visible GPUs). `temperature` is ignored; decoding is greedy.
- **Model loading:** it tries `AutoModelForImageTextToText` first (Gemma 4 is `Gemma4ForConditionalGeneration`, hence the `pillow`/`torchvision` deps), then falls back to `AutoModelForCausalLM` for text-only models. `adapter_path` wraps the model with `peft.PeftModel`.
- **Length tokenizer:** all NIAH length math (haystack truncation, needle insertion, period snapping) goes through `needlehaystack/core/tokens.py`. `configurable_tokenizer.patch` makes it read `NIAH_TOKENIZER` (an HF tokenizer id/path); unset means upstream's `cl100k_base`. The env var must match between `niah run` and `niah reconstruct`, and `plan.json` records it as `length_tokenizer`.
- **Length fitting:** `context_length` counts haystack tokens only. The model sees haystack + needle + question + chat template, and must also fit `max_new_tokens` under `max_position_embeddings`. `niah_helper.py plan` builds the exact prompt for each length and counts it with the model's chat tokenizer. With `--tokenizer model`, an overflowing length (128k on Gemma 4 E4B → 130,901) is trimmed. Otherwise it is skipped. The count matches the provider's reported `usage.input_tokens` exactly.
- **Resume key** is `(run_name, model_id, context_length, depth, seed)`. `run_eval.sh` derives `run_name` (= model config `id`) from model + adapter + tokenizer, so different adapters never collide. Error rows also count as done, so delete the row or the results dir to retry.
- **Scoring (single task):** a case-insensitive substring match of `eat a sandwich and sit in Dolores Park` in the response. The score is 1.0 or 0.0, and any paraphrase scores 0, so check `response` before trusting a 0.
