# llm-vs-llm

This folder contains the codebase for running multi-turn LLM-vs-LLM conversation experiments and saving transcripts for later scoring (NLL/logprob) and plotting.

The project is designed to scale across multiple experiment variants via a single YAML config, while keeping outputs reproducible and easy to audit on an HPC cluster.

---

## Current status (what works now)

- You can run transcript generation end-to-end from a single entrypoint (`main.py`).
- Runs are fully config-driven via `config/settings.yaml`.
- Each run:
  - loads a model + tokenizer (currently tested with `gpt2` locally),
  - generates a multi-turn transcript for an experiment + seed,
  - logs each turn to `transcript.jsonl`,
  - writes `resolved_config.yaml` and `meta.json` into the run folder for reproducibility.

---

## Folder structure

- `config/`
  - `settings.yaml`
    - Global settings (topic prompt, num rounds, seeds, output dirs)
    - Model settings (HF model name, device, dtype, chat template flag)
    - Generation settings (max_new_tokens, temperature, top_p, etc.)
    - Style definitions (hidden prompts)
    - Investigator prompt template
    - Experiments registry (baseline / style schedules / investigator)

- `src/`
  - `model_loader.py`
    - Loads `AutoTokenizer` and `AutoModelForCausalLM`
    - Sets pad token safely (important for decoder-only models)
    - Has `set_global_seed()` for reproducibility
  - `io_utils.py`
    - Defines run directory structure and run paths
    - Atomic writes for config + meta
    - Incremental append for:
      - transcript (`append_jsonl`)
      - scores (`append_csv_row`, if/when scoring is enabled)
  - `prompts.py`
    - Plaintext prompt formatting for non-chat models (GPT2-style)
    - Chat message formatting for chat-template models (Qwen-style via `apply_chat_template`)
  - `style_scheduler.py`
    - Implements style scheduling regimes:
      - fixed
      - every_n (random style per block)
      - random_change (probabilistic style changes)
    - Applies hidden style prompts only when the speaker matches `hidden_speaker`
    - Injects investigator instruction only for investigator speaker when enabled
  - `generate_transcript.py`
    - Core loop:
      - determines speaker per turn
      - computes per-turn hidden style + investigator prompt
      - truncates history to fit a max token budget
      - generates the next message
      - logs a JSONL record per turn

- `main.py`
  - CLI runner:
    - `--list_experiments`
    - `--experiment <id>` or `--all_experiments`
    - `--seed <int>` to run one seed (or runs all seeds from config)

---

## Output format

Runs are written under:

- `outputs/<experiment_id>/seed_<seed>/<run_tag>/`

Files:

- `transcript.jsonl`
  - one JSON object per turn
  - fields include:
    - `turn_idx`, `round_idx`
    - `speaker` ("A" or "B")
    - `role` ("user" or "assistant")
    - `content`
    - `hidden_style_id` (style applied this turn, else null)
    - `investigator_active` (bool)
    - `prompt_tokens`, `gen_tokens`
    - `ts_unix`
- `resolved_config.yaml`
  - full config as used for the run (snapshot for reproducibility)
- `meta.json`
  - timestamp, git commit hash (if available), run directory path

---

## How to run (local)

List experiments:

- `python main.py --list_experiments`

Run one experiment across all configured seeds:

- `python main.py --experiment baseline`

Run one experiment for one seed:

- `python main.py --experiment baseline --seed 1`

Run all experiments:

- `python main.py --all_experiments`

---

## Known limitation right now

- With small base models like GPT-2, generating both sides of a “conversation” can drift into incoherent or forum-like text quickly.
- This is not a bug in the pipeline; it is an expected failure mode when:
  - the model is weak at instruction following,
  - there is no strong conversation format enforcement,
  - there is no stopping rule to prevent the model from continuing into “other roles.”

---

## What’s next

- Add scoring (NLL/logprob) over saved transcripts, varying context length (`k_turns_list`) the same way as in the professor’s reference code.
- Switch generation models from GPT-2 to Qwen (3B) and enable chat templates, which should stabilize multi-turn behavior.
- Consider restructuring experiments to avoid free-form “LLM vs LLM chat” drift when the research goal is primarily about context dependence of NLL.