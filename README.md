# olmix on LUMI

Finds a good pretraining data mixture by training many small models on different
mixtures and fitting a regression to their evaluation scores. The mixture
sampling comes from [olmix](https://github.com/allenai/olmix)
([paper](https://arxiv.org/abs/2602.12237)); everything else here adapts it to
how we run jobs on LUMI.

The pipeline:

```
configs/<experiment>.yaml
  └─ generate_variants.py  ─>  data/runs/<run>/{config.yaml, variants.json, mixes/}
                               launch_all_swarms.sh
  └─ launch_all_swarms.sh  ─>  one training job per mixture
  └─ convert_olmix_models_to_hf.sh
  └─ eval_bpb.sh           ─>  BPB scores
  └─ collect_bpb.py        ─>  ratios.csv + metrics.csv
  └─ olmix fit             ─>  suggested mixture
```

## Before you start

Put your tokens in `~/.hpc_secrets`; `scripts/train-0.05B.sh` sources it:

```bash
export WANDB_API_KEY=""
export HF_TOKEN=""
```

Mixture generation, result collection and the fit all run from a python environment with olmix in it. requirements.txt is a freeze of a working one:

bash
export UV_CACHE_DIR=/scratch/project_465002530/users/$USER/uv-cache   # home quota is small
uv venv /scratch/project_465002530/users/$USER/olmix-venv
uv pip install --no-deps --python /scratch/project_465002530/users/$USER/olmix-venv/bin/python \
    -r requirements.txt
export OLMIX_PYTHON=/scratch/project_465002530/users/$USER/olmix-venv/bin/python

--no-deps is required: olmix pins ai2-olmo-core to a branch that no longer exists upstream, so a normal install fails. Everything it needs is listed in requirements.txt already, resolved from main.

Preprocessed training data lives under
`/scratch/project_465002530/preprocessed/oellm-v1-256k/catalogue/`.

## 1. Describe the experiment

A config has three blocks: `settings`, `swarm`, and `datasets`. Copy an existing
one — `configs/flagship_config.yaml` is the most current.

```yaml
settings:
  max_tokens: 3000000000      # size of the run the mixture is being chosen for
  repetition_factor: 5.0      # how many times a dataset may be repeated
  seed: 42                    # same seed + same config = same swarm

datasets:
  open_web_math:
    multiplier: 1             # >1 if the path is a sample of a larger dataset
    paths:
      - /scratch/.../open-web-math/open-web-math    # no .bin suffix
```

Datasets may instead have `quality:` buckets, which become separate leaves named
`source:bucket`.

Check the paths resolve, and see how many tokens each holds, before generating:

```bash
$OLMIX_PYTHON check_paths.py configs/<experiment>.yaml
```

It also warns when a directory holds more `.bin` files than the config lists —
usually a multi-shard collection where only one shard is referenced.

## 2. Generate the mixtures

```bash
$OLMIX_PYTHON ./scripts/generate_variants.py configs/<experiment>.yaml
```

This writes `data/runs/<run_prefix>/` containing the mixes, a `variants.json`
mapping each variant to its weights, and a copy of the config used. It also
rewrites `launch_all_swarms.sh` with the run prefix baked in.

Keep `variants.json`. It is the only record of which mixture a trained model
saw, and the fit needs it.

### Mixture reuse

To keep an existing mixture fixed and search only over what you add to it, point
`existing_mix_file` at a JSON of `{dataset_name: weight}`:

```yaml
swarm:
  existing_mix_file: "data/flag_frozen.json"
  manual_prior:
    existing: 0.95            # share of the simplex the frozen block occupies
    open_web_math: 0.05
  nonzero_weight:             # source names; stops sparse variants dropping the block
    - existing
    - open_web_math
```

Datasets named in that file are locked into one source called `existing`, keeping
their ratios to each other. Datasets in the config but *not* in that file are what
the swarm varies. So which dataset you leave out decides what the experiment
answers.

Without `manual_prior` the split between frozen and new comes from token counts,
which usually leaves a new dataset with roughly the prior it already has, and the
swarm explores nothing. Set it explicitly.

Two settings need different values than a full swarm: use
`min_source_strength: 1` / `max_source_strength: 20` (low strengths make
degenerate mixes when there are only a few sources), and keep
`minimum_source_weight` at `0.0001` — the frozen block's weight is split among its
members, so a coarse grid rounds the smaller ones to zero.

### Checking a swarm before spending compute

```bash
$OLMIX_PYTHON -c "
import json
v = json.load(open('data/runs/<run>/variants.json'))
print(list(v[0]['mix']))                       # leaf names, alphabetical
print(sorted(round(x['mix']['<dataset>']['weight'], 4) for x in v))
"
```

Weights should span a range that brackets the value you are questioning. If every
variant sits near the same number, the prior or the strengths need adjusting.

## 3. Train

```bash
./launch_all_swarms.sh
```

Submits one array task per mixture. Each task reads
`data/runs/$MIX_PREFIX/mixes/<variant>.txt` and writes checkpoints to
`/flash/project_465002530/users/$USER/ablation_output/<variant>/`.

Model size, token budget and optimizer settings are in `scripts/train-0.05B.sh`.
Lower `TRAIN_TOKENS` for a quick pipeline test; note the final checkpoint's
iteration number changes with it.

## 4. Convert and evaluate

```bash
./scripts/convert_olmix_models_to_hf.sh
```

BPB evaluation needs our fork of oellm-eval:

```bash
uv tool install -p 3.12 --force git+https://github.com/nicher92/oellm-eval.git@bpb-metrics
export HF_HOME=/scratch/project_465002530/cache/huggingface
oellm-eval schedule --models "<path to model>" --task_groups "bpb-core"
```

`./scripts/eval_bpb.sh <hf-dir> [...]` wraps that call: it checks each path holds
a model before scheduling, and takes `TASK_GROUPS=bpb-all` to include MMLU STEM.

## 5. Collect results for the fit

```bash
python scripts/collect_bpb.py \
    --results-dir <eval output>/<timestamp> \
    --variants-json data/runs/<run>/variants.json \
    --output-dir data/runs/<run>/olmix_fit
```

Writes `ratios.csv` (mixture weights) and `metrics.csv` (BPB scores), which is
what `olmix fit` consumes.

## 6. Fit the regression

The fit needs a config of its own. Copy `configs/fit_example.yaml` and set the
two CSV paths, then paste in a priors block:

```bash
$OLMIX_PYTHON scripts/extract_priors.py configs/<experiment>.yaml
```

Check its output lists every dataset in `ratios.csv`: datasets whose `.bin`
files are missing are dropped with only a warning, and a prior missing for a
column that varied makes the fit meaningless.

```bash
$OLMIX_PYTHON -m olmix fit --config configs/<fit>.yaml --output-dir data/runs/<run>/fit
```

Two settings worth knowing:

- `regression.aggregate_task_families` must be `false` unless you also supply an
  eval config defining the families; otherwise the run stops with an error.
- `constraints.enabled: true` caps each weight by its token budget
  (`weight x target_tokens <= tokens x repetition_factor`). Set `target_tokens`
  to the size of the real run you are choosing a mixture for, not the proxy's.

### Reading the result for a reuse swarm

Only the datasets you left *out* of the frozen file carry information. Everything
inside the frozen block moves in fixed proportion across every variant, so the
regression cannot tell which member caused a change — their individual weights in
the output come from the KL term pulling toward the prior, not from evidence.
Read the varied datasets and the block total; treat the rest as unchanged.

This also sets the size of an experiment: one unfrozen dataset answers one
question. To compare several candidates, unfreeze them all so they vary
independently, and expect to need more variants (olmix used 16 for a two-source
round, 64 for six).

## Layout

| Path | What it is |
| --- | --- |
| `configs/*.yaml` | Experiment definitions |
| `configs/30m_config.json`, `configs/run_config.yaml` | Conversion inputs for Megatron-Bridge; not used by the generator |
| `data/runs/<run>/` | One swarm: its config, variants and mixes |
| `data/*_frozen.json`, `data/stage_1_optimal.json` | Frozen mixtures for reuse experiments |
| `scripts/generate_variants.py` | Samples mixtures with olmix |
| `scripts/train-0.05B.sh` | One training run; submitted as an array |
| `scripts/collect_bpb.py` | Eval results + variants → fit inputs |
| `scripts/test_bpb_scoring.py`, `scripts/check_upstream_acc_sh` | Checks on the BPB implementation |

## Notes

The generator calls olmix's `generate_weights_dirichlet` directly rather than its
CLI, because the CLI assumes AI2's launcher. That function returns weights ordered
by source name, so `olmix_leaf_order()` builds the priors in that same order —
without it every dataset receives another dataset's weight. Pin the olmix version
when upgrading, and re-check that ordering.

The original `nested-swarm` results predate that fix: each dataset's prior was
applied to a different dataset. Its mixes and variants file agree with each other,
so the fit describes what was trained, but the exploration was not as intended.

## TODO
- Create separate test for config, checking paths, linting etc
- Create unit tests for all functions
- Underscores under functions that are not used outside of the file
- Split code into smaller, modular parts, ie translate code to what it does, shorten main functions
- Add ruff linter and pre-commit hooks
- Join all steps after creation of mixes into one slurm file to run huggingface model conversion, evaluation and regression mix (using slurm --dependency after ok) 
- `configs/config.yaml` points at a project we no longer have access to
- `iter_0022889` is hardcoded in `train-0.05B.sh` and `convert_olmix_models_to_hf.sh`;
  it is only correct for `TRAIN_TOKENS=3000000000`. Derive it from `TRAIN_ITERS`,
  or read `latest_checkpointed_iteration.txt` from the checkpoint directory
- `eval_bpb.sh` and `generate_mixes.sh` do not yet source `env.sh`
- `extract_priors.py` drops datasets whose `.bin` files are missing, with only a warning
- `make_ratios.py` duplicates what `collect_bpb.py` already does
