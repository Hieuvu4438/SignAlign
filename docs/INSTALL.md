# Full installation

## 1. SignAlign environment

```bash
conda env create -f environment.yml
conda activate sign-align
python -m pip install -e ".[dev]"
```

CPU-only unit tests do not require third-party source trees or checkpoints.

## 2. Pinned upstream source

```bash
python scripts/setup_external.py
python scripts/setup_external.py --verify-only
```

The helper reads `third_party.lock.json`, checks out the detached
DexAvatar–WiLoR integration and WiLoR commits beneath `external/`, initializes
DexAvatar's pinned Sapiens submodule, and refuses a revision mismatch or dirty
checkout. It does not modify either checkout.

Expected layout:

```text
external/
├── DexAvatar/
│   ├── SMPLer-X/
│   ├── dexavatar_fitting/
│   └── sapiens/
└── WiLoR/
```

## 3. Upstream environments

Follow the installation instructions shipped at the locked revisions. Keep
their environments separate because they pin incompatible versions of some
libraries. The public runner defaults to these Conda names:

| Stage | Environment |
|---|---|
| Sapiens | `sapiens_fix` |
| SMPLer-X | `smpler_x` |
| WiLoR | `wilor` |
| DexAvatar fitting | `dexavatar` |

Override a name with `--sapiens-env`, `--smplerx-env`, `--wilor-env`, or
`--fitting-env`. Inspect the full plan with `--dry-run` before launching GPU
jobs.

## 4. Restricted assets

Download model assets only from the upstream providers after accepting their
terms. Do not commit them to this repository.

- Place the SMPL-X/MANO model files where the locked SMPLer-X and DexAvatar
  instructions expect them, including `MANO_SMPLX_vertex_ids.pkl`.
- Install the Sapiens pose checkpoint requested by its demo script.
- Put WiLoR's `wilor_final.ckpt`, `detector.pt`, and `model_config.yaml` in
  `external/WiLoR/pretrained_models/`, or pass explicit paths to the extractor.
- Install the SignBPoser and SignHPoser assets documented by DexAvatar under
  the locked fitting tree.

The `.gitignore` excludes common model, checkpoint, data, and output paths.
Review [THIRD_PARTY.md](../THIRD_PARTY.md) for license boundaries.
