# Data and model preparation

## Protocol files

`signs.txt` contains whitespace-separated sign and class identifiers:

```text
sign_001 class_001
sign_002 class_002
```

`segments.json` maps each sign to inclusive source-frame bounds:

```json
{"sign_001": [1, 48], "sign_002": [7, 52]}
```

RGB filenames must contain a numeric frame identifier. Sign and segment keys
must match exactly. Set `protocol.expected_signs` and
`protocol.expected_frames` to the intended evaluation split; the pipeline
fails closed on a mismatch.

For video input, extract frames deterministically before creating manifests:

```bash
mkdir -p /data/rgb/sign_001
ffmpeg -i /data/videos/sign_001.mp4 -vsync 0 /data/rgb/sign_001/%06d.png
```

Use the resulting frame identifiers when defining inclusive segment bounds.

## Reuse frontend WiLoR observations

The full DexAvatar–WiLoR runner writes a raw sidecar under each sign directory.
Merge them without re-running the model, then import and validate:

```bash
signalign merge-wilor \
  --sidecars /data/frontend/*/wilor/wilor.pkl \
  --output /data/wilor/raw.pkl

signalign import-wilor \
  --manifests /data/signalign-output/manifests \
  --sidecar /data/wilor/raw.pkl \
  --output /data/wilor/cache

signalign validate-wilor \
  --manifests /data/signalign-output/manifests \
  --cache /data/wilor/cache \
  --output /data/wilor/validation.json
```

The merge refuses duplicate image identities or mismatched repository,
checkpoint, detector, model-config, or format provenance.

## Standalone WiLoR extraction

If the initializer was generated separately, create the same cache directly
from the frozen RGB manifest:

```bash
signalign prepare-manifests \
  --rgb-root /data/rgb \
  --signs /data/protocol/signs.txt \
  --segments /data/protocol/segments.json \
  --output /data/signalign-output/manifests \
  --expected-signs 57 \
  --expected-frames 1493

signalign prepare-wilor \
  --manifests /data/signalign-output/manifests \
  --output /data/wilor/frame_manifest.json

python scripts/extract_wilor.py \
  --frame-manifest /data/wilor/frame_manifest.json \
  --repo ./external/WiLoR \
  --checkpoint ./external/WiLoR/pretrained_models/wilor_final.ckpt \
  --detector ./external/WiLoR/pretrained_models/detector.pt \
  --model-config ./external/WiLoR/pretrained_models/model_config.yaml \
  --out /data/wilor/raw.pkl

signalign import-wilor \
  --manifests /data/signalign-output/manifests \
  --sidecar /data/wilor/raw.pkl \
  --output /data/wilor/cache

signalign validate-wilor \
  --manifests /data/signalign-output/manifests \
  --cache /data/wilor/cache \
  --output /data/wilor/validation.json
```

The frame manifest freezes absolute RGB paths, decoded dimensions, order, and
SHA-256 hashes. Extraction aborts if any source changes. For each side, the
importer selects the highest-confidence detection, converts left-hand
rotations with an explicit reflection, checks SO(3), and stores root-centered
21-joint observations plus 15 MANO rotations.

The later `signalign infer` call regenerates the same target-free manifests under
its output root and verifies the configured protocol counts. Always run
`validate-wilor` before inference; an empty cache is not a completed cache.

## Outputs and storage

All data, checkpoints, pickles, arrays, meshes, and output roots are ignored by
default. Keep reproducibility through configuration files, commit hashes, and
generated JSON manifests—not by committing licensed assets or benchmark data.
