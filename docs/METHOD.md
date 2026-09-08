# Method and invariants

SignAlign separates target-free inference from evaluation. The inference
configuration rejects evaluator, reference-mesh, ground-truth, and target-mesh
path keys before any model is constructed.

## C0: initializer boundary

The frontend creates one SMPL-X parameter file and one mesh for every protocol
frame. `signalign build-initializer` accepts the per-sign JSONL manifest
directory produced by `signalign prepare-manifests`. A legacy CSV with `sign`
and `prediction_path` columns is also accepted. For each identity
`(sign, frame)`, it selects the primary source only when both files exist:

```text
<root>/<sign>/smplifyx/results/<frame>.pkl
<root>/<sign>/smplifyx/meshes/<frame>.obj
```

Otherwise it selects both files from the fallback. The locked view uses
absolute symlinks and records the selection and manifest digest. This prevents
silent partial-frame parameter mixing.

The public runner supplies the DexAvatar–WiLoR path used to generate the
primary initializer. SignBPoser and SignHPoser are upstream fitting priors in
C0 only; neither is imported by the SignAlign package.

## C1: shared identity and canonical re-fit

SMPL-X shape vectors from pose-diverse calibration frames are standardized and
aggregated with an iteratively reweighted Huber location estimator. Optional
shape refinement compares model vertices with the frozen initializer while
keeping the estimate close to the robust location.

Every sequence is then re-fit with that shared shape. The optimizer may update
the shoulder, elbow, wrist, and hand chain to compensate for the identity
change. Global orientation, translation, lower body, jaw, expression, and
camera remain inherited from the frozen initializer. The stage emits one
sequence archive per sign plus checksummed completion metadata.

## C2: hand-only local refinement

For each available hand, both SMPL-X and WiLoR joints are transformed into a
palm coordinate system: subtract the wrist, construct a right-handed palm
basis, and normalize scale. This removes global translation, orientation, and
scale from the hand objective.

Only 15 local finger rotations are optimized. A tangent-space residual is
mapped through the SO(3) exponential and hard-bounded to the configured radius
(12 degrees by default). A quadratic residual prior discourages unnecessary
updates. After refinement, the implementation verifies protected SMPL-X state
arrays byte-for-byte. If WiLoR has no valid observation for one side, that
side's canonical state is copied unchanged.

## Artifact contracts

- `manifests/*.jsonl`: ordered RGB records containing no target geometry.
- `identity/signer.npz`: shared signer shape and calibration evidence.
- `canonical_fit/clips/*/mesh_parametric_final.npz`: C1 sequence states.
- `hand_manifest.jsonl`: RGB/state/mesh triplets with SHA-256 digests.
- `predictions/states`: final SMPL-X parameter archives.
- `predictions/meshes`: final OBJ meshes.
- `predictions/decisions`: accepted sides, fallbacks, locks, and output hashes.

Every completed stage is append-only: an existing completion marker is either
verified and reused or causes a refusal to overwrite.
