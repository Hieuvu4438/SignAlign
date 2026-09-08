# Third-party software and assets

SignAlign's MIT license applies only to the original code in this repository.
The projects below are external dependencies and retain their own licenses.
No upstream source, checkpoint, dataset, or body-model file is vendored here.

| Dependency | Role | Source boundary |
|---|---|---|
| DexAvatar–WiLoR fork | Initializer fitting and orchestration reference | Pinned external checkout; its nested fitting code has additional terms |
| Sapiens | Whole-body 2D keypoints | DexAvatar's pinned Git submodule |
| SMPLer-X | Per-frame SMPL-X initialization and model utilities | Included only in the external DexAvatar checkout |
| SignBPoser | Sign-aware body prior used by the DexAvatar fitter | External DexAvatar asset/code terms apply |
| SignHPoser | Sign-aware hand prior used by the DexAvatar fitter | External DexAvatar asset/code terms apply |
| WiLoR | Hand detector and 3D hand expert | Pinned external checkout; checkpoint terms are non-commercial/research-oriented |
| SMPL-X / MANO | Parametric human and hand models | Obtain directly after accepting the provider's license |

The exact Git revisions of the DexAvatar–WiLoR integration fork and WiLoR are recorded in
`third_party.lock.json`. The DexAvatar tree in turn pins Sapiens. Run
`python scripts/setup_external.py --verify-only` to check source provenance.

WiLoR is distributed under CC BY-NC-ND 4.0 at the locked upstream revision.
For that reason, SignAlign neither copies nor patches WiLoR. The adapters in
`scripts/` are independent caller-side code and load an untouched checkout at
runtime. Review upstream terms before use, especially for commercial use or
redistribution.

Useful upstream links:

- DexAvatar–WiLoR integration: <https://github.com/Hieuvu4438/DexAvatar>
- Original DexAvatar release: <https://github.com/kaustesseract/DexAvatar>
- Sapiens: <https://github.com/facebookresearch/sapiens>
- SMPLer-X: <https://github.com/caizhongang/SMPLer-X>
- WiLoR: <https://github.com/rolpotamias/WiLoR>
- SMPL-X models: <https://smpl-x.is.tue.mpg.de/>
