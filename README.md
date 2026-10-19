# SignAlign

SignAlign combines signer-consistent SMPL-X canonicalization with palm-canonical
finger refinement. It estimates a shared signer shape, refits the upper limbs
and hands while preserving the initial reconstruction, then retargets hand-expert
geometry with wrist-locked finger updates bounded to 12 degrees.

Install with `pip install .`. The Python API is `signalign.SignAlign.reconstruct`.
It accepts frozen initializer parameters and meshes, anatomical hand-expert
joints, availability flags, and optional clip IDs. Inputs and coordinate/hand-mean
conventions are specified in the API docstrings. It returns parameters, vertices,
the shared shape, calibration indices, and per-hand refinement flags.

SMPL-X assets and upstream initializer/hand-expert predictions are supplied
separately under their respective licenses. Ground-truth geometry is not an
input to the method.
