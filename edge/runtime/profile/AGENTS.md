# Profile Boot Seam

This package owns boot-time `ML_WORKER_PROFILE` mapping to device and decode,
fail-fast device verification, and global decode preflight. It may import
`edge.runners`, `edge.sources`, and `contracts`; it must not import
`edge.domains`, `edge.perception`, `edge.evidence`, `edge.serving_client`,
`edge.features`, or `backend`. Provisioning threads through the composition
root, not this package.
