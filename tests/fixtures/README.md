# Frozen recall reference

`recall_reference.pt` contains only synthetic tensors, invented IDs, and random
weights. It is loaded with `torch.load(..., weights_only=True)`. No dataset rows,
private artifacts, or trained research weights are included.

Reference: commit `033fdc90d89a1c8b9dc5a6dbb6940290762af3a5`, Python 3.10/CPU.
The recall case uses users [1,2,3], items [10,20,30,40], positive histories
[[0,1],[1],[2]] and observed negatives [[2,3],[0],[]]. It records two seed-42 BPR
epochs and the original popularity/cosine/MF scores, candidate order and support
counts for users [1,2,3,99], including unknown candidate 999. The current sampler and
trainer are checked against the original weights and curve. Matched-runtime local
checks were bit-for-bit equal; portable CI permits float32 rounding differences
(`rtol=1e-6`, `atol=1e-7`) for scores/weights/losses. IDs, ordering, support masks
and counts remain exact. Current uninterrupted/resumed CPU training is also exact.

To reproduce a historical fixture, use the pinned source in a separate checkout;
do not regenerate expected values from the reader being tested.

The fixture retains the unchanged recall payload of the former combined fixture.
Historical ranker recipes/readers are retired; their fixture remains in Git history.
