# filler_pool/

Background cached entries used only in the cache_scale experiment
(configs/cache_scale.yaml), to test collision risk as the corpus
grows. Not part of the labeled correctness dataset itself — these
entries don't need engineered ground-truth labels, they just need to
exist as "noise" in the index.
