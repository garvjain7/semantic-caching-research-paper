# raw/

Source data is immutable. Do not edit files here; derived artifacts belong
under `data/processed/`.

## Selected source

- Dataset: NQ-Open
- Repository: <https://huggingface.co/datasets/google-research-datasets/nq_open>
- Revision pinned in `configs/corpus.yaml`:
  `5dd9790a83002ad084ddeb7c420dc716852c6f28`
- Fields consumed: `question` (string), `answer` (list of strings)
- Splits: `train`, `validation`
- License reported by the dataset card: CC BY-SA 3.0

NQ-Open is derived from Natural Questions; it is not the full Natural
Questions annotation release. Preserve all answer strings as a list.
Review attribution and share-alike obligations before redistributing
derived data.
