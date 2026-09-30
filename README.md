# FIRE: Fisher Information Recommendation Engine

## Results (MovieLens-1M, 3 seeds, full-catalog eval, mean ± SEM)

| Method | AUC | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---:|---:|---:|---:|---:|
| FIRE | 0.7682±0.0002 | 0.0185±0.0007 | 0.0346±0.0011 | 0.230±0.003 | 0.0179±0.0005 |
| IRT | 0.7766±0.0002 | 0.0230±0.0013 | 0.0406±0.0012 | 0.263±0.007 | 0.0100±0.0004 |
| SVD | 0.7823±0.0000 | 0.0227±0.0019 | 0.0441±0.0022 | 0.262±0.012 | 0.0162±0.0008 |
| Pop | 0.6808±0.0000 | 0.0560±0.0019 | 0.1077±0.0058 | 0.447±0.013 | 0.0189±0.0011 |

- FIRE's main advantage over IRT/SVD: catalog coverage with explicit per-component score decompositions.
- Popularity dominates Top-K accuracy; SVD-50 leads AUC.
- LLTM squared correlation (in-sample, warm items): ρ²=0.5639

## Reproduce

```bash
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip -d data/
pip install -r requirements.txt  # Python >= 3.11
python src/experiment_v5.py --data_dir data/ml-1m --seeds 42 123 789 --n_eval_users 300
```

Outputs (figures, `results_v5.json`, `README_generated.md`) are written to `--out_dir`
(default `repro_output/`); the committed `figures/`, `results_v5.json` and `README.md`
are not modified. Pass `--write_readme` to also overwrite `README.md`.

## Author

Jung Min Kang · Independent Researcher, Seoul · ORCID: 0009-0007-9599-2792
