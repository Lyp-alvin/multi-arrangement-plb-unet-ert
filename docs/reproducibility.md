# Reproducibility notes

The repository includes the source code and six representative demo samples.
The full manuscript metrics require the complete 2000-sample synthetic dataset
and the trained checkpoints.

Recommended release procedure:

1. Keep the source code in this GitHub repository.
2. Store large checkpoints and the full dataset in Zenodo or GitHub Releases.
3. Record the exact commit hash in the manuscript.
4. Add the DOI or release URL to `README.md` and `checkpoints/README.md`.

The included demo samples are sufficient for:

- checking data loading,
- inspecting model input/output shapes,
- reproducing figure-style prediction visualizations when checkpoints are
  available.

They are not sufficient for reproducing the manuscript's full quantitative
table.

