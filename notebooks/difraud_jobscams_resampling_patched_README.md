# DiFrauD Job Scams Resampling Experiment: Patched Version

This is the patched Colab-ready script.

It avoids the DiFrauD Hugging Face dataset loader script and reads the raw
`job_scams/*.jsonl` files directly. This avoids the recent `datasets` loader
issues involving `difraud.py` and `datasets.tasks.TextClassification`.

## Colab setup

Upload:

```text
difraud_jobscams_resampling_patched.py
```

Install:

```python
!pip install imbalanced-learn scikit-learn pandas numpy matplotlib
```

Optional:

```python
!pip install umap-learn
```

Run:

```python
!python difraud_jobscams_resampling_patched.py
```

## Outputs

The script writes outputs to:

```text
difraud_jobscams_resampling_outputs/
```

The main chapter-ready figures are:

```text
fig_jobscams_class_distribution_book.png
fig_jobscams_metrics_comparison_book.png
fig_jobscams_confusion_matrices_book.png
```

Optional qualitative visualization:

```text
fig_jobscams_2d_resampling_views_book.png
```

Numerical outputs:

```text
jobscams_resampling_results.csv
jobscams_resampling_class_counts.csv
jobscams_resampling_metadata.json
```

## Download all outputs from Colab

```python
!zip -r difraud_jobscams_resampling_outputs.zip difraud_jobscams_resampling_outputs

from google.colab import files
files.download("difraud_jobscams_resampling_outputs.zip")
```
