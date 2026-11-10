# NeuroCLR thyroid classification study

Use `thyroid-journal_code.ipynb` for the analysis reported in the manuscript. Start Jupyter in the directory containing the notebook, `thyroidDF.csv`, `outputs/`, and `manuscript/`. The software versions in `requirements.txt` describe the environment inspected after training. The original training run recorded Python but did not freeze the complete environment.

## Verify the input data

The analyzed CSV has SHA-256 `17bfe20da369ef710c5bea26fdec8b0210a32e628621cbbe1537e6607b1c544e`. It matches the CSV in the Kaggle archive byte for byte. Its 9,172 records also match the official UCI records in order across all 31 fields after normalization. These checks establish the contents of the local copy. Its original personal download URL is unknown.

Credit J. Ross Quinlan and the UCI Machine Learning Repository for the Thyroid Disease dataset. The UCI entry lists CC BY 4.0. Download the official archive and the Kaggle mirror, then supply their local paths to the command below. [UCI dataset and licence](https://archive.ics.uci.edu/dataset/102/thyroid+disease) · [Kaggle CSV mirror](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data)

```sh
python manuscript/scripts/verify_thyroid_source.py --uci path/to/uci.zip --kaggle path/to/kaggle.zip
```

## Reproduce the reported results

Run these commands from the project root with the saved files in `outputs/data/`.

```sh
python manuscript/scripts/audit_table5.py
python manuscript/scripts/direct_paired_bootstrap.py
```

The Table 5 audit compares all 14 rows against the saved output of Section 16 and its full-precision export, `16_final_test_results.csv`. The original cell prints four decimal places. A supplementary cell immediately after it displays the exported Brier scores to six decimal places and both ECE measures to four. Its output is also in `table5_six_decimals.txt`. The row-level audit in `table5_calibration_audit.csv` identifies which metrics were also recomputed from archived probabilities. Those matrices are available for unadjusted NeuroCLR, probability-only stacking and enriched bagged LightGBM. The other rows can be checked against the evaluation export. The audit script also provides a function to archive all matrices from a completed Section 16 kernel.

The paired analysis reads `direct_neuroclr_predictions.npz`. Both models in each contrast are scored on the same 1,000 test resamples. The seeds are 0 through 999. Point differences and percentile intervals are saved in `direct_neuroclr_paired_bootstrap.csv`. Counts for exploratory deferral are in `direct_deferral_error_audit.csv`, and the console output is in `direct_paired_bootstrap_output.txt`.

| Unadjusted NeuroCLR minus comparator | Macro-F1 difference (95% interval) | Balanced-accuracy difference (95% interval) |
| --- | --- | --- |
| Probability-only stacking | 0.0061 (−0.0034 to 0.0176) | 0.0111 (0.0016 to 0.0220) |
| Enriched bagged LightGBM | 0.0159 (−0.0023 to 0.0362) | 0.0281 (0.0077 to 0.0514) |

The intervals describe test-sampling variability conditional on the fitted models. They are unadjusted for multiple comparisons. At the post hoc deferral cutoff, 30 of 66 errors fell among the 67 deferred records.

## Read the notebook labels

- `Base-48` is a legacy notebook label for the 34-feature input called Base-34 in the manuscript.
- F09 is enriched, unadjusted NeuroCLR, emphasized retrospectively in the manuscript. F12 is the adjusted enriched configuration.
- The Section 16 minority-three recall summary is a bootstrap mean. For F09 it is 0.8384. The manuscript reports the observed mean of the three class recalls, 0.8381. Compute that point estimate from the class counts before rounding.

## Fitted model checkpoint

Section 14 imports `manuscript/scripts/neuroclr_checkpoint.py` to save the fitted experts and combiners under `outputs/models/` and check predictions after loading. The historical run archived test probabilities. A verified fitted-model checkpoint from that run is unavailable. To produce a checkpoint, run the training notebook through Section 14 and check for the successful verification message before using the model on new records.
