print("\n===== SECTION 16: CONFIRMATORY HELD-OUT TEST EVALUATION & CALIBRATION =====")
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score,
    f1_score, recall_score, matthews_corrcoef, roc_auc_score
)
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

hyb_y_trval = np.asarray(hyb_y_trval)
hyb_y_test  = np.asarray(hyb_y_test)
cls_names   = list(hyb_target_le.classes_)
ALL_LABELS  = np.arange(len(cls_names))

idx_binding = cls_names.index('Binding_Protein_Other') if 'Binding_Protein_Other' in cls_names else 0
idx_hyper   = cls_names.index('Hyperthyroid') if 'Hyperthyroid' in cls_names else 1
idx_hypo    = cls_names.index('Hypothyroid') if 'Hypothyroid' in cls_names else 2
idx_normal  = cls_names.index('Normal') if 'Normal' in cls_names else 3
idx_repl    = cls_names.index('Replacement_Therapy_Misc') if 'Replacement_Therapy_Misc' in cls_names else 4
MINORITY_IDS = [idx_binding, idx_hyper, idx_repl]

# Helper that fits a single-fit GBDT while strictly honoring Optuna's chosen 'resample' mode
def hyb_fit_single_honoring(clf, trial_best, X_tr, y_tr, X_te, seed=hyb_RANDOM_STATE):
    mode = trial_best.params.get('resample', 'none')
    imp, sca = SimpleImputer(strategy='median'), StandardScaler()
    X_tr_s = sca.fit_transform(imp.fit_transform(X_tr))
    X_te_s = sca.transform(imp.transform(X_te))
    cls_u, cnt_u = np.unique(y_tr, return_counts=True)
    w_map = {c: len(y_tr) / (len(cls_u) * n) for c, n in zip(cls_u, cnt_u)}
    sw = np.array([w_map[l] for l in y_tr])
    _, p_te = fit_predict_gbdt_honoring_optuna(
        clf, mode, X_tr, X_tr_s, y_tr, sw, X_te, X_te_s, X_te, X_te_s, seed
    )
    return p_te

# 1. Fit Untuned Base-48 (No Resampling) & Matched Single-Fit Tuned Models (Honoring Optuna Resampling)
clf_lgbm_base48_untuned = LGBMClassifier(random_state=hyb_RANDOM_STATE, verbose=-1, n_jobs=-1).fit(hyb_X_trval_base, hyb_y_trval)
clf_xgb_base48_untuned  = XGBClassifier(eval_metric='mlogloss', random_state=hyb_RANDOM_STATE, n_jobs=-1).fit(hyb_X_trval_base, hyb_y_trval)

test_probs_lgbm_base  = clf_lgbm_base48_untuned.predict_proba(hyb_X_test_base)
test_probs_xgb_base   = clf_xgb_base48_untuned.predict_proba(hyb_X_test_base)
y_test_pred_lgbm_base = np.argmax(test_probs_lgbm_base, axis=1)
y_test_pred_xgb_base  = np.argmax(test_probs_xgb_base, axis=1)

# Single-fit tuned models honoring Optuna resample mode
pr_lgbm_b48_single = hyb_fit_single_honoring(
    LGBMClassifier(**hyb_get_clean_params(hyb_lgbm_best), random_state=hyb_RANDOM_STATE, verbose=-1, n_jobs=-1),
    hyb_lgbm_best, hyb_X_trval_base, hyb_y_trval, hyb_X_test_base
)
pr_xgb_b48_single = hyb_fit_single_honoring(
    XGBClassifier(**hyb_get_clean_params(hyb_xgb_best), eval_metric='mlogloss', random_state=hyb_RANDOM_STATE, n_jobs=-1),
    hyb_xgb_best, hyb_X_trval_base, hyb_y_trval, hyb_X_test_base
)
pr_cat_b48_single = hyb_fit_single_honoring(
    CatBoostClassifier(**hyb_get_clean_params(hyb_cat_best), random_state=hyb_RANDOM_STATE, verbose=0),
    hyb_cat_best, hyb_X_trval_base, hyb_y_trval, hyb_X_test_base
)

clf_lgbm_final = LGBMClassifier(**hyb_get_clean_params(hyb_lgbm_best), random_state=hyb_RANDOM_STATE, verbose=-1, n_jobs=-1)
pr_lgbm_fe_single = hyb_fit_single_honoring(
    clf_lgbm_final, hyb_lgbm_best, hyb_X_trval_final, hyb_y_trval, hyb_X_test_final
)
clf_xgb_final = XGBClassifier(**hyb_get_clean_params(hyb_xgb_best), eval_metric='mlogloss', random_state=hyb_RANDOM_STATE, n_jobs=-1)
pr_xgb_fe_single = hyb_fit_single_honoring(
    clf_xgb_final, hyb_xgb_best, hyb_X_trval_final, hyb_y_trval, hyb_X_test_final
)
clf_cat_final = CatBoostClassifier(**hyb_get_clean_params(hyb_cat_best), random_state=hyb_RANDOM_STATE, verbose=0)
pr_cat_fe_single = hyb_fit_single_honoring(
    clf_cat_final, hyb_cat_best, hyb_X_trval_final, hyb_y_trval, hyb_X_test_final
)
# Also fit un-resampled tree on hyb_X_trval_final if needed by Section 17 TreeSHAP
if hyb_lgbm_best.params.get('resample', 'none') == 'smote':
    clf_lgbm_final.fit(hyb_X_trval_final, hyb_y_trval)

# 2. Calibration Helper: Multiclass Brier Score, Top-Label ECE, and Mean Class-Wise ECE
def compute_calibration_metrics(y_true, probs, n_bins=10):
    y_arr = np.asarray(y_true)
    y_bin = label_binarize(y_arr, classes=ALL_LABELS)
    brier = float(np.mean(np.sum((probs - y_bin) ** 2, axis=1)))

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_arr).astype(float)
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    top_ece = 0.0
    for b_low, b_high in zip(bin_bounds[:-1], bin_bounds[1:]):
        in_b = (confidences > b_low) & (confidences <= b_high)
        if np.any(in_b):
            top_ece += np.abs(np.mean(confidences[in_b]) - np.mean(accuracies[in_b])) * np.mean(in_b)

    cw_eces = []
    for c in ALL_LABELS:
        p_c, y_c = probs[:, c], y_bin[:, c]
        ece_c = 0.0
        for b_low, b_high in zip(bin_bounds[:-1], bin_bounds[1:]):
            in_b = (p_c > b_low) & (p_c <= b_high)
            if np.any(in_b):
                ece_c += np.abs(np.mean(p_c[in_b]) - np.mean(y_c[in_b])) * np.mean(in_b)
        cw_eces.append(ece_c)
    return brier, float(top_ece), float(np.mean(cw_eces))

def bootstrap_test_f1_and_min_rec_ci(y_true, preds, n_boot=1000):
    y_arr, p_arr = np.asarray(y_true), np.asarray(preds)
    f1_vals, min_rec_vals = [], []
    for i in range(n_boot):
        idx = resample(np.arange(len(y_arr)), random_state=i)
        f1_vals.append(f1_score(y_arr[idx], p_arr[idx], average='macro', labels=ALL_LABELS, zero_division=0))
        rec_c = recall_score(y_arr[idx], p_arr[idx], average=None, labels=ALL_LABELS, zero_division=0)
        min_rec_vals.append(np.mean([rec_c[c] for c in MINORITY_IDS]))
    return (
        np.percentile(f1_vals, 2.5), np.percentile(f1_vals, 97.5),
        np.mean(min_rec_vals), np.percentile(min_rec_vals, 2.5), np.percentile(min_rec_vals, 97.5)
    )

# 3. De-Confounded Test Set Evaluation Suite
test_eval_suite = [
    ('1. Untuned LightGBM (Base-48, No-Resample)',      'Unadjusted', test_probs_lgbm_base),
    ('2. Untuned XGBoost  (Base-48, No-Resample)',      'Unadjusted', test_probs_xgb_base),
    ('3. Tuned LightGBM   (Base-48, Single Fit)',       'Unadjusted', pr_lgbm_b48_single),
    ('4. Tuned LightGBM   (Base-48, Bagged)',           'Unadjusted', test_fe_off_accum['FE_OFF_Expert1_Tuned_LGBM (Base-48)']),
    ('5. Tuned LightGBM   (FE:ON,   Single Fit)',       'Unadjusted', pr_lgbm_fe_single),
    ('6. Tuned LightGBM   (FE:ON,   Bagged)',           'Unadjusted', hyb_test_list[0]),
    ('7. Tuned XGBoost    (FE:ON,   Bagged)',           'Unadjusted', hyb_test_list[1]),
    ('8. H3: Aligned Simplex Neuro-Booster (Bagged)',   'Unadjusted', hyb_test_list[4]),
    ('9. H1: Simplex Voting Base (F01)',                'Unadjusted', test_fact['F01_Voting   | Pairwise: OFF | Tau: OFF']),
    ('10. H2: Plain Tuned LR Stack (F05)',              'Unadjusted', test_fact['F05_Std_LR   | Pairwise: OFF | Tau: OFF']),
    ('11. H4 Base: NeuroCLR Only (F09)',                'Unadjusted', test_fact['F09_NeuroCLR | Pairwise: OFF | Tau: OFF']),
    ('12. H2+: Std LR + Pairwise + Tau (F08)',          'Adjusted*',  test_fact['F08_Std_LR   | Pairwise: ON  | Tau: ON']),
    ('13. H4 (FE:OFF): NeuroCLR+PW+Tau (Base-48)',      'Adjusted*',  test_fe_off_accum['FE_OFF_F12_NeuroCLR+PW+Tau(Base-48)']),
    ('14. H4 Primary: NeuroCLR+PW+Tau (F12, FE:ON)',    'Adjusted*',  test_fact['F12_NeuroCLR | Pairwise: ON  | Tau: ON'])
]

test_rows = []
for m_name, prob_type, m_probs in test_eval_suite:
    m_preds = np.argmax(m_probs, axis=1)
    acc  = accuracy_score(hyb_y_test, m_preds)
    bacc = balanced_accuracy_score(hyb_y_test, m_preds)
    f1m  = f1_score(hyb_y_test, m_preds, average='macro', labels=ALL_LABELS, zero_division=0)
    f1_l, f1_u, min_rec_m, mr_l, mr_u = bootstrap_test_f1_and_min_rec_ci(hyb_y_test, m_preds)
    mcc  = matthews_corrcoef(hyb_y_test, m_preds)
    roc  = roc_auc_score(hyb_y_test, m_probs, multi_class='ovr', labels=ALL_LABELS)
    brier, top_ece, cw_ece = compute_calibration_metrics(hyb_y_test, m_probs)
    test_rows.append({
        'Model': m_name,
        'Prob_Type': prob_type,
        'Acc': acc,
        'Bal-Acc': bacc,
        'Macro-F1': f1m,
        'F1 95% CI': f"[{f1_l:.4f}, {f1_u:.4f}]",
        'Min-3 Rec': min_rec_m,
        'Min-3 Rec CI': f"[{mr_l:.4f}, {mr_u:.4f}]",
        'MCC': mcc,
        'ROC-AUC': roc,
        'Brier↓': brier,
        'Top-ECE↓': top_ece,
        'CW-ECE↓': cw_ece
    })

test_results_df = pd.DataFrame(test_rows)
print("\n" + "="*152)
print(" "*36 + "CONFIRMATORY HELD-OUT TEST SET RESULTS & CALIBRATION TABLE")
print("     (*Note: 'Adjusted*' models apply Pairwise+Tau decision scaling for classification; evaluate calibration on 'Unadjusted' posteriors F05/F09/LGBM)")
print("="*152)
print(test_results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("="*152)
test_results_df.to_csv(os.path.join(data_path, '16_final_test_results.csv'), index=False)