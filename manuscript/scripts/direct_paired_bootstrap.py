"""Run in the original notebook kernel with: %run -i manuscript/scripts/direct_paired_bootstrap.py"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils import resample

required = ['hyb_y_test', 'test_fact', 'hyb_test_list']
missing = [name for name in required if name not in globals()]
if not missing:
    y = np.asarray(hyb_y_test)
    p09 = np.asarray(test_fact['F09_NeuroCLR | Pairwise: OFF | Tau: OFF'])
    p05 = np.asarray(test_fact['F05_Std_LR   | Pairwise: OFF | Tau: OFF'])
    plgb = np.asarray(hyb_test_list[0])
elif len(missing) == len(required):
    data_dir = Path(__file__).resolve().parents[2] / 'outputs' / 'data'
    archive = next((p for p in (data_dir/'direct_neuroclr_predictions.npz',
                                data_dir/'section14_neuroclr_predictions.npz')
                    if p.is_file()), None)
    if archive is None:
        raise RuntimeError("Run Section 14 to create the prediction archive.")
    with np.load(archive, allow_pickle=False) as data:
        y,p09,p05,plgb = (data[key].copy() for key in ('y','p09','p05','plgb'))
    print("Using predictions from",archive)
else:
    raise RuntimeError("Kernel has an incomplete set of fitted arrays: " + ", ".join(missing))
assert y.shape == (1375,)
for p in [p09,p05,plgb]:
    assert p.shape == (1375,5) and np.isfinite(p).all()
    assert np.all(p >= 0) and np.allclose(p.sum(axis=1),1,atol=1e-5)
preds = [p.argmax(axis=1) for p in [p09,p05,plgb]]
for pred,expected in zip(preds,[0.9000472549042684,0.8939748209277003,0.8841528444226144]):
    assert np.isclose(f1_score(y,pred,average='macro',labels=np.arange(5),zero_division=0),expected,atol=1e-10), "Prediction state differs from manuscript run"
indices = [resample(np.arange(len(y)),random_state=i) for i in range(1000)]
rows=[]
for name,pb in zip(['Probability-only stacking','Enriched bagged LightGBM'],preds[1:]):
    for metric,fn in [('Macro-F1',lambda a,b:f1_score(a,b,average='macro',labels=np.arange(5),zero_division=0)),('Balanced accuracy',balanced_accuracy_score)]:
        vals=np.array([fn(y[ix],preds[0][ix])-fn(y[ix],pb[ix]) for ix in indices])
        rows.append(dict(comparison='Unadjusted NeuroCLR minus '+name,metric=metric,
                         point_difference=fn(y,preds[0])-fn(y,pb),
                         bootstrap_mean=vals.mean(),ci_lower=np.percentile(vals,2.5),
                         ci_upper=np.percentile(vals,97.5),resamples=1000))
out=Path(__file__).resolve().parents[2] / 'outputs' / 'data'
out.mkdir(parents=True,exist_ok=True)
result=pd.DataFrame(rows)
result.to_csv(out/'direct_neuroclr_paired_bootstrap.csv',index=False)
np.savez_compressed(out/'direct_neuroclr_predictions.npz',y=y,p09=p09,p05=p05,plgb=plgb)
deferred = p09.max(axis=1) < 0.640
wrong = preds[0] != y
audit = dict(total=len(y),total_errors=int(wrong.sum()),deferred=int(deferred.sum()),
             deferred_errors=int((wrong & deferred).sum()),retained_errors=int((wrong & ~deferred).sum()),
             error_capture=float((wrong & deferred).sum()/wrong.sum()),
             deferred_error_rate=float(wrong[deferred].mean()))
pd.DataFrame([audit]).to_csv(out/'direct_deferral_error_audit.csv',index=False)
print(result.to_string(index=False))
print(audit)

# Optional retrospective contrast. The direct prediction archive does not contain this arm.
base34_key = 'FE_OFF_F12_NeuroCLR+PW+Tau(Base-48)'
if not missing and base34_key in globals().get('test_fe_off_accum', {}):
    base34 = np.asarray(test_fe_off_accum[base34_key])
    assert base34.shape == p09.shape and np.isfinite(base34).all()
    assert np.all(base34 >= 0) and np.allclose(base34.sum(axis=1), 1)
    base_pred = base34.argmax(axis=1)
    def base_macro(labels, pred):
        return f1_score(labels, pred, average='macro', labels=np.arange(5), zero_division=0)
    assert np.isclose(base_macro(y, base_pred), 0.907820508505625, rtol=0, atol=1e-10)
    differences = np.array([base_macro(y[ix], base_pred[ix]) - base_macro(y[ix], preds[0][ix]) for ix in indices])
    base_result = pd.DataFrame([dict(comparison='Adjusted Base-34 NeuroCLR minus unadjusted enriched NeuroCLR',
                                metric='Macro-F1', point_difference=base_macro(y, base_pred)-base_macro(y, preds[0]),
                                bootstrap_mean=differences.mean(), ci_lower=np.percentile(differences,2.5),
                                ci_upper=np.percentile(differences,97.5), resamples=len(indices))])
    np.savez_compressed(out/'base34_adjusted_paired_predictions.npz', y=y, pbase34=base34, p09=p09)
    base_result.to_csv(out/'base34_adjusted_paired_bootstrap.csv',index=False)
    print(base_result.to_string(index=False))
