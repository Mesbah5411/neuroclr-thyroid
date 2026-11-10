"""Load the enriched, unadjusted NeuroCLR ensemble trained in notebook Section 14."""
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from joblib.externals import cloudpickle

MODEL_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models"


def load_neuroclr(path=None):
    """Load a checkpoint produced by this notebook in the training environment."""
    if path is None:
        path = MODEL_DIR / (MODEL_DIR / "latest_f09.txt").read_text().strip()
    with Path(path).open("rb") as stream:
        return cloudpickle.load(stream)


def predict_proba(model, base_features):
    """Predict from a DataFrame with the same encoded base features as Section 14."""
    import torch
    cols = model["columns"]
    missing = [c for c in cols if c not in base_features]
    if missing or not base_features.columns.is_unique:
        raise ValueError(f"Expected unique Section 14 base columns. Missing: {missing}")
    X = base_features.loc[:, cols]
    seed_probs = []
    for stage1, stage2 in model["seeds"]:
        by_expert = [[] for _ in range(5)]
        for quantile, imputer, scaler, trees, nets, modes in stage1:
            raw = quantile.transform(X)
            scaled = scaler.transform(imputer.transform(raw))
            for k in range(3):
                inputs = scaled if modes[k] == "smote" else raw
                by_expert[k].append(trees[k].predict_proba(inputs))
            by_expert[3].append(trees[3].predict_proba(scaled))
            with torch.no_grad():
                # Match the notebook's float64 accumulation of float32 outputs.
                neural = np.zeros((len(X),len(model["classes"])))
                tensor = torch.as_tensor(scaled,dtype=torch.float32)
                for net in nets:
                    neural += torch.softmax(net.eval()(tensor),dim=1).cpu().numpy() / len(nets)
            augmented = np.hstack([scaled if modes[0]=="smote" else raw.values,
                                   neural, model["uncertainty"](neural)])
            by_expert[4].append(trees[4].predict_proba(augmented))
        features = model["meta_features"]([np.mean(p,axis=0) for p in by_expert],
                                          use_neurologit=True)
        seed_probs.append(np.mean([lr.predict_proba(sc.transform(features))
                                   for sc,lr in stage2],axis=0))
    return np.mean(seed_probs,axis=0)


def save_neuroclr(model, check_features, expected):
    """Publish a new checkpoint only after reloading and comparing its probabilities."""
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    name = 'neuroclr_f09_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_') + uuid4().hex[:8]
    target=MODEL_DIR/(name+'.pkl')
    pending=target.with_suffix('.partial')
    model["versions"]={k:version(k) for k in ['numpy','pandas','scikit-learn','torch','lightgbm','xgboost','catboost','joblib']}
    model["python"]=platform.python_version()
    try:
        with pending.open("xb") as stream:
            cloudpickle.dump(model,stream)
        restored=load_neuroclr(pending)
        np.testing.assert_allclose(predict_proba(restored,check_features),expected,
                                   rtol=1e-5,atol=1e-7)
        pending.replace(target)
        pointer=MODEL_DIR/(name+'.txt')
        pointer.write_text(target.name,encoding='utf-8')
        os.replace(pointer,MODEL_DIR/'latest_f09.txt')
    finally:
        pending.unlink(missing_ok=True)
    return target
