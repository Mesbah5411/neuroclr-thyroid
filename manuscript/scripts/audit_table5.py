"""Audit Results/Table 5 without retraining. Run from any working directory.

    python manuscript/scripts/audit_table5.py

Optional from the completed Section 16 notebook kernel:
    from audit_table5 import save_all_probabilities
    save_all_probabilities(hyb_y_test, test_eval_suite)

The saved cell uses test_eval_suite in its original ordering. A full archive
is accepted only if all metrics agree with the current full-precision export.
"""
from pathlib import Path
import csv
import json
import re
import numpy as np
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs' / 'data'
NOTEBOOK = ROOT / 'thyroid-journal_code.ipynb'
CELL_ID = '47939619-09fe-4741-8ee6-d4bb64c7bfc8'
KEYS = ['Brier↓', 'Top-ECE↓', 'CW-ECE↓']


def section16(path):
    notebook = json.loads(path.read_text(encoding='utf-8-sig'))
    return next(c for c in notebook['cells'] if c.get('id') == CELL_ID)


def metric_function(cell):
    import ast
    module = ast.parse(''.join(cell['source']))
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef)
              and n.name == 'compute_calibration_metrics')
    scope = {'np': np, 'label_binarize': label_binarize, 'ALL_LABELS': np.arange(5)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'Section16', 'exec'), scope)
    return scope[fn.name]


def read_rows():
    with (DATA/'16_final_test_results.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save_all_probabilities(y, suite):
    rows = read_rows()
    assert len(rows) == len(suite) == 14
    arrays = {'y': np.asarray(y)}
    fn = metric_function(section16(NOTEBOOK))
    for i, ((name, kind, probs), row) in enumerate(zip(suite,rows)):
        assert name == row['Model'] and kind == row['Prob_Type']
        np.testing.assert_allclose(fn(y,probs), [float(row[k]) for k in KEYS],
                                   rtol=1e-8,atol=1e-10)
        arrays[f'p{i:02d}'] = np.asarray(probs)
    tmp = DATA/'table5_all_probabilities.pending.npz'
    np.savez_compressed(tmp,**arrays)
    tmp.replace(DATA/'table5_all_probabilities.npz')


def audit():
    cell = section16(NOTEBOOK)
    src = ''.join(cell['source'])
    assert "for m_name, prob_type, m_probs in test_eval_suite:" in src
    assert "compute_calibration_metrics(hyb_y_test, m_probs)" in src
    text = ''.join(''.join(o.get('text','')) for o in cell['outputs'])
    fn = metric_function(cell)
    rows = read_rows()
    assert len(rows) == 14
    manuscript=(ROOT/'manuscript/sections/05_results.tex').read_text()
    table=manuscript.split(r'\label{tab:test_probability}')[1].split(r'\end{tabular}')[0]
    lines=[l for l in table.splitlines() if re.search(r' & 0\.\d{6} & ',l)]
    assert len(lines) == len(rows)
    direct=DATA/'direct_neuroclr_predictions.npz'
    full=DATA/'table5_all_probabilities.npz'
    records=[]
    with np.load(full if full.exists() else direct,allow_pickle=False) as archive:
        for i,(row,line) in enumerate(zip(rows,lines)):
            expected=[float(row[k]) for k in KEYS]
            printed=next(l for l in text.splitlines() if l.strip().startswith(row['Model']+' '))
            assert printed.split()[-3:] == [f'{v:.4f}' for v in expected],row['Model']
            assert [v.strip().strip('\\').strip() for v in line.split('&')[1:]] == [
                f'{expected[0]:.6f}',f'{expected[1]:.4f}',f'{expected[2]:.4f}'],row['Model']
            key=f'p{i:02d}' if full.exists() else {5:'plgb',9:'p05',10:'p09'}.get(i)
            if key is not None:
                np.testing.assert_allclose(fn(archive['y'],archive[key]),expected,
                                           rtol=1e-8,atol=1e-10)
            records.append(dict(model=row['Model'],probability_type=row['Prob_Type'],
                brier=row[KEYS[0]],top_label_ece=row[KEYS[1]],classwise_ece=row[KEYS[2]],
                notebook_output_matches=True,manuscript_matches=True,
                raw_probability_check='matched' if key else 'matrix not archived'))
    with (DATA/'table5_calibration_audit.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    print('Matched %d Section 16 output and manuscript rows. Raw matrices checked: %d.' %
          (len(records),sum(r['raw_probability_check']=='matched' for r in records)))


if __name__=='__main__':
    audit()
