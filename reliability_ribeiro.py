"""Score the published Ribeiro et al. (Nat Commun 2020) 12-lead abnormality model.

Weights: Zenodo doi:10.5281/zenodo.3765717 (CC-BY-4.0), unzipped to WORK/ribeiro.
Run inside WORK/venv_tf (TensorFlow; no torch).

Outputs per ECG: probabilities of 1st-degree AV block, RBBB, LBBB, sinus
bradycardia, atrial fibrillation, and sinus tachycardia (published order).

Actions:
  validate   choose the amplitude multiplier on 5000 I0001 validation ECGs,
             using AUC against GE 12SL v24 statements
  score      all ECGs in rel_records.parquet -> rel_scores_ribeiro/*.parquet
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ecg_io import read_ecg, to_lima
from paths import HEEDB_ROOT, WORK


MODEL = WORK / 'ribeiro' / 'model' / 'model.hdf5'
HEEDB = HEEDB_ROOT / 'ECG'
CLASSES = ['avb1', 'rbbb', 'lbbb', 'sb', 'af', 'st']
CODES_12SL = {'avb1': [101], 'rbbb': [440, 442], 'lbbb': [460], 'sb': [21, 24],
              'af': [161], 'st': [23]}
MULTIPLIERS = (1.0, 10.0)


def load_model():
    try:
        from tensorflow.keras.models import load_model as keras_load
        return keras_load(MODEL, compile=False)
    except Exception:
        import tf_keras
        return tf_keras.models.load_model(MODEL, compile=False)


def _read(path: str) -> np.ndarray | None:
    x, fs, _ = read_ecg(path)
    return None if x is None else to_lima(x, fs).T.astype(np.float32)   # 4096 x 12


def predict(model, paths: list[str], multiplier: float, workers: int,
            batch: int = 512) -> np.ndarray:
    out = np.full((len(paths), len(CLASSES)), np.nan, dtype=np.float32)
    with ProcessPoolExecutor(workers) as pool:
        for start in range(0, len(paths), batch * 8):
            chunk = paths[start:start + batch * 8]
            arrays = list(pool.map(_read, chunk, chunksize=16))
            idx = [i for i, a in enumerate(arrays) if a is not None]
            if not idx:
                continue
            x = np.stack([arrays[i] for i in idx]) * multiplier
            p = model.predict(x, batch_size=batch, verbose=0)
            out[np.array(idx) + start] = p
    return out


def labels_12sl(stems: pd.Series) -> pd.DataFrame:
    path = HEEDB / 'I0001' / '12SL_diagnoses' / 'diagnoses_v24.csv'
    codes = duckdb.sql(f"""
        SELECT regexp_extract(trim(FileName), '([^/]+?)(\\.hea)?\\s*$', 1) AS stem,
               list_transform(string_split(codes, ','), x -> TRY_CAST(trim(x) AS INTEGER)) AS c
        FROM read_csv('{path}', header=true, all_varchar=true, quote='"',
                      strict_mode=false, parallel=false)""").df()
    codes = codes[codes['stem'].isin(set(stems))]
    for name, values in CODES_12SL.items():
        codes[name] = codes['c'].apply(lambda c: any(v in c for v in values) if c is not None else False)
    return codes.drop(columns='c')


def validate(workers: int, n: int) -> None:
    from sklearn.metrics import roc_auc_score
    frame = pd.read_parquet(WORK / 'rel_val_records.parquet').head(n)
    frame['stem'] = frame['waveform_path'].str.extract(r'([^/]+)$')[0]
    lab = frame[['stem']].merge(labels_12sl(frame['stem']), on='stem', how='left')
    model = load_model()
    result = {'n': n}
    for m in MULTIPLIERS:
        p = predict(model, frame['waveform_path'].tolist(), m, workers)
        ok = np.isfinite(p).all(axis=1) & lab['avb1'].notna().to_numpy()
        aucs = {}
        for j, c in enumerate(CLASSES):
            y = lab[c].fillna(False).to_numpy(bool)[ok]
            if 10 <= y.sum() < len(y):
                aucs[c] = float(roc_auc_score(y, p[ok, j]))
        result[f'x{m:g}'] = {'auc': aucs, 'mean_auc': float(np.mean(list(aucs.values()))),
                             'positives': {c: int(lab[c].fillna(False)[ok].sum()) for c in CLASSES}}
    chosen = max(MULTIPLIERS, key=lambda m: result[f'x{m:g}']['mean_auc'])
    result['multiplier'] = chosen
    (WORK / 'rel_ribeiro_validation.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


def score(workers: int, shard_size: int) -> None:
    multiplier = json.loads((WORK / 'rel_ribeiro_validation.json').read_text())['multiplier']
    paths = sorted(pd.read_parquet(WORK / 'rel_records.parquet')['waveform_path'].unique())
    model = load_model()
    out_dir = WORK / 'rel_scores_ribeiro'
    out_dir.mkdir(exist_ok=True)
    for start in range(0, len(paths), shard_size):
        target = out_dir / f'ribeiro_{start // shard_size:04d}.parquet'
        if target.exists():
            continue
        chunk = paths[start:start + shard_size]
        p = predict(model, chunk, multiplier, workers)
        frame = pd.DataFrame(p, columns=[f'ribeiro_{c}' for c in CLASSES])
        frame.insert(0, 'waveform_path', chunk)
        frame.to_parquet(target, index=False)
        print(json.dumps({'done': start + len(chunk), 'total': len(paths)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'score'))
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--shard-size', type=int, default=50000)
    args = parser.parse_args()
    if args.action == 'validate':
        validate(args.workers, args.n)
    else:
        score(args.workers, args.shard_size)


if __name__ == '__main__':
    main()
