"""Extract four fixed physical-unit beats from selected WFDB ECGs.

Each shard has a fixed-shape float16 beat array and a patient-level QC table.
Run only on manifests stored outside the source tree.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import wfdb
from scipy.signal import resample_poly
from wfdb.processing import xqrs_detect


LEADS = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
EXPECTED = {'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'}
TARGET_FS = 250
LEFT = 75
RIGHT = 125
N_BEATS = 4


def peaks_for_signal(x: np.ndarray, fs: int, channels: list[str]) -> np.ndarray:
    for lead in ('II', 'V5'):
        sig = x[:, channels.index(lead)]
        try:
            peaks = xqrs_detect(sig=sig, fs=fs, verbose=False)
        except Exception:
            continue
        peaks = peaks[(peaks >= int(0.30 * fs)) &
                      (peaks < len(sig) - int(0.50 * fs))]
        if 4 <= len(peaks) <= 25:
            return peaks
    return np.empty(0, dtype=np.int64)


def extract_one(path: str) -> tuple[np.ndarray, dict]:
    beats = np.zeros((N_BEATS, len(LEADS), LEFT + RIGHT), dtype=np.float16)
    result = {'ok': False, 'reason': 'unknown', 'n_peaks': 0,
              'heart_rate': None, 'rr_cv': None, 'fs': None}
    try:
        x, hdr = wfdb.rdsamp(path)
        fs = int(hdr['fs'])
        channels = list(hdr['sig_name'])
        result['fs'] = fs
        if fs not in (250, 500) or len(x) / fs < 9.5 or len(x) / fs > 10.5:
            result['reason'] = 'format_or_duration'
            return beats, result
        if len(channels) != 12 or set(channels) != EXPECTED:
            result['reason'] = 'lead_set'
            return beats, result
        if any(unit != 'mV' for unit in hdr['units']):
            result['reason'] = 'unit'
            return beats, result
        if not np.isfinite(x).all():
            result['reason'] = 'nonfinite'
            return beats, result
        if np.any(np.std(x, axis=0) < 0.005):
            result['reason'] = 'flat_lead'
            return beats, result
        if np.max(np.abs(x)) > 20:
            result['reason'] = 'extreme_amplitude'
            return beats, result
        peaks = peaks_for_signal(x, fs, channels)
        result['n_peaks'] = int(len(peaks))
        if len(peaks) < N_BEATS:
            result['reason'] = 'few_peaks'
            return beats, result
        rr = np.diff(peaks) / fs
        result['heart_rate'] = float(60 / np.median(rr))
        result['rr_cv'] = float(np.std(rr) / np.mean(rr))
        if np.median(rr) < 0.35 or np.median(rr) > 2.0:
            result['reason'] = 'rr_range'
            return beats, result
        if result['rr_cv'] > 0.30:
            result['reason'] = 'irregular_rr'
            return beats, result
        # Keep voltage scale; subtract only constant ADC/baseline offset.
        x = x[:, [channels.index(lead) for lead in LEADS]].astype(np.float32)
        x -= np.median(x, axis=0, keepdims=True)
        if fs == 500:
            x = resample_poly(x, 1, 2, axis=0)
            peaks = np.rint(peaks / 2).astype(np.int64)
        valid = peaks[(peaks >= LEFT) & (peaks + RIGHT <= len(x))]
        if len(valid) < N_BEATS:
            result['reason'] = 'few_full_beats'
            return beats, result
        chosen = valid[np.rint(np.linspace(0, len(valid) - 1, N_BEATS)).astype(int)]
        for j, peak in enumerate(chosen):
            beats[j] = x[peak - LEFT:peak + RIGHT].T.astype(np.float16)
        result['ok'] = True
        result['reason'] = ''
        return beats, result
    except Exception as exc:
        result['reason'] = f'read_error:{type(exc).__name__}'
        return beats, result


def run_shard(args: tuple[int, list[dict], str]) -> dict:
    shard_id, records, out_string = args
    out = Path(out_string)
    beat_path = out / f'beats_{shard_id:05d}.npy'
    qc_path = out / f'qc_{shard_id:05d}.parquet'
    if beat_path.exists() and qc_path.exists():
        return {'shard': shard_id, 'skipped': True}
    arr = np.zeros((len(records), N_BEATS, len(LEADS), LEFT + RIGHT), dtype=np.float16)
    qc_rows = []
    for i, rec in enumerate(records):
        arr[i], status = extract_one(rec['waveform_path'])
        qc_rows.append({**rec, **status, 'shard': shard_id, 'row_in_shard': i})
    temp_beat = out / f'beats_{shard_id:05d}.npy.tmp'
    temp_qc = out / f'qc_{shard_id:05d}.parquet.tmp'
    with temp_beat.open('wb') as f:
        np.save(f, arr)
    pq.write_table(pa.Table.from_pylist(qc_rows), temp_qc, compression='zstd')
    os.replace(temp_beat, beat_path)
    os.replace(temp_qc, qc_path)
    return {'shard': shard_id, 'n': len(records),
            'ok': sum(row['ok'] for row in qc_rows), 'skipped': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--shard-size', type=int, default=5000)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    schema = pq.read_schema(args.manifest)
    patient_column = 'patient_id' if 'patient_id' in schema.names else 'subject_id'
    columns = ['waveform_path', patient_column, 'record_id', 'sex_label', 'age_years']
    if 'split' in schema.names:
        columns.append('split')
    if 'stay_id' in schema.names:
        columns.append('stay_id')
    table = pq.read_table(args.manifest, columns=columns)
    if args.limit is not None:
        table = table.slice(0, args.limit)
    records = table.to_pylist()
    if patient_column != 'patient_id':
        for rec in records:
            rec['patient_id'] = rec.pop(patient_column)
    jobs = [(i // args.shard_size,
             records[i:i + args.shard_size], str(args.out))
            for i in range(0, len(records), args.shard_size)]
    completed = 0
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_shard, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            n_ok += result.get('ok', 0)
            if completed % max(1, len(jobs) // 20) == 0 or completed == len(jobs):
                print(json.dumps({'completed_shards': completed,
                                  'total_shards': len(jobs),
                                  'new_ok': n_ok}), flush=True)


if __name__ == '__main__':
    main()
