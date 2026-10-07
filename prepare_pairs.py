"""Join waveform quality decisions to serial ECG pairs."""

import argparse
import json
from pathlib import Path

import duckdb


from paths import WORK

DEFAULT_OUT = WORK


def prepare_heedb(con: duckdb.DuckDBPyConnection, root: Path, site: str) -> None:
    pairs = root / f'heedb_{site}_pairs_1_6h.parquet'
    qc = root / f'beats_{site}' / 'qc_*.parquet'
    output = root / f'heedb_{site}_valid_pairs.parquet'
    query = f"""
        SELECT p.dataset, p.patient_id, p.record_a, p.record_b,
               p.path_a, p.path_b, p.time_a, p.time_b,
               p.gap_seconds, p.split,
               qa.shard AS shard_a, qa.row_in_shard AS row_a,
               qb.shard AS shard_b, qb.row_in_shard AS row_b,
               qa.heart_rate AS hr_a, qb.heart_rate AS hr_b,
               qa.sex_label
        FROM read_parquet('{pairs}') p
        JOIN read_parquet('{qc}') qa ON p.path_a=qa.waveform_path AND qa.ok
        JOIN read_parquet('{qc}') qb ON p.path_b=qb.waveform_path AND qb.ok
    """
    con.sql(query).write_parquet(str(output), compression='zstd')
    counts = con.execute(
        'SELECT split,count(*),count(DISTINCT patient_id) '
        'FROM read_parquet(?) GROUP BY 1 ORDER BY 1', [str(output)]
    ).fetchall()
    print(json.dumps({'dataset': site, 'valid_pairs_by_split': counts}), flush=True)


def prepare_mimic(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    ecgs = root / 'mimic_external_ecgs.parquet'
    qc = root / 'beats_mimic' / 'qc_*.parquet'
    output = root / 'mimic_valid_pairs.parquet'
    query = f"""
        WITH source AS (
            SELECT a.stay_id, a.subject_id,
                   a.record_id AS record_a, b.record_id AS record_b,
                   a.waveform_path AS path_a, b.waveform_path AS path_b,
                   a.acq_ts AS time_a, b.acq_ts AS time_b,
                   date_diff('second', a.acq_ts, b.acq_ts) AS gap_seconds,
                   a.sex_label
            FROM read_parquet('{ecgs}') a
            JOIN read_parquet('{ecgs}') b
              ON a.stay_id=b.stay_id
             AND b.acq_ts BETWEEN a.acq_ts + INTERVAL '1 hour'
                              AND a.acq_ts + INTERVAL '6 hours'
            WHERE a.record_id <> b.record_id
        )
        SELECT 'MIMIC' AS dataset,
               'MIMIC:' || CAST(p.subject_id AS VARCHAR) AS patient_id,
               p.stay_id, p.record_a, p.record_b,
               p.path_a, p.path_b, p.time_a, p.time_b,
               p.gap_seconds, 'external' AS split,
               qa.shard AS shard_a, qa.row_in_shard AS row_a,
               qb.shard AS shard_b, qb.row_in_shard AS row_b,
               qa.heart_rate AS hr_a, qb.heart_rate AS hr_b,
               p.sex_label
        FROM source p
        JOIN read_parquet('{qc}') qa ON p.path_a=qa.waveform_path AND qa.ok
        JOIN read_parquet('{qc}') qb ON p.path_b=qb.waveform_path AND qb.ok
    """
    con.sql(query).write_parquet(str(output), compression='zstd')
    counts = con.execute(
        'SELECT count(*),count(DISTINCT patient_id),count(DISTINCT stay_id) '
        'FROM read_parquet(?)', [str(output)]
    ).fetchone()
    print(json.dumps({'dataset': 'MIMIC', 'valid_pairs': counts[0],
                      'patients': counts[1], 'stays': counts[2]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--part', choices=('i0001', 'i0006', 'mimic', 'all'), default='all')
    args = parser.parse_args()
    con = duckdb.connect()
    con.execute('SET threads=16')
    if args.part in ('i0001', 'all'):
        prepare_heedb(con, args.out, 'i0001')
    if args.part in ('i0006', 'all'):
        prepare_heedb(con, args.out, 'i0006')
    if args.part in ('mimic', 'all'):
        prepare_mimic(con, args.out)


if __name__ == '__main__':
    main()
