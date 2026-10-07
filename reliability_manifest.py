"""Build interval-stratified consecutive ECG pairs for the reliability study.

Outputs (all under the restricted work directory):
  rel_pairs.parquet      one consecutive pair per patient per interval bin
  rel_records.parquet    unique ECGs to score
  rel_train_records.parquet / rel_val_records.parquet
                         one ECG per I0001 train/val patient for model fitting
See reliability_protocol.md for the pre-specified definitions.
"""

import argparse
import json
from pathlib import Path

import duckdb

from paths import WORK


MAX_PATIENTS_PER_BIN = 25000
BIN_SQL = """
    CASE WHEN gap_min < 5 THEN '1_lt5m'
         WHEN gap_min < 60 THEN '2_5_60m'
         WHEN gap_min < 360 THEN '3_1_6h'
         WHEN gap_min < 1440 THEN '4_6_24h'
         WHEN gap_min < 10080 THEN '5_1_7d'
         WHEN gap_min < 43200 THEN '6_7_30d'
         WHEN gap_min < 525600 THEN '7_30d_1y'
         ELSE '8_gt1y' END
"""
SITES = {
    'i0001': ('heedb_i0001_records.parquet', "split = 'test'"),
    'i0006': ('heedb_i0006_records.parquet', 'TRUE'),
    'mimic': ('mimic_records.parquet', 'TRUE'),
}


def site_pairs(site: str, source: Path, where: str) -> str:
    return f"""
        WITH r AS (
            SELECT CAST(patient_id AS VARCHAR) AS patient_id,
                   CAST(record_id AS VARCHAR) AS record_id,
                   acq_ts, waveform_path, sex_label,
                   CAST(age_years AS DOUBLE) AS age_years
            FROM read_parquet('{source}')
            WHERE {where} AND patient_id IS NOT NULL AND acq_ts IS NOT NULL
        ), ordered AS (
            SELECT *,
                   lead(record_id) OVER w AS record_b,
                   lead(acq_ts) OVER w AS time_b,
                   lead(waveform_path) OVER w AS path_b,
                   lead(age_years) OVER w AS age_b,
                   lead(sex_label) OVER w AS sex_b
            FROM r WINDOW w AS (PARTITION BY patient_id
                                ORDER BY acq_ts, record_id)
        ), pairs AS (
            SELECT '{site}' AS site, patient_id,
                   record_id AS record_a, record_b,
                   waveform_path AS path_a, path_b,
                   acq_ts AS time_a, time_b,
                   epoch(time_b - acq_ts) / 60.0 AS gap_min,
                   age_years AS age_a, age_b,
                   sex_label AS sex_a, sex_b
            FROM ordered
            WHERE record_b IS NOT NULL AND time_b > acq_ts
        ), binned AS (
            SELECT *, {BIN_SQL} AS bin FROM pairs
        ), one_per_patient AS (
            SELECT * FROM binned
            QUALIFY row_number() OVER (
                PARTITION BY patient_id, bin
                ORDER BY hash(record_a || '|' || record_b)) = 1
        )
        SELECT * FROM one_per_patient
        QUALIFY row_number() OVER (
            PARTITION BY bin ORDER BY hash('{site}' || patient_id)
        ) <= {MAX_PATIENTS_PER_BIN}
    """


def site_triples(site: str, source: Path, where: str) -> str:
    """Three consecutive ECGs spanning <6 h; one triple per patient."""
    return f"""
        WITH r AS (
            SELECT CAST(patient_id AS VARCHAR) AS patient_id,
                   CAST(record_id AS VARCHAR) AS record_id,
                   acq_ts, waveform_path, sex_label,
                   CAST(age_years AS DOUBLE) AS age_years
            FROM read_parquet('{source}')
            WHERE {where} AND patient_id IS NOT NULL AND acq_ts IS NOT NULL
        ), ordered AS (
            SELECT *,
                   lead(acq_ts, 1) OVER w AS time_b, lead(acq_ts, 2) OVER w AS time_c,
                   lead(waveform_path, 1) OVER w AS path_b,
                   lead(waveform_path, 2) OVER w AS path_c
            FROM r WINDOW w AS (PARTITION BY patient_id ORDER BY acq_ts, record_id)
        ), triples AS (
            SELECT '{site}' AS site, patient_id, waveform_path AS path_a,
                   path_b, path_c, acq_ts AS time_a, time_b, time_c,
                   age_years AS age_a, sex_label AS sex_a
            FROM ordered
            WHERE path_c IS NOT NULL AND time_b > acq_ts AND time_c > time_b
              AND time_c - acq_ts < INTERVAL 6 HOURS
            QUALIFY row_number() OVER (
                PARTITION BY patient_id ORDER BY hash(waveform_path)) = 1
        )
        SELECT * FROM triples
        ORDER BY hash('{site}' || patient_id) LIMIT {MAX_PATIENTS_PER_BIN}
    """


def ed_pairs(work: Path) -> str:
    return f"""
        WITH stable AS (
            SELECT * FROM read_parquet('{work / 'mimic_pairs_clinical_flags.parquet'}')
            WHERE stable_primary
            QUALIFY row_number() OVER (
                PARTITION BY patient_id
                ORDER BY hash(CAST(record_a AS VARCHAR) || '|' ||
                              CAST(record_b AS VARCHAR))) = 1
        ), ages AS (
            SELECT CAST(record_id AS VARCHAR) AS record_id,
                   CAST(age_years AS DOUBLE) AS age_years
            FROM read_parquet('{work / 'mimic_records.parquet'}')
        )
        SELECT 'mimic_ed' AS site,
               CAST(regexp_replace(s.patient_id, '^MIMIC:', '') AS VARCHAR)
                   AS patient_id,
               CAST(s.record_a AS VARCHAR) AS record_a,
               CAST(s.record_b AS VARCHAR) AS record_b,
               s.path_a, s.path_b, s.time_a, s.time_b,
               s.gap_seconds / 60.0 AS gap_min,
               a.age_years AS age_a, b.age_years AS age_b,
               s.sex_label AS sex_a, s.sex_label AS sex_b,
               '3_1_6h' AS bin
        FROM stable s
        JOIN ages a ON a.record_id = CAST(s.record_a AS VARCHAR)
        JOIN ages b ON b.record_id = CAST(s.record_b AS VARCHAR)
    """


def fit_records(work: Path, split: str, n_patients: int) -> str:
    source = work / 'heedb_i0001_records.parquet'
    return f"""
        WITH r AS (
            SELECT patient_id, record_id, waveform_path, sex_label, age_years
            FROM read_parquet('{source}')
            WHERE split = '{split}' AND age_years >= 18
              AND upper(sex_label) IN ('MALE', 'FEMALE')
        ), consistent AS (
            SELECT * FROM r
            QUALIFY count(DISTINCT upper(sex_label))
                    OVER (PARTITION BY patient_id) = 1
        ), one AS (
            SELECT * FROM consistent
            QUALIFY row_number() OVER (
                PARTITION BY patient_id ORDER BY hash(record_id)) = 1
        )
        SELECT patient_id, record_id, waveform_path, age_years,
               CASE WHEN upper(sex_label) = 'MALE' THEN 1 ELSE 0 END AS sex_target
        FROM one ORDER BY hash('fit' || patient_id) LIMIT {n_patients}
    """


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--train-patients', type=int, default=300000)
    parser.add_argument('--val-patients', type=int, default=20000)
    args = parser.parse_args()
    con = duckdb.connect()
    con.execute("SET memory_limit='40GB'")
    con.execute(f"SET temp_directory='{args.work / 'duckdb_tmp'}'")
    parts = [site_pairs(site, args.work / source, where)
             for site, (source, where) in SITES.items()]
    parts.append(ed_pairs(args.work))
    con.execute('CREATE TABLE pairs AS ' +
                ' UNION ALL BY NAME '.join(f'({p})' for p in parts))
    con.execute(f"COPY pairs TO '{args.work / 'rel_pairs.parquet'}' "
                "(FORMAT parquet, COMPRESSION zstd)")
    con.execute('CREATE TABLE triples AS ' + ' UNION ALL '.join(
        f'({site_triples(site, args.work / source, where)})'
        for site, (source, where) in SITES.items()))
    con.execute(f"COPY triples TO '{args.work / 'rel_triples.parquet'}' "
                "(FORMAT parquet, COMPRESSION zstd)")
    con.execute(f"""
        COPY (
            SELECT DISTINCT site, path_a AS waveform_path FROM pairs
            UNION SELECT DISTINCT site, path_b FROM pairs
            UNION SELECT DISTINCT site, path_a FROM triples
            UNION SELECT DISTINCT site, path_b FROM triples
            UNION SELECT DISTINCT site, path_c FROM triples
        ) TO '{args.work / 'rel_records.parquet'}' (FORMAT parquet, COMPRESSION zstd)
    """)
    print(json.dumps({'triples': con.execute(
        'SELECT site, count(*) FROM triples GROUP BY 1 ORDER BY 1').fetchall()}),
        flush=True)
    for split, n in (('train', args.train_patients), ('val', args.val_patients)):
        # Cached signals and trained models index these files row by row.
        if (args.work / f'rel_{split}_records.parquet').exists():
            continue
        con.execute(f"COPY ({fit_records(args.work, split, n)}) TO "
                    f"'{args.work / f'rel_{split}_records.parquet'}' "
                    "(FORMAT parquet, COMPRESSION zstd)")
    summary = con.execute("""
        SELECT site, bin, count(*) AS pairs, count(DISTINCT patient_id) AS patients,
               round(median(gap_min), 1) AS median_gap_min
        FROM pairs GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchall()
    n_records = con.execute(
        f"SELECT count(*), count(DISTINCT waveform_path) FROM "
        f"read_parquet('{args.work / 'rel_records.parquet'}')").fetchone()
    print(json.dumps({'pairs': summary, 'records_rows_unique': n_records},
                     default=str), flush=True)


if __name__ == '__main__':
    main()
