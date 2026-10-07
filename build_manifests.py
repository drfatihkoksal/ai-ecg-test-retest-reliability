"""Build local patient-level ECG and preliminary MIMIC cohort manifests.

Outputs contain patient identifiers and must remain outside the source tree.
No waveform or patient-level table is uploaded or committed.
"""

import argparse
import json
from pathlib import Path

import duckdb


from paths import HEEDB_ROOT, MIMIC_ROOT, WORK

DEFAULT_MIMIC = MIMIC_ROOT
DEFAULT_HEEDB = HEEDB_ROOT
DEFAULT_OUT = WORK


def db(out: Path) -> duckdb.DuckDBPyConnection:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'duckdb_tmp').mkdir(exist_ok=True)
    con = duckdb.connect(str(out / 'pipeline.duckdb'))
    con.execute('SET threads=16')
    con.execute("SET memory_limit='48GB'")
    con.execute('SET temp_directory=?', [str(out / 'duckdb_tmp')])
    return con


def write_parquet(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    con.sql(sql).write_parquet(str(path), compression='zstd')


def build_heedb(con: duckdb.DuckDBPyConnection, root: Path, out: Path) -> None:
    for site in ('I0001', 'I0006'):
        csv = root / 'ECG' / site / 'metadata' / 'metadata.csv'
        sex_column = 'SexDSC' if site == 'I0001' else 'Sex'
        birth_sex_expr = ("NULLIF(TRIM(SexAssignedAtBirthDSC), '')"
                          if site == 'I0001' else 'CAST(NULL AS VARCHAR)')
        path_expr = (
            f"'{root}/ECG/I0001/WFDB' || FileName"
            if site == 'I0001'
            else f"'{root}/ECG/I0006/' || FileName"
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE heedb_{site}_records AS
            WITH raw AS (
                SELECT 'HEEDB_{site}' AS dataset,
                       'HEEDB_{site}:' || TRIM(BDSPPatientID) AS patient_id,
                       TRIM(FileID) AS record_id,
                       TRY_CAST(ECGAcquisitionTime AS TIMESTAMP) AS acq_ts,
                       {path_expr} AS waveform_path,
                       NULLIF(TRIM({sex_column}), '') AS sex_label,
                       {birth_sex_expr} AS sex_birth_label,
                       TRY_CAST(AgeAtAcquisition AS DOUBLE) / 365.25 AS age_years,
                       FileName AS source_file_name
                FROM read_csv(?, header=true, all_varchar=true)
                WHERE NULLIF(TRIM(BDSPPatientID), '') IS NOT NULL
                  AND NULLIF(TRIM(FileName), '') IS NOT NULL
                  AND TRY_CAST(ECGAcquisitionTime AS TIMESTAMP) IS NOT NULL
            )
            SELECT DISTINCT *,
                   CASE WHEN dataset='HEEDB_I0006' THEN 'site_test'
                        WHEN hash(patient_id) % 100 < 70 THEN 'train'
                        WHEN hash(patient_id) % 100 < 85 THEN 'val'
                        ELSE 'test' END AS split
            FROM raw
            """,
            [str(csv)],
        )
        write_parquet(
            con,
            f'SELECT * FROM heedb_{site}_records',
            out / f'heedb_{site.lower()}_records.parquet',
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE heedb_{site}_pairs AS
            SELECT a.dataset, a.patient_id, a.record_id AS record_a,
                   b.record_id AS record_b,
                   a.waveform_path AS path_a, b.waveform_path AS path_b,
                   a.acq_ts AS time_a, b.acq_ts AS time_b,
                   date_diff('second', a.acq_ts, b.acq_ts) AS gap_seconds,
                   a.split
            FROM heedb_{site}_records a
            JOIN heedb_{site}_records b
              ON a.patient_id=b.patient_id
             AND b.acq_ts BETWEEN a.acq_ts + INTERVAL '1 hour'
                              AND a.acq_ts + INTERVAL '6 hours'
            WHERE a.record_id <> b.record_id
            """
        )
        write_parquet(
            con,
            f'SELECT * FROM heedb_{site}_pairs',
            out / f'heedb_{site.lower()}_pairs_1_6h.parquet',
        )
        write_parquet(
            con,
            f"""
            SELECT r.* FROM heedb_{site}_records r
            SEMI JOIN (
                SELECT path_a AS waveform_path FROM heedb_{site}_pairs
                UNION SELECT path_b FROM heedb_{site}_pairs
            ) p USING (waveform_path)
            """,
            out / f'heedb_{site.lower()}_pair_records.parquet',
        )
        n = con.execute(
            f'SELECT count(*), count(DISTINCT patient_id) FROM heedb_{site}_pairs'
        ).fetchone()
        print(json.dumps({'site': site, 'pairs_1_6h': n[0], 'patients': n[1]}), flush=True)
        con.execute(f'DROP TABLE heedb_{site}_pairs')
        con.execute(f'DROP TABLE heedb_{site}_records')


def build_mimic(con: duckdb.DuckDBPyConnection, root: Path, out: Path) -> None:
    ecg = root / 'mimic-iv-ecg/1.0/record_list.csv'
    ed = root / 'mimic-iv-ed-2.2/ed/edstays.csv.gz'
    patients = root / 'mimiciv/3.1/hosp/patients.csv.gz'
    labs = root / 'mimiciv/3.1/hosp/labevents.csv.gz'
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_records AS
        SELECT 'MIMIC' AS dataset, e.subject_id AS patient_id,
               e.study_id AS record_id, e.ecg_time AS acq_ts,
               ? || '/' || e.path AS waveform_path,
               p.gender AS sex_label,
               p.anchor_age + year(e.ecg_time) - p.anchor_year AS age_years
        FROM read_csv(?) e
        LEFT JOIN read_csv(?) p USING (subject_id)
        """,
        [str(root / 'mimic-iv-ecg/1.0'), str(ecg), str(patients)],
    )
    write_parquet(con, 'SELECT * FROM mimic_records', out / 'mimic_records.parquet')
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_ed_matches AS
        SELECT d.stay_id, d.subject_id, d.intime, d.outtime,
               r.record_id, r.acq_ts, r.waveform_path, r.age_years,
               r.sex_label
        FROM mimic_records r
        JOIN read_csv(?) d
          ON r.patient_id=d.subject_id
         AND r.acq_ts BETWEEN d.intime AND d.outtime
        """,
        [str(ed)],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_candidate_stays AS
        WITH ambiguous AS (
            SELECT record_id FROM mimic_ed_matches
            GROUP BY record_id HAVING count(*) > 1
        ), clean AS (
            SELECT m.* FROM mimic_ed_matches m
            ANTI JOIN ambiguous a USING (record_id)
            WHERE m.age_years >= 18
        ), gaps AS (
            SELECT stay_id,
                   count(*) OVER (
                       PARTITION BY stay_id ORDER BY acq_ts
                       RANGE BETWEEN INTERVAL '6 hours' PRECEDING
                       AND INTERVAL '1 hour' PRECEDING
                   ) AS n_pairs
            FROM clean
        )
        SELECT DISTINCT c.stay_id, c.subject_id, c.intime, c.outtime
        FROM clean c
        JOIN (SELECT DISTINCT stay_id FROM gaps WHERE n_pairs > 0) g
          USING (stay_id)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_troponin_rows AS
        SELECT c.stay_id, l.charttime, l.itemid, l.value, l.valuenum,
               l.valueuom, l.ref_range_upper, l.flag, l.comments
        FROM read_csv(?) l
        JOIN mimic_candidate_stays c
          ON l.subject_id=c.subject_id
         AND l.charttime BETWEEN c.intime AND c.outtime
        WHERE l.itemid IN (51002, 51003, 52642)
        """,
        [str(labs)],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_troponin_classified AS
        WITH parsed AS (
            SELECT *,
                   coalesce(valuenum, try_cast(value AS DOUBLE)) AS result_num,
                   try_cast(regexp_extract(
                       comments,
                       '(?i)^\\s*(?:<\\s*|LESS\\s+THAN\\s+)([0-9]+(?:\\.[0-9]+)?)',
                       1
                   ) AS DOUBLE) AS upper_bound
            FROM mimic_troponin_rows
        )
        SELECT stay_id, charttime, itemid,
               CASE WHEN lower(coalesce(flag, ''))='abnormal'
                          OR result_num > ref_range_upper
                        THEN 'positive_or_abnormal'
                    WHEN ref_range_upper IS NOT NULL
                         AND (result_num <= ref_range_upper
                              OR upper_bound <= ref_range_upper)
                        THEN 'negative'
                    ELSE 'unresolved' END AS result_class
        FROM parsed
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_troponin_stays AS
        SELECT stay_id,
               count(DISTINCT charttime) FILTER (
                   WHERE result_class='negative') AS negative_times,
               count(*) FILTER (
                   WHERE result_class='positive_or_abnormal') AS positive_rows,
               count(*) FILTER (
                   WHERE result_class='unresolved') AS unresolved_rows
        FROM mimic_troponin_classified
        GROUP BY stay_id
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_external_stays AS
        SELECT c.* FROM mimic_candidate_stays c
        JOIN mimic_troponin_stays t USING (stay_id)
        WHERE t.negative_times >= 2
          AND t.positive_rows = 0 AND t.unresolved_rows = 0
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE mimic_external_ecgs AS
        WITH ambiguous AS (
            SELECT record_id FROM mimic_ed_matches
            GROUP BY record_id HAVING count(*) > 1
        )
        SELECT m.* FROM mimic_ed_matches m
        JOIN mimic_external_stays s USING (stay_id, subject_id)
        ANTI JOIN ambiguous a USING (record_id)
        """
    )
    write_parquet(
        con, 'SELECT * FROM mimic_external_stays', out / 'mimic_external_stays.parquet'
    )
    write_parquet(
        con, 'SELECT * FROM mimic_external_ecgs', out / 'mimic_external_ecgs.parquet'
    )
    print(json.dumps({
        'mimic_external_stays': con.execute(
            'SELECT count(*) FROM mimic_external_stays'
        ).fetchone()[0],
        'mimic_external_patients': con.execute(
            'SELECT count(DISTINCT subject_id) FROM mimic_external_stays'
        ).fetchone()[0],
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--heedb', type=Path, default=DEFAULT_HEEDB)
    parser.add_argument('--mimic', type=Path, default=DEFAULT_MIMIC)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--part', choices=('heedb', 'mimic', 'all'), default='all')
    args = parser.parse_args()
    con = db(args.out)
    if args.part in ('heedb', 'all'):
        build_heedb(con, args.heedb, args.out)
    if args.part in ('mimic', 'all'):
        build_mimic(con, args.mimic, args.out)


if __name__ == '__main__':
    main()
