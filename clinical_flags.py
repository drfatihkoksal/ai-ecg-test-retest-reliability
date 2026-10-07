"""Predefined structured clinical stability flags for MIMIC external pairs."""

import argparse
import json
from pathlib import Path

import duckdb


from paths import MIMIC_ROOT, WORK

DEFAULT_DATA = MIMIC_ROOT
DEFAULT_OUT = WORK


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    con = duckdb.connect()
    con.execute('SET threads=12')
    pairs = args.out / 'mimic_valid_pairs.parquet'
    machine = args.data_root / 'mimic-iv-ecg/1.0/machine_measurements.csv'
    diagnoses = args.data_root / 'mimic-iv-ed-2.2/ed/diagnosis.csv.gz'
    vitals = args.data_root / 'mimic-iv-ed-2.2/ed/vitalsign.csv.gz'
    labs = args.data_root / 'mimiciv/3.1/hosp/labevents.csv.gz'
    report_columns = ', '.join(f'report_{i}' for i in range(18))
    con.execute(
        f"""
        CREATE TEMP TABLE machine_flags AS
        WITH reports AS (
            SELECT study_id,
                   lower(concat_ws(' ', {report_columns})) AS report_text,
                   cart_id, bandwidth, filtering,
                   qrs_end - qrs_onset AS qrs_duration_ms
            FROM read_csv(?)
        )
        SELECT study_id, cart_id, bandwidth, filtering, qrs_duration_ms,
               regexp_matches(report_text, 'paced|pacemaker') AS paced,
               CASE WHEN regexp_matches(report_text, 'atrial fibrillation|atrial flutter')
                         THEN 'af_flutter'
                    WHEN regexp_matches(report_text, 'sinus rhythm|sinus bradycardia|sinus tachycardia')
                         THEN 'sinus'
                    ELSE 'unknown' END AS rhythm_class
        FROM reports
        """,
        [str(machine)],
    )
    con.execute(
        """
        CREATE TEMP TABLE acute_diagnoses AS
        SELECT stay_id,
               bool_or((icd_version=10 AND regexp_matches(icd_code, '^(I21|I22|I24|R570)'))
                    OR (icd_version=9 AND regexp_matches(icd_code, '^(410|411|78551)')))
                   AS acute_mi_acs_or_shock
        FROM read_csv(?) GROUP BY stay_id
        """,
        [str(diagnoses)],
    )
    con.execute(
        """
        CREATE TEMP TABLE pair_vitals AS
        SELECT p.stay_id, p.record_a, p.record_b,
               count(v.charttime) AS vital_rows,
               bool_or(try_cast(v.sbp AS DOUBLE) < 90
                    OR try_cast(v.o2sat AS DOUBLE) < 90
                    OR try_cast(v.heartrate AS DOUBLE) > 150)
                   AS unstable_vital
        FROM read_parquet(?) p
        LEFT JOIN read_csv(?) v
          ON p.stay_id=v.stay_id
         AND v.charttime BETWEEN p.time_a AND p.time_b
        GROUP BY p.stay_id,p.record_a,p.record_b
        """,
        [str(pairs), str(vitals)],
    )
    con.execute(
        """
        CREATE TEMP TABLE post_ed_troponin AS
        WITH parsed AS (
            SELECT s.stay_id,
                   coalesce(l.valuenum, try_cast(l.value AS DOUBLE)) AS result_num,
                   try_cast(regexp_extract(
                       l.comments,
                       '(?i)^\\s*(?:<\\s*|LESS\\s+THAN\\s+)([0-9]+(?:\\.[0-9]+)?)',
                       1
                   ) AS DOUBLE) AS upper_bound,
                   l.ref_range_upper, l.flag
            FROM read_csv(?) l
            JOIN read_parquet(?) s
              ON l.subject_id=s.subject_id
             AND l.charttime > s.outtime
             AND l.charttime <= s.outtime + INTERVAL '6 hours'
            WHERE l.itemid IN (51002,51003,52642)
        ), classified AS (
            SELECT stay_id,
                   CASE WHEN lower(coalesce(flag,''))='abnormal'
                                  OR result_num > ref_range_upper
                             THEN 'positive_or_abnormal'
                        WHEN ref_range_upper IS NOT NULL
                             AND (result_num <= ref_range_upper
                                  OR upper_bound <= ref_range_upper)
                             THEN 'negative'
                        ELSE 'unresolved' END AS result_class
            FROM parsed
        )
        SELECT stay_id, count(*) AS post_ed_troponin_rows,
               count(*) FILTER (WHERE result_class='positive_or_abnormal')
                   AS post_ed_positive_rows,
               count(*) FILTER (WHERE result_class='unresolved')
                   AS post_ed_unresolved_rows
        FROM classified GROUP BY stay_id
        """,
        [str(labs), str(args.out / 'mimic_external_stays.parquet')],
    )
    query = f"""
        SELECT p.*,
               coalesce(a.paced, false) OR coalesce(b.paced, false)
                   AS paced_either,
               a.rhythm_class AS rhythm_a,
               b.rhythm_class AS rhythm_b,
               a.rhythm_class <> 'unknown'
                   AND b.rhythm_class <> 'unknown'
                   AND a.rhythm_class <> b.rhythm_class
                   AS rhythm_change,
               coalesce(d.acute_mi_acs_or_shock, false) AS acute_diagnosis,
               coalesce(v.unstable_vital, false) AS unstable_vital,
               v.vital_rows,
               a.qrs_duration_ms AS qrs_a_ms,
               b.qrs_duration_ms AS qrs_b_ms,
               a.cart_id <> b.cart_id AS device_change,
               a.filtering <> b.filtering OR a.bandwidth <> b.bandwidth
                   AS filter_change,
               coalesce(post.post_ed_troponin_rows,0) AS post_ed_troponin_rows,
               coalesce(post.post_ed_positive_rows,0) AS post_ed_positive_rows,
               coalesce(post.post_ed_unresolved_rows,0) AS post_ed_unresolved_rows,
               NOT (coalesce(a.paced, false) OR coalesce(b.paced, false)
                    OR coalesce(a.rhythm_class <> 'unknown'
                       AND b.rhythm_class <> 'unknown'
                       AND a.rhythm_class <> b.rhythm_class, false)
                    OR coalesce(d.acute_mi_acs_or_shock, false)
                    OR coalesce(v.unstable_vital, false)) AS stable_primary
        FROM read_parquet('{pairs}') p
        LEFT JOIN machine_flags a ON p.record_a=a.study_id
        LEFT JOIN machine_flags b ON p.record_b=b.study_id
        LEFT JOIN acute_diagnoses d USING (stay_id)
        LEFT JOIN pair_vitals v USING (stay_id,record_a,record_b)
        LEFT JOIN post_ed_troponin post USING (stay_id)
    """
    output = args.out / 'mimic_pairs_clinical_flags.parquet'
    con.sql(query).write_parquet(str(output), compression='zstd')
    stats = con.execute(
        """
        SELECT count(*),count(*) FILTER (WHERE stable_primary),
               count(DISTINCT patient_id) FILTER (WHERE stable_primary),
               count(*) FILTER (WHERE paced_either),
               count(*) FILTER (WHERE rhythm_change),
               count(*) FILTER (WHERE acute_diagnosis),
               count(*) FILTER (WHERE unstable_vital),
               count(*) FILTER (WHERE vital_rows=0),
               count(*) FILTER (WHERE post_ed_positive_rows > 0),
               count(*) FILTER (WHERE post_ed_troponin_rows = 0)
        FROM read_parquet(?)
        """,
        [str(output)],
    ).fetchone()
    print(json.dumps({'all_pairs': stats[0], 'stable_pairs': stats[1],
                      'stable_patients': stats[2], 'paced': stats[3],
                      'rhythm_change': stats[4], 'acute_diagnosis': stats[5],
                      'unstable_vital': stats[6], 'no_interval_vitals': stats[7]}),
          flush=True)
    print(json.dumps({'post_ed_positive_pairs': stats[8],
                      'post_ed_no_lab_pairs': stats[9]}),
          flush=True)


if __name__ == '__main__':
    main()
