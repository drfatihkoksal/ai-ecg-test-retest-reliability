# AI-ECG test–retest reliability

Code and aggregate results for the study **"Short-interval test–retest reliability of artificial intelligence–derived ECG age and sex: most long-term within-person variation is already present within hours"** (Fatih Köksal; submitted manuscript ).

Chronological age and recorded sex cannot change between ECGs recorded hours apart, so any difference in an AI model's output between two such ECGs is variability of the measurement–model system. The study analyses 532 891 consecutive ECG pairs from 253 995 patients in two Harvard-Emory ECG Database institutions and MIMIC-IV-ECG, at intervals from under 5 minutes to over 1 year. The models are:

- two published ECG-age networks (Lima *et al.*, 2021; Bracke *et al.*, 2026);
- a published 12-lead abnormality classifier (Ribeiro *et al.*, 2020);
- in-house five-seed age and sex ensembles.

**Main finding.** Over 1–6 hours, the repeatability coefficient of the ECG-age gap was 15–19 years for both published models. The within-hours variance was 67–99% of the within-person variance across ECGs more than one year apart.

This repository contains **no patient-level data**. All result files are aggregate statistics.

## Contents

| Path | What it is |
| --- | --- |
| `reliability_protocol.md` | Analysis protocol written before model outputs were computed, with a dated change log of every deviation and post hoc analysis (in Turkish) |
| `paths.py` | Data locations (environment variables) |
| `ecg_io.py` | WFDB reading, quality checks, resampling |
| `build_manifests.py`, `extract_beats.py`, `prepare_pairs.py`, `clinical_flags.py` | Upstream cohort construction: HEEDB and MIMIC record tables, signal QC, MIMIC troponin-negative emergency subcohort and clinical flags |
| `sex_model.py` | Network architecture shared by the in-house models; trains the limb-only and precordial-only sex models |
| `reliability_manifest.py` | Interval-stratified consecutive pairs and three-ECG sets |
| `reliability_models.py` | In-house age/sex training (five seeds), Lima model scoring, signal descriptors, duplicate fingerprints |
| `reliability_bracke.py` | Bracke *et al.* ECG-age model scoring |
| `reliability_ribeiro.py` | Ribeiro *et al.* abnormality classifier scoring |
| `reliability_analysis.py` | Primary and secondary agreement analyses |
| `reliability_diagnostic_analysis.py` | Abnormality classifier versus machine statements |
| `reliability_artifact_check.py`, `reliability_reversal_check.py`, `reliability_heedb_12sl_check.py` | Artifact, electrode-reversal, and 12SL sensitivity analyses (post hoc) |
| `make_reliability_figure.py`, `make_reliability_figure2.py`, `make_graphical_abstract.py`, `make_supplement.py` | Figures, graphical abstract, supplementary material |
| `reliability_*_2026-10-0*.json` | Aggregate results |
| `reliability_supplement.{md,docx,pdf}`, `figures/` | Supplementary material and figures |
| `requirements/` | Package versions for the three Python environments |

## Data access

The source data are credentialed and are **not** distributed here. Each user needs their own access:

- **Harvard-Emory ECG Database (HEEDB) v5**, Brain Data Science Platform. Waveforms, metadata, and 12SL v24 statements for I0001 (Massachusetts General Hospital) and I0006 (Emory University Hospital) are used.
- **PhysioNet:**
  - [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/)
  - [MIMIC-IV-ED v2.2](https://physionet.org/content/mimic-iv-ed/2.2/)
  - [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)

Set the locations before running anything:

```bash
export HEEDB_ROOT=/path/to/heedb      # contains ECG/I0001 and ECG/I0006
export MIMIC_ROOT=/path/to/physionet  # contains mimic-iv-ecg/1.0, mimic-iv-ed-2.2, mimiciv/3.1
export EKG_WORK=/path/to/restricted/work   # patient-level derived files are written here
```

Alternatively, define `WORK`, `HEEDB_ROOT` and `MIMIC_ROOT` in an untracked `paths_local.py`.

## Published model weights

The weights are downloaded into `$EKG_WORK`; they are not redistributed here.

```bash
# Lima et al. 2021 (Zenodo 10.5281/zenodo.4892365, CC-BY-4.0; code MIT)
mkdir -p $EKG_WORK/lima_age && cd $EKG_WORK/lima_age
curl -L -o model.zip https://zenodo.org/api/records/4892365/files/model.zip/content && unzip model.zip
curl -LO https://raw.githubusercontent.com/antonior92/ecg-age-prediction/main/resnet.py

# Bracke et al. 2026 (MIT)
git clone https://github.com/B-Bracke/demographic-ssm-ecg-age $EKG_WORK/bracke
curl -L -o $EKG_WORK/bracke/model.pth \
  https://huggingface.co/BBracke/demographic-ssm-ecg-age/resolve/main/model.pth

# Ribeiro et al. 2020 (Zenodo 10.5281/zenodo.3765717, CC-BY-4.0)
mkdir -p $EKG_WORK/ribeiro && cd $EKG_WORK/ribeiro
curl -L -o model.zip https://zenodo.org/api/records/3765717/files/model.zip/content && unzip model.zip
```

## Environments

| Environment | Used for | Requirements |
| --- | --- | --- |
| main | everything except the two models below | `requirements/main.txt` |
| bracke | `reliability_bracke.py` | `requirements/bracke.txt` (venv with `--system-site-packages`) |
| ribeiro | `reliability_ribeiro.py` | `requirements/ribeiro.txt` |

Python 3.13 was used. GPU runs used an RTX 5090 with CUDA 12.8. The mamba-ssm CUDA extension could not be compiled for this hardware, so the Bracke model runs through the library's equivalent Triton-kernel path (see `requirements/bracke.txt`).

## Run order

```bash
# Upstream cohort construction
python build_manifests.py --part all
python extract_beats.py --manifest $EKG_WORK/heedb_i0001_pair_records.parquet --out $EKG_WORK/beats_i0001
python extract_beats.py --manifest $EKG_WORK/heedb_i0006_pair_records.parquet --out $EKG_WORK/beats_i0006
python extract_beats.py --manifest $EKG_WORK/mimic_external_ecgs.parquet --out $EKG_WORK/beats_mimic
python prepare_pairs.py --part all
python clinical_flags.py
python sex_model.py prepare --model-site i0001
for g in limb chest; do
  python sex_model.py train --model-site i0001 --group $g --epochs 6 \
    --max-train-patients 100000 --max-val-patients 10000
done

# Reliability study
python reliability_manifest.py
python reliability_models.py cache --workers 24
for s in 1 2 3 4 5; do
  python reliability_models.py train --task age --seed $s
  python reliability_models.py train --task sex --seed $s
done
python reliability_models.py lima-select --n 5000
python reliability_models.py score --workers 28 --batch-size 256
$BRACKE_PY reliability_bracke.py validate && $BRACKE_PY reliability_bracke.py score
$RIBEIRO_PY reliability_ribeiro.py validate && $RIBEIRO_PY reliability_ribeiro.py score
python reliability_analysis.py
python reliability_diagnostic_analysis.py
python reliability_reversal_check.py run && python reliability_reversal_check.py summarize
python reliability_artifact_check.py
python reliability_heedb_12sl_check.py
python make_reliability_figure.py
python make_reliability_figure2.py
python make_graphical_abstract.py
python make_supplement.py   # needs pandoc and LibreOffice
```

`$BRACKE_PY` and `$RIBEIRO_PY` are the Python interpreters of the two separate environments.



## License

Code: MIT (see `LICENSE`). The published models and source datasets are subject to their own licenses and data-use agreements.
# ai-ecg-test-retest-reliability
# ai-ecg-test-retest-reliability
