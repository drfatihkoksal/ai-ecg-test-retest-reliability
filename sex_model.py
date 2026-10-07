"""Train and freeze AI ECG sex-score models on patient-disjoint HEEDB I0001.

The primary target is the HEEDB SexDSC field. Scores are model outputs, not
measurements of hormones, anatomy, or gender identity.
"""

import argparse
import json
import random
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from paths import WORK


ROOT = WORK
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
GROUPS = {'all': list(range(12)), 'limb': list(range(6)),
          'chest': list(range(6, 12))}


def label_value(value: str) -> int | None:
    if value is None:
        return None
    value = value.strip().upper()
    if value in ('MALE', 'M'):
        return 1
    if value in ('FEMALE', 'F'):
        return 0
    return None


def load_signal(path: str) -> np.ndarray:
    signal, header = wfdb.rdsamp(path)
    channels = list(header['sig_name'])
    x = signal[:, [channels.index(lead) for lead in LEADS]].astype(np.float32)
    if int(header['fs']) == 500:
        x = resample_poly(x, 1, 2, axis=0)
    if x.shape != (2500, 12):
        raise ValueError(f'Unexpected waveform shape: {x.shape}')
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite waveform: {path}')
    x -= np.median(x, axis=0, keepdims=True)
    return x.T


class ECGDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scale: np.ndarray,
                 group: str):
        self.paths = frame['waveform_path'].tolist()
        self.labels = frame['target'].to_numpy(dtype=np.float32)
        self.scale = scale[GROUPS[group]].astype(np.float32)
        self.indices = GROUPS[group]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        x = load_signal(self.paths[index])[self.indices]
        x = x / self.scale[:, None]
        return torch.from_numpy(x.copy()), torch.tensor(self.labels[index])


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride, 3)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, 1, 2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1, stride) \
            if in_channels != out_channels or stride != 1 else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class ECGSexNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(channels, 32, 2),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        return self.head(self.features(x).squeeze(-1)).squeeze(-1)


def prepare(root: Path, model_site: str) -> None:
    con = duckdb.connect()
    if model_site == 'mimic':
        records = root / 'mimic_records.parquet'
        external = root / 'mimic_external_stays.parquet'
        output = root / 'sex_mimic_representatives.parquet'
        query = f"""
            WITH eligible AS (
                SELECT 'MIMIC:' || CAST(r.patient_id AS VARCHAR) AS patient_id,
                       r.record_id, r.waveform_path, r.sex_label,
                       CASE WHEN upper(r.sex_label) IN ('M','MALE') THEN 1
                            WHEN upper(r.sex_label) IN ('F','FEMALE') THEN 0
                            ELSE NULL END AS target
                FROM read_parquet('{records}') r
                WHERE r.patient_id NOT IN (
                    SELECT DISTINCT subject_id FROM read_parquet('{external}')
                )
            ), one_per_patient AS (
                SELECT *, row_number() OVER (
                    PARTITION BY patient_id ORDER BY hash(record_id)
                ) AS rn
                FROM eligible WHERE target IS NOT NULL
            )
            SELECT patient_id, record_id, waveform_path, sex_label, target,
                   CASE WHEN hash(patient_id) % 100 < 85
                        THEN 'train' ELSE 'val' END AS split
            FROM one_per_patient WHERE rn=1
        """
        con.sql(query).write_parquet(str(output), compression='zstd')
        counts = con.execute(
            'SELECT split,count(*) FROM read_parquet(?) GROUP BY 1',
            [str(output)],
        ).fetchall()
        print(json.dumps({'model_site': model_site,
                          'representatives': counts}), flush=True)
        return
    records = root / 'heedb_i0001_pair_records.parquet'
    qc = root / 'beats_i0001' / 'qc_*.parquet'
    output = root / 'sex_i0001_representatives.parquet'
    query = f"""
        WITH eligible AS (
            SELECT r.patient_id, r.record_id, r.waveform_path,
                   r.sex_label, r.sex_birth_label, r.split,
                   CASE WHEN upper(r.sex_label)='MALE' THEN 1
                        WHEN upper(r.sex_label)='FEMALE' THEN 0
                        ELSE NULL END AS target
            FROM read_parquet('{records}') r
            JOIN read_parquet('{qc}') q USING (waveform_path)
            WHERE q.ok
        ), consistent AS (
            SELECT * FROM eligible WHERE target IS NOT NULL
            QUALIFY count(DISTINCT target) OVER (PARTITION BY patient_id)=1
        ), one_per_patient AS (
            SELECT *, row_number() OVER (
                PARTITION BY patient_id ORDER BY hash(record_id)
            ) AS rn
            FROM consistent
        )
        SELECT patient_id, record_id, waveform_path, sex_label,
               sex_birth_label, split, target
        FROM one_per_patient WHERE rn=1
    """
    con.sql(query).write_parquet(str(output), compression='zstd')
    counts = con.execute('SELECT split,count(*) FROM read_parquet(?) GROUP BY 1',
                         [str(output)]).fetchall()
    print(json.dumps({'representatives': counts}), flush=True)


def compute_scale(frame: pd.DataFrame, output: Path) -> np.ndarray:
    sample = frame.sample(n=min(10000, len(frame)), random_state=42)
    total = np.zeros(12, dtype=np.float64)
    count = 0
    for i, path in enumerate(sample['waveform_path']):
        x = load_signal(path)
        total += np.mean(x.astype(np.float64) ** 2, axis=1)
        count += 1
        if (i + 1) % 2000 == 0:
            print(json.dumps({'scale_records': i + 1}), flush=True)
    scale = np.maximum(np.sqrt(total / count), 0.01).astype(np.float32)
    np.save(output, scale)
    return scale


def predict_logits(model: nn.Module, loader: DataLoader,
                   device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device == 'cuda'):
                score = model(x)
            logits.append(score.float().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def train(root: Path, model_site: str, group: str, epochs: int, batch_size: int,
          num_workers: int, max_train_patients: int | None,
          max_val_patients: int | None) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    representative_path = (root / 'sex_mimic_representatives_qc.parquet'
                           if model_site == 'mimic' else
                           root / 'sex_i0001_representatives.parquet')
    frame = pd.read_parquet(representative_path)
    train_frame = frame[frame['split'] == 'train'].reset_index(drop=True)
    val_frame = frame[frame['split'] == 'val'].reset_index(drop=True)
    if max_train_patients is not None:
        train_frame = train_frame.sample(
            n=min(max_train_patients, len(train_frame)), random_state=42
        ).reset_index(drop=True)
    if max_val_patients is not None:
        val_frame = val_frame.sample(
            n=min(max_val_patients, len(val_frame)), random_state=43
        ).reset_index(drop=True)
    print(json.dumps({'group': group, 'train_patients': len(train_frame),
                      'val_patients': len(val_frame)}), flush=True)
    scale_path = root / f'sex_input_scale_{model_site}.npy'
    scale = np.load(scale_path) if scale_path.exists() else None
    if scale is None or not np.isfinite(scale).all():
        scale = compute_scale(train_frame, scale_path)
    train_loader = DataLoader(
        ECGDataset(train_frame, scale, group), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        ECGDataset(val_frame, scale, group), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ECGSexNet(len(GROUPS[group])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    stale = 0
    checkpoint = root / f'sex_model_{model_site}_{group}.pt'
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                                enabled=device == 'cuda'):
                logits = model(x)
                loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
        scores, labels = predict_logits(model, val_loader, device)
        auc = roc_auc_score(labels, scores)
        print(json.dumps({'group': group, 'epoch': epoch + 1,
                          'train_loss': total_loss / len(train_frame),
                          'val_auc': auc}), flush=True)
        if auc > best_auc + 1e-4:
            best_auc = auc
            stale = 0
            torch.save({'state_dict': model.state_dict(), 'group': group,
                        'val_auc': auc, 'epoch': epoch + 1}, checkpoint)
        else:
            stale += 1
            if stale >= 2:
                break


def score_manifest(root: Path, site: str) -> pd.DataFrame:
    if site == 'i0001':
        pairs = pd.read_parquet(root / 'heedb_i0001_valid_pairs.parquet')
        pairs = pairs[pairs['split'] == 'test']
    elif site == 'i0006':
        pairs = pd.read_parquet(root / 'heedb_i0006_valid_pairs.parquet')
    else:
        pairs = pd.read_parquet(root / 'mimic_pairs_clinical_flags.parquet')
        pairs = pairs[pairs['stable_primary']]
    first = pairs[['dataset', 'patient_id', 'record_a', 'path_a', 'sex_label']].copy()
    first.columns = ['dataset', 'patient_id', 'record_id', 'waveform_path', 'sex_label']
    second = pairs[['dataset', 'patient_id', 'record_b', 'path_b', 'sex_label']].copy()
    second.columns = first.columns
    result = pd.concat([first, second], ignore_index=True)
    result = result.drop_duplicates('waveform_path').reset_index(drop=True)
    result['target'] = result['sex_label'].map(label_value)
    if site == 'i0001':
        birth = pd.read_parquet(
            root / 'heedb_i0001_pair_records.parquet',
            columns=['waveform_path', 'sex_birth_label'],
        ).drop_duplicates('waveform_path')
        result = result.merge(birth, on='waveform_path', how='left',
                              validate='one_to_one')
        result['birth_target'] = result['sex_birth_label'].map(label_value)
    return result


def score(root: Path, site: str, model_site: str,
          batch_size: int, num_workers: int) -> None:
    frame = score_manifest(root, site)
    frame = frame[frame['target'].notna()].reset_index(drop=True)
    scale = np.load(root / f'sex_input_scale_{model_site}.npy')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for group in GROUPS:
        checkpoint = torch.load(root / f'sex_model_{model_site}_{group}.pt', map_location=device,
                                weights_only=False)
        model = ECGSexNet(len(GROUPS[group])).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        loader = DataLoader(
            ECGDataset(frame, scale, group), batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        logits, labels = predict_logits(model, loader, device)
        frame[f'{group}_logit'] = logits
        auc = roc_auc_score(labels, logits)
        print(json.dumps({'site': site, 'model_site': model_site,
                          'group': group, 'n': len(frame),
                          'auc': auc}), flush=True)
    suffix = '_local' if site == 'mimic' and model_site == 'mimic' else ''
    frame.to_parquet(root / f'sex_scores_{site}{suffix}.parquet', index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'train', 'score'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--group', choices=tuple(GROUPS), default='all')
    parser.add_argument('--site', choices=('i0001', 'i0006', 'mimic'), default='i0001')
    parser.add_argument('--model-site', choices=('i0001', 'mimic'), default='i0001')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-train-patients', type=int)
    parser.add_argument('--max-val-patients', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root, args.model_site)
    elif args.action == 'train':
        train(args.root, args.model_site, args.group, args.epochs, args.batch_size,
              args.workers, args.max_train_patients, args.max_val_patients)
    else:
        score(args.root, args.site, args.model_site,
              args.batch_size, args.workers)


if __name__ == '__main__':
    main()
