"""Train, select, and score AI-ECG models for the reliability study.

Actions:
  cache        signals for I0001 train/val fitting records (250 Hz, float16)
  train        HEEDB age or sex model for one seed (patient-disjoint I0001 train)
  lima-select  amplitude multiplier for the published Lima et al. age model,
               chosen on I0001 validation patients only
  score        signal features, duplicate fingerprints, and all model outputs
               for every ECG in rel_records.parquet
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from wfdb.processing import xqrs_detect

from paths import WORK
from ecg_io import LEADS, N_250, N_LIMA, read_ecg, to_250, to_lima
from sex_model import ECGSexNet


LIMA_DIR = WORK / 'lima_age'
SEEDS = (1, 2, 3, 4, 5)
FINGERPRINT_BLOCK = 10


def signal_features(x250: np.ndarray) -> dict:
    """x250: 12 x 2500 mV."""
    spec = np.abs(np.fft.rfft(x250, axis=1)) ** 2
    freqs = np.fft.rfftfreq(x250.shape[1], 1 / 250)
    total = spec[:, freqs > 0].sum(axis=1) + 1e-12
    hf = spec[:, freqs > 40].sum(axis=1) / total
    bw = spec[:, (freqs > 0) & (freqs < 0.5)].sum(axis=1) / total
    rms = np.sqrt(np.mean(x250 ** 2, axis=1))
    out = {'hf_noise': float(np.median(hf)), 'baseline_wander': float(np.median(bw))}
    out.update({f'rms_{lead}': float(v) for lead, v in zip(LEADS, rms)})
    hr, rr_cv = None, None
    try:
        peaks = xqrs_detect(sig=x250[1], fs=250, verbose=False)
        if len(peaks) >= 3:
            rr = np.diff(peaks) / 250
            hr = float(60 / np.median(rr))
            rr_cv = float(np.std(rr) / np.mean(rr))
    except Exception:
        pass
    out['heart_rate'] = hr
    out['rr_cv'] = rr_cv
    return out


def fingerprint(x250: np.ndarray) -> tuple[str, np.ndarray]:
    quantized = np.round(x250 / 0.005).astype(np.int16)
    digest = hashlib.md5(quantized.tobytes()).hexdigest()
    blocks = x250.reshape(12, -1, FINGERPRINT_BLOCK).mean(axis=2)
    return digest, blocks.astype(np.float16)


# ---------------------------------------------------------------- cache/train

def _cache_one(path: str) -> tuple[np.ndarray | None, str]:
    x, fs, reason = read_ecg(path)
    if x is None:
        return None, reason
    return to_250(x, fs).astype(np.float16), ''


def cache(work: Path, workers: int) -> None:
    for split in ('train', 'val'):
        frame = pd.read_parquet(work / f'rel_{split}_records.parquet')
        out = np.lib.format.open_memmap(
            work / f'rel_{split}_signals.npy', mode='w+', dtype=np.float16,
            shape=(len(frame), 12, N_250))
        ok = np.zeros(len(frame), dtype=bool)
        with ProcessPoolExecutor(workers) as pool:
            for i, (sig, _) in enumerate(pool.map(
                    _cache_one, frame['waveform_path'], chunksize=64)):
                if sig is not None:
                    out[i] = sig
                    ok[i] = True
        out.flush()
        frame['ok'] = ok
        frame.to_parquet(work / f'rel_{split}_records_ok.parquet', index=False)
        print(json.dumps({'split': split, 'n': len(frame), 'ok': int(ok.sum())}),
              flush=True)


class CachedDataset(Dataset):
    def __init__(self, signals: np.ndarray, index: np.ndarray,
                 target: np.ndarray, scale: np.ndarray):
        self.signals = signals
        self.index = index
        self.target = target.astype(np.float32)
        self.scale = scale.astype(np.float32)[:, None]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        x = self.signals[self.index[i]].astype(np.float32) / self.scale
        return torch.from_numpy(x), torch.tensor(self.target[i])


def target_values(frame: pd.DataFrame, task: str) -> np.ndarray:
    return (frame['age_years'] if task == 'age' else frame['sex_target']).to_numpy(np.float64)


def evaluate(model, loader, device, task, age_mean, age_sd) -> tuple[float, np.ndarray]:
    model.eval()
    preds, ys = [], []
    with torch.inference_mode():
        for x, y in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                p = model(x.to(device, non_blocking=True)).float()
            preds.append(p.cpu().numpy())
            ys.append(y.numpy())
    p, y = np.concatenate(preds), np.concatenate(ys)
    if task == 'age':
        p = p * age_sd + age_mean
        return float(np.mean(np.abs(p - y))), p
    return float(roc_auc_score(y, p)), p


def train(work: Path, task: str, seed: int, epochs: int, batch_size: int,
          workers: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scale = np.load(work / 'sex_input_scale_i0001.npy')
    data = {}
    for split in ('train', 'val'):
        frame = pd.read_parquet(work / f'rel_{split}_records_ok.parquet')
        signals = np.load(work / f'rel_{split}_signals.npy', mmap_mode='r')
        index = np.flatnonzero(frame['ok'].to_numpy())
        data[split] = (frame.iloc[index].reset_index(drop=True), signals, index)
    y_train = target_values(data['train'][0], task)
    age_mean, age_sd = float(y_train.mean()), float(y_train.std())
    if task == 'age':
        y_train = (y_train - age_mean) / age_sd
    y_val = target_values(data['val'][0], task)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader = DataLoader(
        CachedDataset(data['train'][1], data['train'][2], y_train, scale),
        batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(
        CachedDataset(data['val'][1], data['val'][2], y_val, scale),
        batch_size=1024, shuffle=False, num_workers=workers, pin_memory=True)
    model = ECGSexNet(12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=epochs * len(train_loader))
    loss_fn = nn.L1Loss() if task == 'age' else nn.BCEWithLogitsLoss()
    best = None
    checkpoint = work / f'rel_model_{task}_seed{seed}.pt'
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                loss = loss_fn(model(x).float(), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(y)
        metric, _ = evaluate(model, val_loader, device, task, age_mean, age_sd)
        improved = best is None or (metric < best if task == 'age' else metric > best)
        print(json.dumps({'task': task, 'seed': seed, 'epoch': epoch + 1,
                          'train_loss': total / (len(train_loader) * batch_size),
                          'val_mae' if task == 'age' else 'val_auc': metric}),
              flush=True)
        if improved:
            best = metric
            torch.save({'state_dict': model.state_dict(), 'task': task,
                        'seed': seed, 'epoch': epoch + 1, 'val_metric': metric,
                        'age_mean': age_mean, 'age_sd': age_sd}, checkpoint)


# ------------------------------------------------------------------- scoring

def load_lima(device: str) -> nn.Module:
    sys.path.insert(0, str(LIMA_DIR))
    from resnet import ResNet1d
    config = json.loads((LIMA_DIR / 'model' / 'config.json').read_text())
    model = ResNet1d(input_dim=(12, config['seq_length']),
                     blocks_dim=list(zip(config['net_filter_size'],
                                         config['net_seq_lengh'])),
                     n_classes=1, kernel_size=config['kernel_size'],
                     dropout_rate=config['dropout_rate'])
    ckpt = torch.load(LIMA_DIR / 'model' / 'model.pth', map_location='cpu',
                      weights_only=False)
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()


class ScoreDataset(Dataset):
    def __init__(self, paths: list[str], scale: np.ndarray, features: bool):
        self.paths = paths
        self.scale = scale.astype(np.float32)[:, None]
        self.features = features

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x, fs, reason = read_ecg(self.paths[i])
        info = {'reason': reason}
        if x is None:
            return (torch.zeros(12, N_250), torch.zeros(12, N_LIMA),
                    False, json.dumps(info), np.zeros((12, N_250 // FINGERPRINT_BLOCK),
                                                      dtype=np.float16))
        x250 = to_250(x, fs)
        info['fs'] = fs
        fp = np.zeros((12, N_250 // FINGERPRINT_BLOCK), dtype=np.float16)
        if self.features:
            info.update(signal_features(x250))
            info['md5'], fp = fingerprint(x250)
        return (torch.from_numpy(x250 / self.scale), torch.from_numpy(to_lima(x, fs)),
                True, json.dumps(info), fp)


def heedb_models(work: Path, device: str) -> dict:
    models = {}
    for task in ('age', 'sex'):
        for seed in SEEDS:
            path = work / f'rel_model_{task}_seed{seed}.pt'
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model = ECGSexNet(12)
            model.load_state_dict(ckpt['state_dict'])
            models[f'{task}_seed{seed}'] = (model.to(device).eval(), ckpt)
    return models


def run_models(x250, xlima, models, lima, lima_multiplier, device, extra) -> dict:
    out = {}
    with torch.inference_mode():
        x250 = x250.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            for name, (model, ckpt) in models.items():
                p = model(x250).float().cpu().numpy()
                if name.startswith('age'):
                    p = p * ckpt['age_sd'] + ckpt['age_mean']
                out[name] = p
            for name, (model, idx) in extra.items():
                out[name] = model(x250[:, idx]).float().cpu().numpy()
        if lima is not None:
            for mult in np.atleast_1d(lima_multiplier):
                p = lima(xlima.to(device, non_blocking=True) * float(mult))
                key = 'lima_age' if np.ndim(lima_multiplier) == 0 else f'lima_age_x{mult:g}'
                out[key] = p.float().cpu().numpy().ravel()
    return out


def lima_select(work: Path, n: int, workers: int) -> None:
    frame = pd.read_parquet(work / 'rel_val_records.parquet').head(n)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lima = load_lima(device)
    scale = np.load(work / 'sex_input_scale_i0001.npy')
    loader = DataLoader(ScoreDataset(frame['waveform_path'].tolist(), scale, False),
                        batch_size=128, num_workers=workers)
    preds, oks = {1.0: [], 10.0: []}, []
    for x250, xlima, ok, _, _ in loader:
        out = run_models(x250, xlima, {}, lima, np.array([1.0, 10.0]), device, {})
        for m in preds:
            preds[m].append(out[f'lima_age_x{m:g}'])
        oks.append(ok.numpy())
    ok = np.concatenate(oks)
    age = frame['age_years'].to_numpy()[ok]
    result = {f'x{m:g}': {'mae': float(np.mean(np.abs(np.concatenate(p)[ok] - age))),
                          'r': float(np.corrcoef(np.concatenate(p)[ok], age)[0, 1])}
              for m, p in preds.items()}
    chosen = min(result, key=lambda k: result[k]['mae'])
    result.update({'n_val_patients': int(ok.sum()), 'chosen': chosen,
                   'multiplier': float(chosen[1:])})
    (work / 'rel_lima_selection.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def score(work: Path, batch_size: int, workers: int, shard_size: int) -> None:
    records = pd.read_parquet(work / 'rel_records.parquet')
    paths = sorted(records['waveform_path'].unique())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = heedb_models(work, device)
    extra = {}
    for group, idx in (('limb', list(range(6))), ('chest', list(range(6, 12)))):
        ckpt = torch.load(work / f'sex_model_i0001_{group}.pt', map_location='cpu',
                          weights_only=False)
        model = ECGSexNet(6)
        model.load_state_dict(ckpt['state_dict'])
        extra[f'sex_{group}_only'] = (model.to(device).eval(), idx)
    lima = load_lima(device)
    multiplier = json.loads((work / 'rel_lima_selection.json').read_text())['multiplier']
    scale = np.load(work / 'sex_input_scale_i0001.npy')
    out_dir = work / 'rel_scores'
    out_dir.mkdir(exist_ok=True)
    for start in range(0, len(paths), shard_size):
        shard = start // shard_size
        target = out_dir / f'scores_{shard:04d}.parquet'
        if target.exists():
            continue
        chunk = paths[start:start + shard_size]
        loader = DataLoader(ScoreDataset(chunk, scale, True), batch_size=batch_size,
                            num_workers=workers, pin_memory=True)
        rows, fps = [], []
        for x250, xlima, ok, info, fp in loader:
            out = run_models(x250, xlima, models, lima, multiplier, device, extra)
            for i in range(len(info)):
                row = json.loads(info[i])
                row['ok'] = bool(ok[i])
                for name, values in out.items():
                    row[name] = float(values[i]) if row['ok'] else None
                rows.append(row)
            fps.append(fp.numpy())
        frame = pd.DataFrame(rows)
        frame.insert(0, 'waveform_path', chunk)
        np.save(out_dir / f'fingerprint_{shard:04d}.npy', np.concatenate(fps))
        tmp = target.with_suffix('.tmp')
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp,
                       compression='zstd')
        tmp.rename(target)
        print(json.dumps({'shard': shard, 'n': len(chunk), 'ok': int(frame['ok'].sum()),
                          'done': start + len(chunk), 'total': len(paths)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('cache', 'train', 'lima-select', 'score'))
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--task', choices=('age', 'sex'), default='age')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--shard-size', type=int, default=50000)
    args = parser.parse_args()
    if args.action == 'cache':
        cache(args.work, args.workers)
    elif args.action == 'train':
        train(args.work, args.task, args.seed, args.epochs, args.batch_size, args.workers)
    elif args.action == 'lima-select':
        lima_select(args.work, args.n, args.workers)
    else:
        score(args.work, args.batch_size, args.workers, args.shard_size)


if __name__ == '__main__':
    main()
