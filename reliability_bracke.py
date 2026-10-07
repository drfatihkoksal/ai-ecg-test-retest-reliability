"""Score the published Bracke et al. (MICCAI 2026) sex-conditioned BiMamba2 ECG-age model.

Weights: https://huggingface.co/BBracke/demographic-ssm-ecg-age (MIT).
Code:    https://github.com/B-Bracke/demographic-ssm-ecg-age (cloned to WORK/bracke).
Run inside WORK/venv_mamba. mamba-ssm is installed without its CUDA extensions,
so Mamba2 runs with use_mem_eff_path=False (the Triton-kernel path computing
the same function as the fused path).

Actions:
  validate   MAE and correlation on 5000 I0001 validation patients
  score      all ECGs in rel_records.parquet -> rel_scores_bracke/*.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from reliability_models import WORK, read_ecg, to_lima


BRACKE = WORK / 'bracke'
AGE_SCALE = 100.0


def load_model(device: str):
    import mamba_ssm
    original = mamba_ssm.Mamba2.__init__

    def patched(self, *args, **kwargs):
        kwargs['use_mem_eff_path'] = False
        original(self, *args, **kwargs)

    mamba_ssm.Mamba2.__init__ = patched
    sys.path.insert(0, str(BRACKE))
    import model as bracke_model
    bracke_model.Mamba2.__init__ = patched
    net = bracke_model.MultimodalDeepMambaECG(num_layers=4, hidden_dim=128, num_leads=12,
                                              num_sex_classes=2, downsample_factor=8)
    state = torch.load(BRACKE / 'model.pth', map_location='cpu', weights_only=False)['model']
    result = net.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return net.to(device).eval()


class BrackeDataset(Dataset):
    def __init__(self, paths: list[str], sex: np.ndarray):
        self.paths = paths
        self.sex = sex

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x, fs, _ = read_ecg(self.paths[i])
        if x is None or self.sex[i] < 0:
            return torch.zeros(12, 4096), 0, False
        return torch.from_numpy(to_lima(x, fs)), int(self.sex[i]), True


def predict(net, paths, sex, device, workers, batch_size=256) -> np.ndarray:
    loader = DataLoader(BrackeDataset(paths, sex), batch_size=batch_size,
                        num_workers=workers, pin_memory=True)
    out = []
    with torch.inference_mode():
        for x, s, ok in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
                pred, _, _ = net(x.to(device), s.to(device))
            p = pred.float().cpu().numpy().ravel() * AGE_SCALE
            p[~ok.numpy()] = np.nan
            out.append(p)
    return np.concatenate(out)


def sex_code(series: pd.Series) -> np.ndarray:
    s = series.fillna('').astype(str).str.strip().str.upper()
    return s.map({'MALE': 1, 'M': 1, 'FEMALE': 0, 'F': 0}).fillna(-1).astype(int).to_numpy()


def validate(device: str, workers: int, n: int) -> None:
    frame = pd.read_parquet(WORK / 'rel_val_records.parquet').head(n)
    net = load_model(device)
    pred = predict(net, frame['waveform_path'].tolist(), frame['sex_target'].to_numpy(), device, workers)
    ok = np.isfinite(pred)
    age = frame['age_years'].to_numpy()[ok]
    result = {'n': int(ok.sum()), 'mae': float(np.mean(np.abs(pred[ok] - age))),
              'r': float(np.corrcoef(pred[ok], age)[0, 1]),
              'published_code15_mae': 6.72, 'published_code15_r': 0.90}
    (WORK / 'rel_bracke_validation.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


def path_sex() -> pd.DataFrame:
    pairs = pd.read_parquet(WORK / 'rel_pairs.parquet')
    triples = pd.read_parquet(WORK / 'rel_triples.parquet')
    parts = [pairs[['path_a', 'sex_a']].set_axis(['waveform_path', 'sex'], axis=1),
             pairs[['path_b', 'sex_b']].set_axis(['waveform_path', 'sex'], axis=1)]
    for s in 'abc':
        parts.append(triples[[f'path_{s}', 'sex_a']].set_axis(['waveform_path', 'sex'], axis=1))
    frame = pd.concat(parts)
    frame['code'] = sex_code(frame['sex'])
    # One code per ECG; conflicting or unknown labels are left unscored (-1).
    agg = frame.groupby('waveform_path')['code'].agg(lambda c: c.iloc[0] if c.nunique() == 1 else -1)
    return agg.reset_index()


def score(device: str, workers: int, shard_size: int) -> None:
    frame = path_sex().sort_values('waveform_path').reset_index(drop=True)
    net = load_model(device)
    out_dir = WORK / 'rel_scores_bracke'
    out_dir.mkdir(exist_ok=True)
    for start in range(0, len(frame), shard_size):
        target = out_dir / f'bracke_{start // shard_size:04d}.parquet'
        if target.exists():
            continue
        chunk = frame.iloc[start:start + shard_size]
        pred = predict(net, chunk['waveform_path'].tolist(), chunk['code'].to_numpy(), device, workers)
        pd.DataFrame({'waveform_path': chunk['waveform_path'].to_numpy(), 'bracke_age': pred}) \
          .to_parquet(target, index=False)
        print(json.dumps({'done': start + len(chunk), 'total': len(frame),
                          'scored': int(np.isfinite(pred).sum())}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('validate', 'score'))
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--n', type=int, default=5000)
    parser.add_argument('--shard-size', type=int, default=50000)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.action == 'validate':
        validate(device, args.workers, args.n)
    else:
        score(device, args.workers, args.shard_size)


if __name__ == '__main__':
    main()
