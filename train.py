import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import Config

from model import BCNE_Original

# ================= 🏆 配置 =================
TARGET_LATENT_DIM = 3
BATCH_SIZE = 128
SEQ_LEN = 30
LR = 1e-4
EPOCHS = 60
TEMPERATURE = 0.1
TR_VALUE = 0.46

MODEL_DIR = os.path.join(Config.OUTPUT_ROOT, "models", "contrastive")
os.makedirs(MODEL_DIR, exist_ok=True)
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "bcne_ses01_dim3.pth")

TRAIN_SUBS = Config.TRAIN_SUBJECTS


# ================= Dataset (修改为 Ses-02) =================
class DirectPairedDataset(Dataset):
    def __init__(self, sub_list, seq_len):
        self.seq_len = seq_len
        self.pairs = []
        self.data_cache = {}

        video_pool = {}
        print(f">>> [Dataset] Scanning {len(sub_list)} subjects (Ses-01)...")

        stats = {'npy_found': 0, 'valid_videos': 0}

        for sub in sub_list:
            s_dir = os.path.join(Config.IMG_ROOT, sub)
            all_files = sorted(glob.glob(os.path.join(s_dir, "**", "*.npy"), recursive=True))
            # 过滤 ses-02
            files = [f for f in all_files if 'ses-01' in os.path.basename(f)]
            stats['npy_found'] += len(files)

            for f_path in files:
                try:
                    fname = os.path.basename(f_path)
                    import re
                    run_match = re.search(r'run-(\d+)', fname)
                    if not run_match: continue
                    run_str = run_match.group(1)

                    # 找 TSV (修改为 ses-02)
                    tsv_patterns = [
                        os.path.join(Config.DATA_ROOT, sub, 'ses-01', 'func',
                                     f"*{sub}*ses-01*task-alignvideo*run-{run_str}*events.tsv")
                    ]
                    tsv_path = None
                    for p in tsv_patterns:
                        g = glob.glob(p)
                        if g: tsv_path = g[0]; break

                    if not tsv_path: continue

                    data_shape = np.load(f_path, mmap_mode='r').shape
                    total_tr = data_shape[0]

                    df = pd.read_csv(tsv_path, sep='\t')
                    vid_rows = df[df['trial_type'].str.contains('video|movie|film|clip', case=False, na=False)]

                    for _, row in vid_rows.iterrows():
                        if 'stim_file' in row and pd.notna(row['stim_file']):
                            vid_id = os.path.splitext(os.path.basename(row['stim_file']))[0]
                        else:
                            vid_id = f"V_{int(row['onset'])}"

                        s = int(row['onset'] / TR_VALUE)
                        e = int((row['onset'] + row['duration']) / TR_VALUE)

                        if e - s < seq_len: continue
                        if s >= total_tr: continue
                        real_e = min(e, total_tr)
                        if real_e - s < seq_len: continue

                        if vid_id not in video_pool: video_pool[vid_id] = {}
                        video_pool[vid_id][sub] = (f_path, s, real_e)
                        stats['valid_videos'] += 1

                except Exception as e:
                    continue

        print(f"  > Valid Segments: {stats['valid_videos']} | Unique Videos: {len(video_pool)}")

        # 生成配对
        for vid_id, sub_dict in video_pool.items():
            subs = list(sub_dict.keys())
            if len(subs) < 2: continue

            n_augment = len(subs) * 20
            for _ in range(n_augment):
                s1, s2 = np.random.choice(subs, 2, replace=False)
                info1 = sub_dict[s1];
                info2 = sub_dict[s2]
                common = min(info1[2] - info1[1], info2[2] - info2[1])
                if common < seq_len: continue

                offset = np.random.randint(0, common - seq_len + 1)
                self.pairs.append({
                    'path_a': info1[0], 'start_a': info1[1] + offset,
                    'path_b': info2[0], 'start_b': info2[1] + offset,
                    'len': seq_len
                })

        print(f"  > Pairs Generated: {len(self.pairs)}")
        if len(self.pairs) == 0: raise ValueError("No pairs generated!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        return (self._load(item['path_a'], item['start_a'], item['len']),
                self._load(item['path_b'], item['start_b'], item['len']))

    def _load(self, path, start, length):
        if path not in self.data_cache: self.data_cache[path] = np.load(path, mmap_mode='r')
        d = np.array(self.data_cache[path][start:start + length], dtype=np.float32)
        if d.ndim == 2:
            d = d.reshape(-1, 1, 20, 20)
        elif d.ndim == 3:
            d = d[:, np.newaxis, :, :]
        flat = d.reshape(d.shape[0], -1)
        return (d - flat.mean()) / (flat.std() + 1e-6)


# ================= Loss =================
def info_nce_loss(features_a, features_b, temperature=0.1):
    features_a = F.normalize(features_a, dim=1)
    features_b = F.normalize(features_b, dim=1)
    logits = torch.matmul(features_a, features_b.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ================= 主程序 =================
def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training BCNE Original on Ses-02")

    dataset = DirectPairedDataset(TRAIN_SUBS, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    model = BCNE_Original(latent_dim=TARGET_LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0

        for ba, bb in loader:
            ba, bb = ba.to(device), bb.to(device)
            # Flatten
            b, s, c, h, w = ba.shape
            flat_a = ba.view(-1, c, h, w)
            flat_b = bb.view(-1, c, h, w)

            optimizer.zero_grad()
            emb_a, _ = model(flat_a)
            emb_b, _ = model(flat_b)

            loss = info_nce_loss(emb_a, emb_b, temperature=TEMPERATURE)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"  Ep {ep + 1}/{EPOCHS}: Loss={total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"✅ Saved to {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()