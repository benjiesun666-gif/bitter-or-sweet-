# data/npz_loader.py
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

class KataGoOfflineDataset(Dataset):
    def __init__(self, npz_paths: List[str]):
        self.states_board = []
        self.states_global = []
        self.expert_actions = []
        self.game_results = []

        print(f"Loading {len(npz_paths)} games into memory...")
        for path in npz_paths:
            try:
                data = np.load(path)
                packed_nchw = data['binaryInputNCHWPacked']
                N, C, _ = packed_nchw.shape
                unpacked = np.unpackbits(packed_nchw.view(np.uint8), axis=-1)
                unpacked_board = unpacked[:, :, :361].reshape(N, C, 19, 19)
                self.states_board.append(torch.tensor(unpacked_board, dtype=torch.float32))

                global_nc = data['globalInputNC']
                self.states_global.append(torch.tensor(global_nc, dtype=torch.float32))

                policy_targets = data['policyTargetsNCMove']
                if len(policy_targets.shape) == 3:
                    policy_dist = policy_targets[:, 0, :]
                else:
                    policy_dist = policy_targets

                expert_a = np.argmax(policy_dist, axis=-1)
                self.expert_actions.append(torch.tensor(expert_a, dtype=torch.long))

                results = data['globalTargetsNC'][:, 0]
                self.game_results.append(torch.tensor(results, dtype=torch.float32))

            except Exception as e:
                print(f"[Warning] Failed to load {path}: {e}")
                continue

        if len(self.states_board) > 0:
            self.states_board = torch.cat(self.states_board, dim=0)
            self.states_global = torch.cat(self.states_global, dim=0)
            self.expert_actions = torch.cat(self.expert_actions, dim=0)
            self.game_results = torch.cat(self.game_results, dim=0)
            print(f"Dataset compiled. Total transitions: {len(self.states_board)}")

    def __len__(self):
        return len(self.states_board)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.states_board[idx], self.states_global[idx],
                self.expert_actions[idx], self.game_results[idx])

def get_dataloaders(data_dir: str, config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    npz_paths = glob.glob(os.path.join(data_dir, "*.npz"))
    total_files = len(npz_paths)
    if total_files == 0:
        raise ValueError(f"No .npz files found in {data_dir}.")

    random.seed(42)
    random.shuffle(npz_paths)

    train_count = min(int(config['data'].get('train_split', 400)), total_files)
    val_count = min(int(config['data'].get('hpo_val_split', 100)), total_files - train_count)

    train_files = npz_paths[:train_count]
    val_files = npz_paths[train_count:train_count + val_count]
    test_files = npz_paths[train_count + val_count:]

    print(f"=== File-Level Split ===")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    train_dataset = KataGoOfflineDataset(train_files)
    val_dataset = KataGoOfflineDataset(val_files)
    test_dataset = KataGoOfflineDataset(test_files) if len(test_files) > 0 else None

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0) if test_dataset else None

    return train_loader, val_loader, test_loader
