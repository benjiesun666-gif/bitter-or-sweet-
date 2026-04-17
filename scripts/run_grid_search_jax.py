# scripts/run_grid_search_jax.py
import yaml
from data.npz_loader import get_dataloaders
from scripts.train_single_jax import train_run_jax

def main():
    with open('configs/grid_search_params.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_dir = "data/kata_games"
    train_loader, val_loader, _ = get_dataloaders(data_dir, config)

    lrs = config['grid_search']['learning_rates']
    wds = config['grid_search']['weight_decays']

    total_runs = len(lrs) * len(wds)
    current_run = 0

    print(f"🚀 JAX Grid Search Initiated: {total_runs} total configurations.")

    for lr in lrs:
        for wd in wds:
            run_id = f"LR_{lr}_WD_{wd}"
            current_run += 1
            print(f"\n" + "="*55)
            print(f"Executing Grid Point {current_run}/{total_runs}: {run_id}")
            print("="*55)

            train_run_jax(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=lr,
                weight_decay=wd,
                run_id=run_id
            )

if __name__ == "__main__":
    main()
