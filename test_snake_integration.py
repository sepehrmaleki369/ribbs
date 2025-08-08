import torch
import yaml
from torch.utils.data import DataLoader
from core.regression_dataset import RegressionDataset
from losses_n import SnakeFastLoss
from utils.graphs import load_training_graph_by_id


def run_test():
    with open('configs/dataset/drive_regression.yaml', 'r') as f:
        config = yaml.safe_load(f)
    ds = RegressionDataset(config, split='train')
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(dl))
    preds = torch.randn((len(batch['distance_map']), 1, batch['distance_map'].shape[-2], batch['distance_map'].shape[-1]))

    lbl_graphs = []
    for sid in batch['sample_id']:
        lbl_graphs.append(load_training_graph_by_id(str(sid)))

    snake = SnakeFastLoss(stepsz=0.5, alpha=0.1, beta=0.1, fltrstdev=2.0, ndims=2, nsteps=3,
                          cropsz=[128, 128], dmax=155.0, maxedgelen=10.0, extgradfac=1.0)
    loss = snake(preds, lbl_graphs)
    print(f"SnakeFastLoss ran. Loss={loss.item():.4f}")


if __name__ == '__main__':
    run_test() 