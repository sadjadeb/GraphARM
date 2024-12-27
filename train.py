from torch_geometric.datasets import ZINC, Planetoid
from torch_geometric import seed_everything
from tqdm import trange
import torch
import wandb

from models import DiffusionOrderingNetwork, DenoisingNetwork
from utils import NodeMasking
from grapharm import GraphARM

# improve reproducibility
seed = 42
seed_everything(seed)
torch.autograd.set_detect_anomaly(True)

device = 'cuda:3'
epoch_num = 2000
batch_size = 5
print(f"Using device {device}")

# instanciate the dataset
dataset = ZINC(root='./data/ZINC', transform=None, pre_transform=None, subset=True)
# dataset = Planetoid(root='./data/Cora', name='Cora', transform=None, pre_transform=None)
print(f"Dataset: {dataset}")

diff_ord_net = DiffusionOrderingNetwork(node_feature_dim=1,
                                        num_node_types=dataset.x.unique().shape[0],
                                        num_edge_types=dataset.edge_attr.unique().shape[0],
                                        num_layers=3,
                                        out_channels=1,
                                        device=device)

masker = NodeMasking(dataset)

denoising_net = DenoisingNetwork(
    node_feature_dim=dataset.num_features,
    edge_feature_dim=dataset.num_edge_features,
    num_node_types=dataset.x.unique().shape[0],
    num_edge_types=dataset.edge_attr.unique().shape[0],
    num_layers=7,
    # hidden_dim=32,
    device=device
)

wandb.init(
        project="GraphARM",
        name=f"ZINC_GraphARM",
        config={
            "policy": "train",
            "n_epochs": epoch_num,
            "batch_size": batch_size,
        },
        # mode='disabled'
    )

grapharm = GraphARM(
    dataset=dataset,
    denoising_network=denoising_net,
    diffusion_ordering_network=diff_ord_net,
    device=device
)

try:
    grapharm.load_model()
    print("Loaded model")
except:
    print ("No model to load")

# train loop
for epoch in trange(epoch_num):
    grapharm.train_step(
        train_batch=dataset[2*epoch*batch_size:(2*epoch + 1)*batch_size],
        val_batch=dataset[(2*epoch + 1)*batch_size:batch_size*(2*epoch + 2)],
        M=4,
        epoch=epoch
    )
grapharm.save_model()