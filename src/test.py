import torch
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, random, pickle
from torch_geometric.loader import DataLoader
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve

from PPI import *
from model import *
from utils import *


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def Metric(preds, labels):
    AUC = roc_auc_score(labels, preds)
    precisions, recalls, _ = precision_recall_curve(labels, preds)
    AUPR = auc(recalls, precisions)
    return AUC, AUPR


def Write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text) 
    logFile.write(text) 
    logFile.write('\n')


def process_dataset(csv_file):
    df = pd.read_csv(csv_file)
    result_dict = {}
    for row in df.values:
        key = f"{row[0]}_{row[1]}"
        result_dict[key] = row.tolist()
    return result_dict


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./examples')
parser.add_argument("--feature_path", type=str, default='./examples/Feature')
parser.add_argument("--output_path", type=str, default='./outputs')
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--r", type=int, default=10)
parser.add_argument("--layer", type=int, default=4) 
parser.add_argument("--batch_size", type=int, default=1) 
parser.add_argument("--hid", type=int, default=128) 
parser.add_argument("--aug", type=float, default=0.1) 
parser.add_argument("--dropout", type=float, default=0.2) 
parser.add_argument("--patience", type=int, default=10) 
parser.add_argument("--epochs", type=int, default=200) 
parser.add_argument("--folds", type=int, default=1) 
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--seed", type=int, default=2025)
parser.add_argument("--run_id", type=str, default=None)
args = parser.parse_args()


Seed_everything(seed=args.seed)
device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
print(device)

if args.run_id == None:
    run_id = 'run_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
else: 
    run_id = args.run_id
output_path = os.path.join(args.output_path, run_id)
os.makedirs(output_path, exist_ok = True)

log = open(os.path.join(output_path, 'test.log'), 'w', buffering=1)

test_data = process_dataset(os.path.join(args.dataset_path, 'examples.csv'))

test_dataset = ProteinGraphDataset(test_data, range(len(test_data)), args)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, prefetch_factor=2)

state_dict = torch.load('./models/PMHCI_Binding/best_model_random.ckpt', device)
model = PMHCBinder(node_input_dim=1238, edge_input_dim=451, hidden_dim=args.hid, num_layers=args.layer, dropout=args.dropout, augment_eps=args.aug, device=device).to(device)
model.load_state_dict(state_dict)
model.eval()

test_pred_dict = {} 
test_pred = []
test_y = []
for data in tqdm(test_dataloader):
    data = data.to(device)
    with torch.no_grad():
        outputs = model(data.X, data.node_feat, data.edge_index, data.seq, data.batch, 0, data.edge_feat, data.entity).sigmoid()
        
    test_y += list(data.y.detach().cpu().numpy().tolist())
    test_pred += list(outputs.detach().cpu().numpy().tolist())

    IDs = data.name
    for i, ID in enumerate(IDs):
        test_pred_dict[ID] = outputs[i].detach().cpu().numpy()

print(test_pred)
print(test_y)

with open(os.path.join(output_path, "test_pred_dict.pkl"), "wb") as f:
    pickle.dump(test_pred_dict, f)