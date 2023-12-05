import torch
import torch.nn as nn
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import sys
import argparse
sys.path.append('/workspace/cvair/')
from utils import EchoDataset
from utils import MultiClassificationModel
from utils import BinaryClassificationModel
from pathlib import Path
import torchvision
import os
import math
# from utils import bootstrap_ppv_f1_recall
from sklearn import metrics
from matplotlib import pyplot as plt
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

# CLASS_LIST = ['Control', 'Mild', 'Moderate', 'Severe']

# if __name__ == '__main__':
#     if os.name == 'nt':
#         os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
#         print('Windows detected!')

# parser = argparse.ArgumentParser(description='Run inference on dataset')
# parser.add_argument('--MR_label_col', type=str, help='Name of MR ground truth label column in manifest; drops cases with NaN ground truth')
# parser.add_argument('--manifest_path', type=str, help='Path to Manifest File')
# parser.add_argument('--data_path', type=str, help='Path to Normalized EKGs')
# parser.add_argument('--save_predictions_path', type=str, default='./')
# args = parser.parse_args()

# data_path = args.data_path
# manifest_path = args.manifest_path
# manifest = pd.read_csv(manifest_path)
# manifest['split'] = 'test'
# manifest['case'] = 1
# manifest['final_class'] = manifest['MR_label_col']
# manifest['final_class_label'] = manifest.final_class.apply(CLASS_LIST.index)
# manifest.to_csv(manifest_path, index = False)

# view_classifier_weights_path = 'view_classifier_iteration_3_weights.pt'

# ## Run view classification prediction model
# test_ds = EchoDataset(split='test',
#                     data_path=data_path,
#                     manifest_path=manifest_path,
#                     labels=['case'],
#                     update_manifest_func=None)

# test_dl = DataLoader(test_ds, num_workers=16, batch_size=124, drop_last=False, shuffle=False)
# backbone = torchvision.models.video.r2plus1d_18(pretrained=False)
# backbone.fc = nn.Linear(512, 1)
# model = BinaryClassificationModel(backbone, save_predictions_path = os.getcwd())
# weights = torch.load(view_classifier_weights_path)
# print(model.load_state_dict(weights))
# trainer = Trainer(gpus=1, devices=[0])
# trainer.predict(model, dataloaders=test_dl)

# os.rename(Path(os.getcwd())/Path('dataloader_predictions_0.csv'), Path(os.getcwd())/Path('view_classification_predictions.csv'))
# manifest = pd.read_csv(Path(os.getcwd())/Path('view_classification_predictions.csv'))
# manifest.preds = manifest.preds.apply(sigmoid)
# manifest = manifest[manifest.preds > 0.519]
# manifest.to_csv(Path(os.path.dirname(os.path.abspath(__file__)))/Path('view_classification_predictions_above_threshold.csv'), index = False)

# # ## Run MR prediction model

# # MR_weights_path = '.pt'

# test_ds = EchoDataset(split='test',
#                     data_path=data_path,
#                     manifest_path=Path(os.path.dirname(os.path.abspath(__file__)))/Path('view_classification_predictions_above_threshold.csv'),
#                     labels=args.MR_label_col,
#                     update_manifest_func=None)

# test_dl = DataLoader(test_ds, num_workers=8, batch_size=24, drop_last=False, shuffle=False)

# backbone = torchvision.models.video.r2plus1d_18(num_classes = len(CLASS_LIST))

# model = MultiClassificationModel(backbone, save_predictions_path = Path(os.getcwd()), index_labels = CLASS_LIST)

# weights = torch.load(MR_weights_path)
# print(model.load_state_dict(weights))
# trainer = Trainer(gpus = [1])
# trainer.predict(model, dataloaders=test_dl)




