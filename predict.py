import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import sys
import argparse
from pathlib import Path
import os
from utils import bootstrap_ppv_f1_recall, MR_preds_cm, sigmoid, bootstrap, EchoDataset, process_preds
from sklearn import metrics
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
from multiprocessing import freeze_support

if __name__ == "__main__":
    if os.name == 'nt':
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        print('Windows detected!')

with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    CLASS_LIST = ["Control", "Mild", "Moderate", "Severe"]

    parser = argparse.ArgumentParser(description='Run inference on dataset')
    parser.add_argument('--MR_label_col', type=str, help='Name of MR ground truth label column in manifest; should have values of one of ["Control", "Mild", "Moderate", "Severe"]')
    parser.add_argument('--manifest_path', type=str, help='Path to Manifest File')
    parser.add_argument('--data_path', type=str, help='Path to Normalized EKGs')

    args = parser.parse_args()
    data_path = args.data_path
    manifest_path = args.manifest_path

    manifest = pd.read_csv(manifest_path)
    manifest["split"] = "test"
    manifest['final_class'] = manifest[args.MR_label_col]
    manifest = manifest[manifest.final_class.isin(CLASS_LIST)]
    manifest['final_class_label'] = manifest.final_class.apply(CLASS_LIST.index)
    manifest.to_csv(manifest_path, index = False)

    view_classifier_weights_path = "view_classifier_iteration_3_weights.pt"

    ## Run view classification prediction model
    test_ds = EchoDataset(
        split="test", data_path=data_path, manifest_path=manifest_path)

    test_dl = DataLoader(
        test_ds, num_workers=0, batch_size=1, drop_last=False, shuffle=False
    )
    backbone = torchvision.models.video.r2plus1d_18(pretrained=False, num_classes=1)

    weights = torch.load(view_classifier_weights_path)
    weights = {k[2:]: v for k, v in weights.items()}
    print(backbone.load_state_dict(weights))
    backbone = backbone.to(device).eval()
    filenames = []
    predictions = []

    for batch in tqdm(test_dl):
        preds = backbone(batch["primary_input"].to(device))
        filenames.extend(batch["filename"])
        predictions.extend(preds.detach().cpu().squeeze(dim = 1))
    df_preds = pd.DataFrame({'filename': filenames, 'preds': predictions})
    manifest = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    manifest.preds = manifest.preds.apply(sigmoid)
    manifest = manifest[manifest.preds > 0.519]
    manifest.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / Path("view_classification_predictions_above_threshold.csv"),
        index=False,
    )

    # ## Run MR prediction model

    MR_weights_path = ("MR_manifest_nov_21_resplit_lowest_loss_epoch_11.pt")

    manifest = pd.read_csv('view_classification_predictions_above_threshold.csv')
    test_ds = EchoDataset(
        split="test",
        data_path=data_path,
        manifest_path=Path(os.path.dirname(os.path.abspath(__file__)))
        / Path("view_classification_predictions_above_threshold.csv")
    )

    test_dl = DataLoader(
        test_ds, num_workers=0, batch_size=1, drop_last=False, shuffle=False
    )
    backbone = torchvision.models.video.r2plus1d_18(
        pretrained=False, num_classes=len(CLASS_LIST)
    )
    weights = torch.load(MR_weights_path)
    weights = {k[2:]: v for k, v in weights.items()}
    print(backbone.load_state_dict(weights))
    backbone = backbone.to(device).eval()
    filenames = []
    predictions = []

    for batch in tqdm(test_dl):
        preds = backbone(batch["primary_input"].to(device))
        filenames.extend(batch["filename"])
        predictions.extend(preds.detach().cpu())
    predictions = torch.cat(predictions, dim=0).T
    predictions = torch.reshape(predictions, (int(len(predictions)/4), int(4)))
    control_preds = predictions[:,0]
    mild_preds = predictions[:,1]
    moderate_preds = predictions[:,2]
    severe_preds = predictions[:,3]
    d = {'filename':filenames,'Control_preds':control_preds,'Mild_preds':mild_preds,'Moderate_preds':moderate_preds,'Severe_preds':severe_preds}
    df_preds = pd.DataFrame(d)
    manifest = manifest.merge(df_preds, on="filename", how="inner")
    manifest = manifest.drop_duplicates('filename')
    manifest.index = range(0,len(manifest))
    manifest_processed = process_preds(manifest)

    manifest_processed.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__))) / Path("MR_model_predictions_non_anonymized.csv"),
        index=False,
    )

    process_preds(manifest[['filename','final_class','final_class_label','Control_preds','Mild_preds',
    'Moderate_preds','Severe_preds']]).to_csv(Path(os.path.dirname(os.path.abspath(__file__))
    ) / Path('MR_model_predictions_anonymized.csv'))

    print('Inference Complete. Please run the the notebook analyze_predictions.ipynb to analyze the results.')

    
