import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import sys
import argparse
from pathlib import Path
import os
from utils import bootstrap_ppv_f1_recall, MR_preds_cm, sigmoid, bootstrap, EchoDataset
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

    manifest_path = r'stanford_view_classified_iteration_3_vc_preds_removed.csv'
    data_path = r'D:\amey\stanford_echos_MR_ext_val'

    # parser = argparse.ArgumentParser(description='Run inference on dataset')
    # parser.add_argument('--MR_label_col', type=str, help='Name of MR ground truth label column in manifest; drops cases with NaN ground truth')
    # parser.add_argument('--manifest_path', type=str, help='Path to Manifest File')
    # parser.add_argument('--data_path', type=str, help='Path to Normalized EKGs')
    # parser.add_argument('--save_predictions_path', type=str, default='./')
    # args = parser.parse_args()
    # data_path = args.data_path
    # manifest_path = args.manifest_path

    manifest = pd.read_csv(manifest_path)
    manifest["split"] = "test"
    manifest['final_class'] = manifest['MR_label_col']
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
    # df_preds = pd.DataFrame(data=[filenames, predictions], columns=["filename", "preds"])
    manifest = manifest.merge(df_preds, on="filename", how="inner")
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

    test_predictions = pd.read_csv('MR_model_predictions')
    test_predictions = test_predictions.drop_duplicates('filename')
    test_predictions.index = range(0,len(test_predictions))

    cols = ['Control_preds','Mild_preds','Moderate_preds','Severe_preds']

    for i in cols:
        manifest[i] = manifest[i].apply(sigmoid)

    manifest['predicted'] = manifest[cols].idxmax(axis = 1).astype(str).str.slice(stop = -6)

    manifest['severe_binary'] = (manifest['final_class'].isin(['Severe'])*1)
    manifest['severe_binary_pred'] = (manifest['predicted'].isin(['Severe'])*1)
    manifest['mod_severe_binary'] = (manifest['final_class'].isin(['Moderate','Severe'])*1)
    manifest['mod_severe_binary_pred'] = (manifest['predicted'].isin(['Moderate','Severe'])*1)
    manifest['control_mild_binary'] = (manifest['final_class'].isin(['Control','Mild'])*1)
    manifest['control_mild_binary_pred'] = (manifest['predicted'].isin(['Control','Mild'])*1)
    manifest['moderate_binary'] = (manifest['final_class'].isin(['Moderate'])*1)
    manifest['moderate_binary_pred'] = (manifest['predicted'].isin(['Moderate'])*1)

    manifest['not_severe_binary'] = (~manifest['final_class'].isin(['Severe'])*1)
    manifest['not_severe_binary_pred'] = (~manifest['predicted'].isin(['Severe'])*1)

    manifest['Mod_Severe_preds'] = manifest[
        ['Moderate_preds','Severe_preds']].max(axis = 1, skipna = True)

    manifest.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__))) / Path("MR_model_predictions.csv"),
        index=False,
    )

    
