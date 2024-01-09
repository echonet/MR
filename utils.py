import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from typing import Tuple, Union, List
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from functools import lru_cache
import pytorch_lightning as pl
import torch
import numpy as np
from sklearn import metrics
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import math
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib import rc


def crop_and_scale(
    img: ArrayLike, res: Tuple[int], interpolation=cv2.INTER_CUBIC, zoom: float = 0.0
) -> ArrayLike:
    """Takes an image, a resolution, and a zoom factor as input, returns the
    zoomed/cropped image."""
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    # Crop to correct aspect ratio
    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]

    # Apply zoom
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    # Resize image
    img = cv2.resize(img, res, interpolation=interpolation)

    return img

def read_video(
    path: Union[str, Path],
    n_frames: int = None,
    sample_period: int = 1,
    out_fps: float = None,  # Output fps
    fps: float = None,  # input fps of video (default to avi metadata)
    frame_interpolation: bool = True,
    random_start: bool = False,
    res: Tuple[int] = None,  # (width, height)
    interpolation=cv2.INTER_CUBIC,
    zoom: float = 0):

    # Check path
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Get video properties
    cap = cv2.VideoCapture(str(path))
    vid_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    if out_fps is not None:
        sample_period = 1
        # Figuring out how many frames to read, and at what stride, to achieve the target
        # output FPS if one is given.
        if n_frames is not None:
            out_n_frames = n_frames
            n_frames = int(np.ceil((n_frames - 1) * fps / out_fps + 1))
        else:
            out_n_frames = int(np.floor((vid_size[0] - 1) * out_fps / fps + 1))

    # Setup output array
    if n_frames is None:
        n_frames = vid_size[0] // sample_period
    if n_frames * sample_period > vid_size[0]:
        raise Exception(
            f"{n_frames} frames requested (with sample period {sample_period}) but video length is only {vid_size[0]} frames"
        )
    if res is None:
        out = np.zeros((n_frames, *vid_size[1:], 3), dtype=np.uint8)
    else:
        out = np.zeros((n_frames, res[1], res[0], 3), dtype=np.uint8)

    # Read video, skipping sample_period frames each time
    if random_start:
        si = np.random.randint(vid_size[0] - n_frames * sample_period + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, si)
    for frame_i in range(n_frames):
        _, frame = cap.read()
        if res is not None:
            frame = crop_and_scale(frame, res, interpolation, zoom)
        out[frame_i] = frame
        for _ in range(sample_period - 1):
            cap.read()
    cap.release()

    # if a particular output fps is desired, either get the closest frames from the input video
    # or interpolate neighboring frames to achieve the fps without frame stutters.
    if out_fps is not None:
        i = np.arange(out_n_frames) * fps / out_fps
        if frame_interpolation:
            out_0 = out[np.floor(i).astype(int)]
            out_1 = out[np.ceil(i).astype(int)]
            t = (i % 1)[:, None, None, None]
            out = (1 - t) * out_0 + t * out_1
        else:
            out = out[np.round(i).astype(int)]

    if n_frames == 1:
        out = np.squeeze(out)
    return out, vid_size, fps

class EchoDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        n_frames: int = 16,
        random_start: bool = False,
        sample_rate: Union[int, Tuple[int], float] = 2,
        verbose: bool = True,
        resize_res: Tuple[int] = None,
        zoom: float = 0
    ):
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.verify_existing = verify_existing
        self.n_frames = n_frames
        self.random_start = random_start
        self.sample_rate = sample_rate
        self.resize_res = resize_res
        self.zoom = zoom
       
        # Read manifest file
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        if self.manifest_path.exists():
            self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )
        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]
        if self.verbose:
            print(
                f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist. This can be disabled for efficiency if
        # you have an especially large dataset
        if self.verify_existing:
            old_len = len(self.manifest)
            existing_files = os.listdir(self.data_path)
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing from {self.data_path}."
                )
        elif (not self.verify_existing) and self.verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present in {data_path}"
            )

        
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        output = {}
        row = self.manifest.iloc[index]
        filename = row["filename"]
        output["filename"] = filename

        # self.read_file expected in child classes
        primary_input = self.read_file(self.data_path / filename, row)
        output["primary_input"] = primary_input
        return output
   
    def read_file(self, filepath, row=None):

        if isinstance(self.sample_rate, int):  # Simple single int sample period
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=self.sample_rate,
                random_start=self.random_start,
            )
        elif isinstance(self.sample_rate, float):  # Fixed fps
            target_fps = self.sample_rate
            fps = row["fps"]

            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                1,
                fps=row["fps"],
                out_fps=target_fps,
                frame_interpolation=self.interpolate_frames,
                random_start=self.random_start,
                res=self.resize_res,
                zoom=self.zoom,
            )
        else:  # Tuple sample period ints to be randomly sampled from (1, 2, 3)
            sample_period = np.random.choice(
                [x for x in self.sample_rate if row["frames"] > x * self.n_frames]
            )
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=sample_period,
                random_start=self.random_start,
            )
        vid = torch.from_numpy(vid)
        vid = torch.movedim(vid / 255, -1, 0).to(torch.float32)
        return vid

    def process_manifest_one(self, manifest):
        if "mrn" in self.manifest.columns:
            self.manifest["mrn"] = self.manifest["mrn"].apply(zero_pad_20_digits)
        if "study_date" in self.manifest.columns:
            self.manifest["study_date"] = pd.to_datetime(self.manifest["study_date"])
        if "dob" in self.manifest.columns:
            self.manifest["dob"] = pd.to_datetime(self.manifest["dob"])
        if ("study_date" in self.manifest.columns) and ("dob" in self.manifest.columns):
            self.manifest["study_age"] = (
                self.manifest["study_date"] - self.manifest["dob"]
            ) / np.timedelta64(1, "Y")
        return manifest

    def process_manifest(self, manifest):
        manifest = process_manifest_one(manifest)
        if self.view is not None:
            if not isinstance(self.view, (List, Tuple)):
                self.view = [self.view]
            m = np.zeros(len(manifest), dtype=bool)
            for v in self.view:
                m |= manifest[v] >= self.view_threshold
            manifest = manifest[m]
        if "frames" in self.manifest.columns:
            if isinstance(self.sample_rate, int):  # Single sample period
                min_length = self.sample_rate * self.n_frames
                manifest = manifest[manifest["frames"] >= min_length]
            elif isinstance(self.sample_rate, float):  # Target fps
                target_fps = self.sample_rate
                manifest = manifest[
                    manifest["frames"]
                    > self.n_frames * manifest["fps"] / target_fps + 1
                ]
            else:  # Multiple possible sample periods
                min_length = min(self.sample_rate) * self.n_frames
                manifest = manifest[manifest["frames"] >= min_length]
        if "filename" not in manifest.columns and "file_uid" in manifest.columns:
            manifest["filename"] = manifest["file_uid"] + ".avi"
        return manifest

def process_preds(test_predictions):
    cols = ['Control_preds','Mild_preds','Moderate_preds','Severe_preds']

    for i in cols:
        test_predictions[i] = test_predictions[i].apply(sigmoid)

    test_predictions['predicted'] = test_predictions[cols].idxmax(axis = 1).astype(str).str.slice(stop = -6)
    ordered_indices = ['Control','Mild','Moderate','Severe']
    test_predictions['severe_binary'] = (test_predictions['final_class'].isin(['Severe'])*1)
    test_predictions['severe_binary_pred'] = (test_predictions['predicted'].isin(['Severe'])*1)
    test_predictions['mod_severe_binary'] = (test_predictions['final_class'].isin(['Moderate','Severe'])*1)
    test_predictions['mod_severe_binary_pred'] = (test_predictions['predicted'].isin(['Moderate','Severe'])*1)
    test_predictions['control_mild_binary'] = (test_predictions['final_class'].isin(['Control','Mild'])*1)
    test_predictions['control_mild_binary_pred'] = (test_predictions['predicted'].isin(['Control','Mild'])*1)
    test_predictions['moderate_binary'] = (test_predictions['final_class'].isin(['Moderate'])*1)
    test_predictions['moderate_binary_pred'] = (test_predictions['predicted'].isin(['Moderate'])*1)
    test_predictions['not_severe_binary'] = (~test_predictions['final_class'].isin(['Severe'])*1)
    test_predictions['not_severe_binary_pred'] = (~test_predictions['predicted'].isin(['Severe'])*1)
    test_predictions['Mod_Severe_preds'] = test_predictions[
        ['Moderate_preds','Severe_preds']].max(axis = 1, skipna = True)

    return(test_predictions)

def MR_preds_cm(test_predictions):
    cm = metrics.confusion_matrix(test_predictions['final_class'], test_predictions['predicted'])

    cm = pd.DataFrame(cm, columns = np.sort(test_predictions.final_class.unique()),
                    index = np.sort(test_predictions.final_class.unique()))

    plt.figure(figsize=(10,10))
    sns.set(font_scale=2)

    cm_try = cm.copy()
    cm_try.loc['Control'] = cm.loc['Control']/cm.loc['Control'].sum()
    cm_try.loc['Mild'] = cm.loc['Mild']/cm.loc['Mild'].sum()
    cm_try.loc['Moderate'] = cm.loc['Moderate']/cm.loc['Moderate'].sum()
    cm_try.loc['Severe'] = cm.loc['Severe']/cm.loc['Severe'].sum()

    cm_try = cm_try.rename(columns = {'Control':'None'}, index = {'Control':'None'})

    res = sns.heatmap(cm_try, annot=cm.to_numpy(), cmap='Greens', linewidths = 5, linecolor='black',fmt=',d',
            vmin = 0, vmax = cm_try.to_numpy().max(), cbar = False, square = True)   
        
    # Drawing the frame 
    res.axhline(y = 0, color='k',linewidth = 10) 
    res.axhline(y = cm.shape[1], color = 'k', 
                linewidth = 10) 
    
    res.axvline(x = 0, color = 'k', 
                linewidth = 10) 
    
    res.axvline(x = cm.shape[0],  
                color = 'k', linewidth = 10) 
    
    # show plot 
    plt.title('Mitral Regurgitation Classification - CSMC\n', fontsize = 25, fontweight = 'bold')
    plt.ylabel('Actual  ', fontsize = 23, rotation = 0, fontweight = 'bold')
    plt.xlabel('\nPredicted', fontsize = 23, fontweight = 'bold')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20, rotation = 0)

    plt.show()

def bootstrap(x,y):
    y_total,yhat_total = x,y
    fpr_boot = []
    tpr_boot = []
    aucs = []
    
    # bootstrap for confidence interval
    for i in tqdm(range(0,10000)):
        choices = np.random.choice(range(0,len(yhat_total)),int(len(yhat_total)/2))
        ground = y_total[choices]
        ground.index = range(0,len(ground))
        preds = yhat_total[choices]
        preds.index = range(0,len(preds))
        fpr,tpr, _ = metrics.roc_curve(ground,preds)
        fpr_boot.append(fpr)
        tpr_boot.append(tpr)
        aucs.append(metrics.auc(fpr,tpr))
    low,high = np.nanmean(aucs)-np.nanstd(aucs)*1.96,np.nanmean(aucs)+np.nanstd(aucs)*1.96
    lower_point = round(np.percentile(aucs,2.5),3)
    higher_point = round(np.percentile(aucs,97.5),3)
    preds = [lower_point, higher_point]
    return(preds)

def bootstrap_ppv_f1_recall(x,y):
    y_total,yhat_total = x,y
    ppv_list = []
    f1_score_list = []
    recall_list = []
    
    # bootstrap for confidence interval
    for i in tqdm(range(0,10000)):
        choices = np.random.choice(range(0,len(yhat_total)),int(len(yhat_total)/2))
        ground = y_total[choices]
        ground.index = range(0,len(ground))
        preds = yhat_total[choices]
        preds.index = range(0,len(preds))
        ppv_list.append(metrics.precision_score(ground, preds))
        f1_score_list.append(metrics.f1_score(ground, preds))
        recall_list.append(metrics.recall_score(ground, preds))
        
    lower_point_ppv = round(np.percentile(ppv_list,2.5),3)
    higher_point_ppv = round(np.percentile(ppv_list,97.5),3)
    
    lower_point_f1 = round(np.percentile(f1_score_list,2.5),3)
    higher_point_f1 = round(np.percentile(f1_score_list,97.5),3)
    
    lower_point_recall = round(np.percentile(recall_list,2.5),3)
    higher_point_recall = round(np.percentile(recall_list,97.5),3)

    
    ppv_preds = [lower_point_ppv, higher_point_ppv]
    f1_preds = [lower_point_f1, higher_point_f1]
    recall_preds = [lower_point_recall, higher_point_recall]
    
    return ppv_preds, f1_preds, recall_preds

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

def make_roc_curve(test_predictions):

    fpr_severe, tpr_severe, thresholds = metrics.roc_curve((test_predictions.final_class == 'Severe') * 1, 
                                            test_predictions.Severe_preds)
    fpr_mod_severe, tpr_mod_severe, thresholds = metrics.roc_curve((test_predictions.final_class.isin(
        ['Moderate','Severe']) * 1), test_predictions[['Moderate_preds','Severe_preds']].max(axis = 1))
    scaling_factor = 1.5
    fig = plt.figure(figsize=(8*scaling_factor,8*scaling_factor))
    lw = 2*scaling_factor
    ls = 'dashed'
    ext_val_color = 'C2'

    ## Make severe a solid line
    ## Make moderate to severe a dashed line

    plt.plot(fpr_severe, tpr_severe, linewidth = lw, color = ext_val_color, label = ('Severe - AUC: ' + str(round(metrics.auc(fpr_severe, tpr_severe),3))))
    plt.plot(fpr_mod_severe, tpr_mod_severe, linestyle = ls, linewidth = lw, color = ext_val_color, label = '≥ Moderate - AUC: ' + str(round(metrics.auc(fpr_mod_severe, tpr_mod_severe),3)))
    plt.xlabel('1- Specificity', fontsize = 20*scaling_factor, rotation = 0, labelpad=10)
    plt.ylabel('Sensitivity', fontsize = 20*scaling_factor, rotation = 90, labelpad=15)
    plt.xticks(fontsize = 17*scaling_factor)
    plt.yticks(fontsize = 17*scaling_factor)

    legend = plt.legend(title = "Severity", fontsize = 15*1.5)
    plt.setp(legend.get_title(),fontsize= 17*1.5)

def make_cm(test_predictions):
    ordered_indices = ['Control','Mild','Moderate','Severe']
    cm = metrics.confusion_matrix(test_predictions['final_class'], test_predictions['predicted'])
    cm = pd.DataFrame(cm, columns = np.sort(test_predictions.final_class.unique()),
                    index = np.sort(test_predictions.final_class.unique()))
    plt.figure(figsize=(10,10))
    sns.set(font_scale=2)

    # cm_try = cm/cm.sum(axis = 1)
    cm_try = cm.copy()
    cm_try.loc['Control'] = cm.loc['Control']/cm.loc['Control'].sum()
    cm_try.loc['Mild'] = cm.loc['Mild']/cm.loc['Mild'].sum()
    cm_try.loc['Moderate'] = cm.loc['Moderate']/cm.loc['Moderate'].sum()
    cm_try.loc['Severe'] = cm.loc['Severe']/cm.loc['Severe'].sum()

    cm_try = cm_try.rename(columns = {'Control':'None'}, index = {'Control':'None'})

    res = sns.heatmap(cm_try, annot=cm.to_numpy(), cmap='Greens', linewidths = 5, linecolor='black',fmt=',d',
            vmin = 0, vmax = cm_try.to_numpy().max(), cbar = False, square = True)   
        
    # Drawing the frame 
    res.axhline(y = 0, color='k',linewidth = 10) 
    res.axhline(y = cm.shape[1], color = 'k', 
                linewidth = 10) 
    
    res.axvline(x = 0, color = 'k', 
                linewidth = 10) 
    
    res.axvline(x = cm.shape[0],  color = 'k', linewidth = 10) 
    
    # show plot 
    plt.title('Mitral Regurgitation Classification\n', fontsize = 25, fontweight = 'bold')
    plt.ylabel('Actual  ', fontsize = 23, rotation = 0, fontweight = 'bold')
    plt.xlabel('\nPredicted', fontsize = 23, fontweight = 'bold')
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20, rotation = 0)
        
    plt.show()

def get_stats(manifest):
    severe_auc_range = str(bootstrap((manifest.final_class == 'Severe') * 1, manifest.Severe_preds))
    
    mod_severe_auc_range = str(bootstrap(manifest.final_class.isin(
        ['Moderate','Severe']) * 1, manifest[
        ['Moderate_preds','Severe_preds']].max(axis = 1)))
    
    severe_ppv_range, severe_f1_range, severe_recall_range  = bootstrap_ppv_f1_recall(manifest['severe_binary'], manifest['severe_binary_pred'])
    severe_npv_range, _, _  = bootstrap_ppv_f1_recall(manifest['not_severe_binary'], manifest['not_severe_binary_pred'])

    mod_severe_ppv_range, mod_severe_f1_range, mod_severe_recall_range  = bootstrap_ppv_f1_recall(manifest['mod_severe_binary'], manifest['mod_severe_binary_pred'])
    mod_severe_npv_range, _, _  = bootstrap_ppv_f1_recall(manifest['control_mild_binary'], manifest['control_mild_binary_pred'])

    fpr_severe, tpr_severe, thresholds = metrics.roc_curve((manifest.final_class == 'Severe') * 1, 
                                            manifest.Severe_preds)
    severe_auc = str(round(metrics.auc(fpr_severe, tpr_severe), 3))

    fpr_mod_severe, tpr_mod_severe, thresholds = metrics.roc_curve((manifest.final_class.isin(
        ['Moderate','Severe']) * 1), manifest[['Moderate_preds','Severe_preds']].max(axis = 1))

    mod_severe_auc = str(round(metrics.auc(fpr_mod_severe, tpr_mod_severe), 3))

    print('\nSevere MR Stats' )
    print('Severe MR AUC is ' + severe_auc + " " + severe_auc_range)
    


    print('Severe PPV is ' + str(round(metrics.precision_score(manifest['severe_binary'], manifest['severe_binary_pred'],
                       labels = ['Severe']),3)) + " " + str(severe_ppv_range))

    print('Severe NPV is ' + str(round(metrics.precision_score(manifest['not_severe_binary'], manifest['not_severe_binary_pred'],
                       labels = ['Severe']),3)) + " " + str(severe_npv_range))

    print('Severe Recall is ' + str(round(metrics.recall_score(manifest['severe_binary'], manifest['severe_binary_pred'],
                       labels = ['Severe']),3)) + " " + str(severe_recall_range))

    print('Severe F1-Score is ' + str(round(metrics.f1_score(manifest['severe_binary'], manifest['severe_binary_pred'],
                       labels = ['Severe']),3)) + " " + str(severe_f1_range) + '\n\n')


    print('Moderate/Severe MR Stats' )
    print('Moderate/Severe MR AUC is ' + mod_severe_auc + ' ' + mod_severe_auc_range)

    print('Moderate/Severe PPV is ' + str(round(metrics.precision_score(manifest['mod_severe_binary'], manifest['mod_severe_binary_pred'],
                       labels = ['Moderate/Severe']),3)) + " " + str(mod_severe_ppv_range))

    print('Moderate/Severe NPV is ' + str(round(metrics.precision_score(manifest['control_mild_binary'], manifest['control_mild_binary_pred'],
                       labels = ['Moderate/Severe']),3)) + " " + str(mod_severe_npv_range))

    print('Moderate/Severe Recall is ' + str(round(metrics.recall_score(manifest['mod_severe_binary'], manifest['mod_severe_binary_pred'],
                       labels = ['Moderate/Severe']),3)) + " " + str(mod_severe_recall_range))

    print('Moderate/Severe F1-Score is ' + str(round(metrics.f1_score(manifest['mod_severe_binary'], manifest['mod_severe_binary_pred'],
                 labels = ['Moderate/Severe']),3)) + " " + str(mod_severe_f1_range))


