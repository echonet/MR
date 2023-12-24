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


# class ClassificationModel(pl.LightningModule):
#     def __init__(
#         self,
#         model,
#         index_labels=None,
#         save_predictions_path=None,
#     ):
#         super().__init__()
#         self.m = model
#         self.labels = index_labels
#         if self.labels is not None and isinstance(self.labels, str):
#             self.labels = [self.labels]
#         if save_predictions_path is not None and isinstance(save_predictions_path, str):
#             save_predictions_path = Path(save_predictions_path)
#         else:
#             save_predictions_path = Path(os.path.dirname(os.path.abspath(__file__)))
#         self.save_predictions_path = save_predictions_path
#     def prepare_batch(self, batch):
#         batch["labels"] = batch["labels"].long()
#         batch["primary_input"] = batch["primary_input"].float()
#         return batch

#     def forward(self, inputs, target):
#         return self.model(inputs, target)

def MR_preds_cm(test_predictions):
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

    # for _, spine in res.spines.items(): 
    #     spine.set_visible(True) 
    #     spine.set_linewidth(5)
        
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
    
    print('PPV is ' + str(ppv_preds))
    print('F1 is ' + str(f1_preds))
    print('Recall is ' + str(recall_preds))
