import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from typing import Callable, Iterable, Tuple, Union, List
from pandas import DataFrame, Timedelta
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from functools import lru_cache
from typing import Iterable
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics as skl_metrics
import os
from pathlib import Path
import pandas as pd

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
        sample_rate: Union[int, Tuple[int], float] = 2
        **kwargs,
    ):
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.verify_existing = verify_existing
        self.drop_na_labels = drop_na_labels
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.kwargs = kwargs

        # Additional inputs are optional.
        self.extra_inputs = extra_inputs
        if self.extra_inputs is not None and isinstance(self.extra_inputs, str):
            self.extra_inputs = [self.extra_inputs]

        self.labels = labels
        if self.labels is None and self.verbose:
            print(
                "No label column names were provided, only filenames and inputs will be returned."
            )
        if (self.labels is not None) and isinstance(self.labels, str):
            self.labels = [self.labels]

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

        # Make sure that there are no NAN labels
        if (self.labels is not None) and self.drop_na_labels:
            old_len = len(self.manifest)
            self.manifest = self.manifest.dropna(subset=self.labels)
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} examples contained NaN value(s) in their labels and were dropped."
                )
        elif (self.labels is not None) and (not self.drop_na_labels):
            print(
                "self.drop_na_labels is set to False, so it's possible for the manifest to contain NaN-valued labels."
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

        labels = row[self.labels] if self.labels is not None else None
        if self.labels is not None and not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.float32)

        extra_inputs = row[self.extra_inputs] if self.extra_inputs is not None else None
        if self.extra_inputs is not None and not torch.is_tensor(extra_inputs):
            output["extra_inputs"] = torch.tensor(extra_inputs, dtype=torch.float32)

        output["labels"] = labels

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

    def process_manifest(self, manifest):
        manifest = super().process_manifest(manifest)
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


    def __len__(self):
        return len(self.manifest)

class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model,
        index_labels=None,
        save_predictions_path=None,
    ):
        super().__init__()
        self.m = model
        self.labels = index_labels
        if self.labels is not None and isinstance(self.labels, str):
            self.labels = [self.labels]
        if isinstance(save_predictions_path, str):
            save_predictions_path = Path(save_predictions_path)
        self.save_predictions_path = save_predictions_path
    def prepare_batch(self, batch):
        batch["labels"] = batch["labels"].long()
        batch["primary_input"] = batch["primary_input"].float()
        return batch