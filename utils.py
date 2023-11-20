import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms.functional as TF
import wandb
import yaml
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
import wandb
from sklearn import metrics as skl_metrics
import torchvision
import os
from pathlib import Path
import pandas as pd


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
def write_video(
    path: Union[str, Path],
    frames: np.array,  # (n_frames, height, width, 3) as np.uint8
    fps: float = 50.0,
    fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
):
    """
    Writes and saves an avi video file from an array of frames.

    Args:
        path: Filepath to save video to
        frames: Numpy array of shape (n_frames, height, width, 3) and dtype np.uint8
        fps: Float framerate of video
        fourcc: Four character code representing the video codec to be used. Default is MJPG
    """
    n_frames, height, width, channels = frames.shape
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
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
    zoom: float = 0,
) -> ArrayLike:

class TrainingModel(pl.LightningModule):
    def __init__(
        self,
        model,
        metrics: Iterable[TrainingMetric] = None,
        tracked_metric=None,
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        super().__init__()
        self.epoch_preds = {"train": ([], []), "val": ([], [])}
        self.epoch_losses = {"train": [], "val": []}
        self.metrics = {}
        self.metric_funcs = {m.name: m for m in metrics}
        self.tracked_metric = f"epoch_val_{tracked_metric}"
        self.best_tracked_metric = None
        self.early_stop_epochs = early_stop_epochs
        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.metrics["epochs_since_last_best"] = 0
        self.m = model
        self.training_steps = 0
        self.steps_since_checkpoint = 0
        self.labels = index_labels
        if self.labels is not None and isinstance(self.labels, str):
            self.labels = [self.labels]
        if isinstance(save_predictions_path, str):
            save_predictions_path = Path(save_predictions_path)
        self.save_predictions_path = save_predictions_path
        self.lr = lr
        self.step_loss = (None, None)

        self.log_path = Path(wandb.run.dir) if wandb.run is not None else None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def forward(self, x):
        return self.m(x)

    def step(self, batch, step_type="train"):
        batch = self.prepare_batch(batch)
        if "extra_inputs" not in batch.keys():
            y_pred = self(batch["primary_input"])
        else:
            x = (batch["primary_input"], batch["extra_inputs"])
            y_pred = self(x)

        if not step_type == "predict":
            loss = self.loss_func(y_pred, batch["labels"])
            if torch.isnan(loss):
                raise ValueError(loss)

            self.log_step(step_type, batch["labels"], y_pred, loss)

            return loss
        else:
            return y_pred
    def training_step(self, batch, i):
        return self.step(batch, "train")

    def validation_step(self, batch, i):
        return self.step(batch, "val")

    def predict_step(self, batch, *args):
        y_pred = self.step(batch, "predict")
        return {"filename": batch["filename"], "prediction": y_pred.cpu().numpy()}

    def on_predict_epoch_end(self, results):

        for i, predict_results in enumerate(results):
            filename_df = pd.DataFrame(
                {
                    "filename": np.concatenate(
                        [batch["filename"] for batch in predict_results]
                    )
                }
            )

            if self.labels is not None:
                columns = [f"{class_name}_preds" for class_name in self.labels]
            else:
                columns = ["preds"]
            outputs_df = pd.DataFrame(
                np.concatenate(
                    [batch["prediction"] for batch in predict_results], axis=0
                ),
                columns=columns,
            )

            prediction_df = pd.concat([filename_df, outputs_df], axis=1)

            dataloader = self.trainer.predict_dataloaders[i]
            manifest = dataloader.dataset.manifest
            prediction_df = prediction_df.merge(manifest, on="filename", how="outer")
            if wandb.run is not None:
                prediction_df.to_csv(
                    Path(wandb.run.dir).parent
                    / "data"
                    / f"dataloader_{i}_predictions.csv",
                    index=False,
                )
            if self.save_predictions_path is not None:

                if ".csv" in self.save_predictions_path.name:
                    prediction_df.to_csv(
                        self.save_predictions_path.parent
                        / self.save_predictions_path.name.replace(".csv", f"_{i}_.csv"),
                        index=False,
                    )
                else:
                    prediction_df.to_csv(
                        self.save_predictions_path / f"dataloader_{i}_predictions.csv",
                        index=False,
                    )

            if wandb.run is None and self.save_predictions_path is None:
                print(
                    "WandB is not active and self.save_predictions_path is None. Predictions will be saved to the directory this script is being run in."
                )
                prediction_df.to_csv(f"dataloader_{i}_predictions.csv", index=False)

    def log_step(self, step_type, labels, output_tensor, loss):
        self.step_loss = (step_type, loss.detach().item())
        self.epoch_preds[step_type][0].append(labels.detach().cpu().numpy())
        self.epoch_preds[step_type][1].append(output_tensor.detach().cpu().numpy())
        self.epoch_losses[step_type].append(loss.detach().item())
        if step_type == "train":
            self.training_steps += 1
            self.steps_since_checkpoint += 1
            if (
                self.checkpoint_every_n_steps is not None
                and self.steps_since_checkpoint > self.checkpoint_every_n_steps
            ):
                self.steps_since_checkpoint = 0
                self.checkpoint_weights(f"step_{self.training_steps}")

    def checkpoint_weights(self, name=""):
        if wandb.run is not None:
            weights_path = Path(wandb.run.dir).parent / "weights"
            if not weights_path.is_dir():
                weights_path.mkdir()
            torch.save(self.state_dict(), weights_path / f"model_{name}.pt")
        else:
            print("Did not checkpoint model. wandb not initialized.")

    def validation_epoch_end(self, preds):

        # Save weights
        self.metrics["epoch"] = self.current_epoch
        if self.checkpoint_every_epoch:
            self.checkpoint_weights(f"epoch_{self.current_epoch}")

        # Calculate metrics
        for m_type in ["train", "val"]:

            y_true, y_pred = self.epoch_preds[m_type]
            if len(y_true) == 0 or len(y_pred) == 0:
                continue
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

            self.metrics[f"epoch_{m_type}_loss"] = np.mean(self.epoch_losses[m_type])
            for m in self.metric_funcs.values():
                self.metrics.update(
                    m(
                        y_true,
                        y_pred,
                        labels=self.labels,
                        split=m_type,
                        step_type="epoch",
                    )
                )

            # Reset predictions
            self.epoch_losses[m_type] = []
            self.epoch_preds[m_type] = ([], [])

        # Check if new best epoch
        if self.metrics is not None and self.tracked_metric is not None:
            if self.tracked_metric == "epoch_val_loss":
                metric_optimization = "min"
            else:
                metric_optimization = self.metric_funcs[
                    self.tracked_metric.replace("epoch_val_", "")
                ].optimum
            if (
                self.metrics[self.tracked_metric] is not None
                and (
                    self.best_tracked_metric is None
                    or (
                        metric_optimization == "max"
                        and self.metrics[self.tracked_metric] > self.best_tracked_metric
                    )
                    or (
                        metric_optimization == "min"
                        and self.metrics[self.tracked_metric] < self.best_tracked_metric
                    )
                )
                and self.current_epoch > 0
            ):
                print(
                    f"New best epoch! {self.tracked_metric}={self.metrics[self.tracked_metric]}, epoch={self.current_epoch}"
                )
                self.checkpoint_weights(f"best_{self.tracked_metric}")
                self.metrics["epochs_since_last_best"] = 0
                self.best_tracked_metric = self.metrics[self.tracked_metric]
            else:
                self.metrics["epochs_since_last_best"] += 1
            if self.metrics["epochs_since_last_best"] >= self.early_stop_epochs:
                raise KeyboardInterrupt("Early stopping condition met")
        # Log to w&b
        if wandb.run is not None:
            wandb.log(self.metrics)

class EchoDataset(CedarsDataset):
    """
    Dataset class intended for use with a large directory of echo videos stored
    in .avi format.

    Expects a manifest with filenames and labels, and optionally split,
    view, frame count, and/or view confidence columns for further
    subset-making. Returns videos of a set length and at a given
    resolution in the form of PyTorch Tensors with shape (3, H, W)
    """

    def __init__(
        self,
        # CedarsDataset params
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: Union[List[str], str] = None,
        update_manifest_func: Callable = None,
        subsample: float = None,
        verbose: bool = True,
        do_augmentation: bool = False,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        # EchoDataset params
        view: str = None,
        view_threshold: float = None,
        random_start: bool = False,
        n_frames: int = 16,
        sample_rate: Union[int, Tuple[int], float] = 2,
        interpolate_frames: bool = False,
        resize_res: Tuple[int] = None,
        zoom: float = 0,
        do_random_affine: bool = True,
        random_rotation: float = 0,
        random_translation: Tuple[int] = None,
        random_scale: Tuple[int] = None,
        random_shear: Tuple[int] = None,
        **kwargs,
    ):
        """
        Args:
            view: str -- optional, if a manifest has a column called 'view', keep only rows where 'view' is equal to this argument. Useful if you want to
                         create a dataset that has only A4C echoes, for example.
            view_threshold: float -- optional, float
        """

        self.view = view
        self.view_threshold = view_threshold
        self.random_start = random_start
        self.n_frames = n_frames
        self.sample_rate = sample_rate
        self.interpolate_frames = interpolate_frames
        self.resize_res = resize_res
        self.zoom = zoom
        self.do_random_affine = do_random_affine
        if self.do_random_affine:
            self.random_affine = torchvision.transforms.RandomAffine(
                degrees=random_rotation,
                translate=random_translation,
                scale=random_scale,
                shear=random_shear,
            )
        super().__init__(
            data_path=data_path,
            manifest_path=manifest_path,
            split=split,
            labels=labels,
            update_manifest_func=update_manifest_func,
            subsample=subsample,
            verbose=verbose,
            do_augmentation=do_augmentation,
            verify_existing=verify_existing,
            drop_na_labels=drop_na_labels,
            **kwargs,
        )

    def __len__(self):
        return len(self.manifest)

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

    def augment(self, x, y):
        if self.do_random_affine:
            x = self.random_affine(x)
        return x, y

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
class CedarsDataset(Dataset):
    """
    Generic parent class for several differnet kinds of common datasets we use
    here at Cedars CVAIR.

    Expects to be used in a scenario where you have a big folder full of
    input examples (videos, ecgs, 3d arrays, images, etc.) and a big CSV
    that contains metadata and labels for those examples, called a
    'manifest'. This dataset
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        extra_inputs: List[str] = None,
        labels: List[str] = None,
        update_manifest_func: Callable = None,
        subsample: Union[int, float] = None,
        verbose: bool = True,
        do_augmentation: bool = False,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        **kwargs,
    ):
        """
        Args:
            data_path: the path to a directory full of files you want to read from when accessing the dataset.
            manifest_path: the path to a CSV filed containing filenames found in data_path and labels for those filenames.
            split: optional, allows user to select which split of the manifest to use assuming the presence of a categorical 'split' column.
                   defaults to None so that the entire manifest is used by default.
            inputs: optional, a list of column names in the manifest that contain the inputs to the model. "filename" will always be
                    the first item in this list.
            labels: name(s) of column(s) in manifest that contain labels, in the order you want them returned. Defaults to
                                               None. If set to None, the dataset will not return any labels, only filenames and inputs.
            update_manifest_func: optional, allows user to pass in a function to preprocess the manifest after it is loaded and before the
                                              dataset does anything to it.
            subsample: optional, a number indicating how many examples to randomly subsample from the manifest. Useful for debugging
                                     runs and sweeps where you want to speed up the run by using only a part of the dataset.
            verbose: whether to print out progress statements when initializing. Defaults to True.
            do_augmentation: whether to apply data augmentation to each example as specified in a child class's self.augment method.
        """

        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.subsample = subsample
        self.do_augmentation = do_augmentation
        self.verify_existing = verify_existing
        self.drop_na_labels = drop_na_labels
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

        # do manifest processing that's specific to a given task (different from update_manifest_func,
        # exists as a method overridden in child classes)
        self.manifest = self.process_manifest(self.manifest)

        # Apply user-provided update function to manifest
        if update_manifest_func is not None:
            self.manifest = update_manifest_func(self, self.manifest)

        # Usually set to "train", "val", or "test". If set to None, the entire manifest is used.
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

        # Option to subsample dataset for doing smaller, faster runs
        if self.subsample is not None:
            if isinstance(self.subsample, int):
                self.manifest = self.manifest.sample(n=self.subsample)
            else:
                self.manifest = self.manifest.sample(frac=self.subsample)
            if verbose:
                print(f"{self.subsample} examples subsampled.")

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

        # Save manifest to weights and biases run directory
        if wandb.run is not None:
            run_data_path = Path(wandb.run.dir).parent / "data"
            if not run_data_path.is_dir():
                run_data_path.mkdir()

            save_name = "manifest.csv"
            if split is not None:
                save_name = f"{split}_{save_name}"

            self.manifest.to_csv(run_data_path / save_name)

            if self.verbose:
                print(f"Copy of manifest saved to {run_data_path}")

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

        if self.do_augmentation:
            # self.augment expected in child classes. Must also handle labels=None
            primary_input, labels = self.augment(primary_input, labels)
        if not torch.is_tensor(primary_input):
            primary_input = torch.tensor(primary_input, dtype=torch.float32)
        output["primary_input"] = primary_input

        extra_inputs = row[self.extra_inputs] if self.extra_inputs is not None else None
        if self.extra_inputs is not None and not torch.is_tensor(extra_inputs):
            output["extra_inputs"] = torch.tensor(extra_inputs, dtype=torch.float32)

        output["labels"] = labels

        return output

    def process_manifest(self, manifest: DataFrame) -> DataFrame:
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

    def augment(self, primary_input, labels):
        if self.verbose:
            print("self.augment method has not been overridden.")
        return primary_input, labels
class BinaryClassificationModel(TrainingModel):
    def __init__(
        self,
        model,
        metrics=(roc_auc_metric, CumulativeMetric(roc_auc_metric, np.nanmean, "mean")),
        tracked_metric="mean_roc_auc",
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        super().__init__(
            model=model,
            metrics=metrics,
            tracked_metric=tracked_metric,
            early_stop_epochs=early_stop_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            index_labels=index_labels,
            save_predictions_path=save_predictions_path,
            lr=lr,
        )
        self.loss_func = nn.BCEWithLogitsLoss()

    def prepare_batch(self, batch):
        if len(batch["labels"].shape) == 1:
            batch["labels"] = batch["labels"][:, None]
        return batch
class MultiClassificationModel(TrainingModel):
    def __init__(
        self,
        model,
        metrics=(roc_auc_metric, CumulativeMetric(roc_auc_metric, np.mean, "mean")),
        tracked_metric="mean_roc_auc",
        early_stop_epochs=10,
        checkpoint_every_epoch=False,
        checkpoint_every_n_steps=None,
        index_labels=None,
        save_predictions_path=None,
        lr=0.01,
    ):
        metrics = [*metrics]
        super().__init__(
            model=model,
            metrics=metrics,
            tracked_metric=tracked_metric,
            early_stop_epochs=early_stop_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            index_labels=index_labels,
            save_predictions_path=save_predictions_path,
            lr=lr,
        )
        self.loss_func = SqueezeCrossEntropyLoss()

    def prepare_batch(self, batch):
        batch["labels"] = batch["labels"].long()
        batch["primary_input"] = batch["primary_input"].float()
        return batch

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

    