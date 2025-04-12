from soundevent import data, io
import numpy as np
import pandas as pd
from torchaudio import load
from torchaudio.functional import resample
from torch.utils.data import Dataset
import torch
from torchaudio.transforms import Vol

def load_audio(file, sr_to=48000):
    s, sr = load(file)
    if s.max() > 1: # if float, convert to int16
        s = s / 2**16
    s = resample(s, sr, sr_to)
    return s


class WhombatDataset(Dataset):
    def __init__(self, proj_path: str, label_names: list, win_sec: int = 4, 
                 sr: int = 48000, transform=None):
        """
        Load Whombat project data and prepare it for training.
        Since I annotated 4-sec clips for each 12 sec sound sources,
        3 clips will be returned for each audio file if win_sec=4.
        2 clips will be returned for each audio file if win_sec=8.
        1 clip will be returned for each audio file if win_sec=12.

        Parameters:
        ----------
        proj_path: str
            Path to the Whombat project file.
        label_names: list
            List of label names.
        win_sec: int
            Window size in seconds. 4, 8 or 12. Default is 4.
            If 4, 3 clips will be returned for each audio file.
            If 8, 2 clips will be returned for each audio file.
            If 12, 1 clip will be returned for each audio file.
        sr: int
            Sampling rate. Default is 48000.
        """
        self.proj = io.load(proj_path)
        self.win_sec = win_sec
        self.sr = sr
        self.label_names = label_names
        self.clip_status = {
            task.clip.uuid: self.task_is_complete(task)
            for task in self.proj.tasks
        }
        self.transform = transform
        self.clips = [
            annotation
            for annotation in self.proj.clip_annotations
            if not self.clip_annotation_has_issues(annotation)
            and self.clip_annotation_is_complete(annotation)
        ]
        self.df = pd.DataFrame({
            "path": np.array([clip.clip.recording.path for clip in self.clips]),
            "start": np.array([clip.clip.start_time for clip in self.clips]),
            "end": np.array([clip.clip.end_time for clip in self.clips]),
        })
        # Add labels 
        for label in self.label_names:
            self.df[label] = 0
        for clip in self.clips:
            for tag in clip.tags:
                # 頭文字が大文字でラベルをつけてしまったので、小文字に変換して
                if tag.value.lower() in self.label_names:
                    self.df.loc[self.df.path == clip.clip.recording.path, tag.value.lower()] = 1
        # Load audio sources and labels
        print("Loading audio sources and labels...")
        if self.win_sec == 4:
            self.sources = []
            self.labels = []
            for i in range(len(self.df)):
                start = self.df.loc[i, "start"]
                end = self.df.loc[i, "end"]
                source = load_audio(self.df.loc[i, "path"], sr_to=self.sr)
                self.sources.append(source[:, int(start*self.sr):int(end*self.sr)])
                self.labels.append(torch.tensor(self.df.loc[i, self.label_names].values.astype(np.float32)))
        elif self.win_sec == 8:
            self.sources = []
            self.labels = []
            for path in np.unique(self.df.path):
                clips = self.df[self.df.path == path].sort_values("start")
                source = load_audio(path, sr_to=self.sr)
                for i in range(2):
                    start = clips.iloc[i, :]["start"]
                    end = clips.iloc[i+1, :]["end"]
                    self.sources.append(source[:, int(start*self.sr):int(end*self.sr)])
                    label = np.max(np.stack([clips.iloc[i, :][self.label_names].values.astype(np.float32),
                        clips.iloc[i+1, :][self.label_names].values.astype(np.float32)]), axis=0)
                    self.labels.append(torch.tensor(label))
        elif self.win_sec == 12:
            self.sources = []
            self.labels = []
            for path in np.unique(self.df.path):
                clips = self.df[self.df.path == path].sort_values("start")
                source = load_audio(path, sr_to=self.sr)
                self.sources.append(source)
                label = np.max(np.stack([clips.iloc[0, :][self.label_names].values.astype(np.float32),
                    clips.iloc[1, :][self.label_names].values.astype(np.float32),
                    clips.iloc[2, :][self.label_names].values.astype(np.float32)]), axis=0)
                self.labels.append(torch.tensor(label))
        
        self.sources = torch.stack(self.sources)
        if self.transform:
            self.sources = self.transform(self.sources)
            
        self.labels = torch.stack(self.labels)
    
    def __len__(self):
        return len(self.sources)
    
    def __getitem__(self, idx):
        return self.sources[idx], self.labels[idx]
                
    def task_is_complete(self, task: data.AnnotationTask) -> bool:
        """Check if an annotation task is complete.

        A task is considered complete if it has a 'completed' status badge
        and does not have a 'rejected' badge (indicating it needs review).
        """
        for badge in task.status_badges:
            if badge.state == data.AnnotationState.rejected:
                # Task needs review, so it's not considered complete.
                return False

            if badge.state == data.AnnotationState.completed:
                # Task is explicitly marked as complete.
                return True

        # If no 'completed' badge is found, the task is not complete.
        return False

    def clip_annotation_is_complete(self, annotation: data.ClipAnnotation) -> bool:
        """Check if a clip annotation is complete based on its task status."""
        if annotation.clip.uuid not in self.clip_status:
            # If the clip is not part of the project's tasks, ignore it.
            return False

        # Return the pre-computed completion status from the clip_status dictionary.
        return self.clip_status[annotation.clip.uuid]

    def clip_annotation_has_issues(self, annotation: data.ClipAnnotation) -> bool:
        """Check if a clip annotation has any associated issues."""
        return any(note.is_issue for note in annotation.notes)