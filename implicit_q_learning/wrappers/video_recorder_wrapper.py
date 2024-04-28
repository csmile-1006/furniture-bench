import os

import numpy as np
import cv2
import imageio
from pathlib import Path


class VideoRecorder:
    def __init__(self, root_dir: Path, render_size: int = 256, fps: int = 20, target_key: str = "image"):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            # self.save_dir.mkdir(exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self._target_key = target_key
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            self.frames.append(obs)

    def save(self, file_name, *args, **kwargs):
        if self.enabled:
            path = self.save_dir / f"{file_name}.gif"
            imageio.mimsave(str(path), self.frames, duration=int(1000 / self.fps))


class VideoRecorderList:
    def __init__(self):
        self.recorders = []

    def add_recorder(self, root_dir: Path, render_size: int = 256, fps: int = 20, target_key: str = "image"):
        recorder = VideoRecorder(root_dir, render_size, fps, target_key)
        self.recorders.append(recorder)

    def init(self, obs, enabled=True):
        for i, recorder in enumerate(self.recorders):
            recorder.init(obs[i], enabled)
        
    def init_idx(self, obs, idx, enabled=True):
        self.recorders[idx].init(obs, enabled)

    def record(self, obs):
        for i, recorder in enumerate(self.recorders):
            recorder.record(obs[i])

    def save(self, file_name, env_idx: int=None):
        for i, recorder in enumerate(self.recorders):
            if env_idx is not None and env_idx != i:
                continue
            if recorder.enabled:
                # Save it in mp4 format.
                path = recorder.save_dir / f"{file_name}.mp4"
                fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
                width, height = recorder.frames[0].shape[1], recorder.frames[0].shape[0]
                out = cv2.VideoWriter(str(path), fourcc, recorder.fps, (width, height))
                for frame in recorder.frames:
                    # OpenCV expects frames in BGR format
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                out.release()


class TrainVideoRecorder:
    def __init__(self, root_dir: Path, render_size: int = 256, fps: int = 20):
        if root_dir is not None:
            self.save_dir = root_dir / "train_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(
                obs[-3:].transpose(1, 2, 0), dsize=(self.render_size, self.render_size), interpolation=cv2.INTER_CUBIC
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
