"""Public biosignal dataset loaders for NeuroAugment benchmarks.

Each class exposes:
  - download(root)          — fetch raw data into root/
  - load(root, ...)         — return (X, y, subjects, metas)
  - recommended_split(...)  — return standard train/val/test subject ids

ECG
----
MitBih    — MIT-BIH Arrhythmia (PhysioNet mitdb, 48 records, 360 Hz, 2ch)
PtbXl     — PTB-XL 12-Lead ECG  (PhysioNet ptb-xl, 21 799 records, 500 Hz, 12ch)

EEG
----
EegMmidb  — EEG Motor Movement/Imagery (PhysioNet eegmmidb, 109 subjects, 160 Hz, 64ch)
BciIv2a   — BCI-Competition IV 2a      (MOABB BNCI2014_001,  9 subjects,  250 Hz, 22ch)

IMU / HAR
----------
UciHar    — UCI Human Activity Recognition (30 subjects, 50 Hz, 6ch)
Pamap2    — PAMAP2 Physical Activity       (9 subjects, 100 Hz, 36ch)
Wisdm     — WISDM Activity Recognition    (36 subjects,  20 Hz,  3ch)
"""
from __future__ import annotations

import os
import zipfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(path: Path, msg: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{msg}\nExpected path: {path}\n"
            "Call .download(root) first, or set root to a directory that already contains the data."
        )


def _window_labels(label_trace: np.ndarray, window: int, stride: int, majority: bool = True) -> np.ndarray:
    """Stride over a 1-D label trace and return one label per window."""
    out = []
    for start in range(0, len(label_trace) - window + 1, stride):
        seg = label_trace[start : start + window]
        if majority:
            vals, counts = np.unique(seg, return_counts=True)
            out.append(int(vals[np.argmax(counts)]))
        else:
            out.append(int(seg[0]))
    return np.asarray(out, dtype=np.int64)


def _segment(X: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Extract non-overlapping or overlapping windows from (T, C) signal."""
    segs = []
    T = X.shape[0]
    for start in range(0, T - window + 1, stride):
        segs.append(X[start : start + window])
    return np.stack(segs).astype(np.float32) if segs else np.empty((0, window, X.shape[1]), dtype=np.float32)


# ---------------------------------------------------------------------------
# MIT-BIH Arrhythmia Database
# ---------------------------------------------------------------------------

class MitBih:
    """MIT-BIH Arrhythmia Database (mitdb).

    48 half-hour, 2-lead (MLII + V1) ECG recordings at 360 Hz.
    Annotated with ~110 000 beat labels; remapped to 5 AAMI classes.

    Citation
    --------
    Moody GB & Mark RG (2001). The impact of the MIT-BIH Arrhythmia Database.
    IEEE Eng Med Biol 20(3):45-50. https://doi.org/10.13026/C2F305
    """

    FS = 360
    N_CHANNELS = 2
    WINDOW_S = 10       # seconds — standard beat-context window
    # AAMI EC57 5-class mapping from wfdb symbols
    AAMI_MAP = {
        # N — Normal
        "N": 0, ".": 0, "N ": 0,
        # S — Supraventricular ectopic
        "A": 1, "a": 1, "J": 1, "S": 1, "e": 1, "j": 1,
        # V — Ventricular ectopic
        "V": 2, "E": 2,
        # F — Fusion
        "F": 3,
        # Q — Unknown
        "?": 4, "/": 4, "f": 4, "Q": 4,
    }
    CLASSES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    # Hold-out test records per Chazal et al. (2004) inter-patient split
    TEST_RECORDS = {201, 202, 205, 208, 210, 213, 215, 220, 223, 230}
    ALL_RECORDS = {
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        111, 112, 113, 114, 115, 116, 117, 118, 119,
        121, 122, 123, 124,
        200, 201, 202, 203, 205, 207, 208, 209, 210,
        212, 213, 214, 215, 217, 219, 220, 221, 222, 223,
        228, 230, 231, 232, 233, 234,
    }

    @staticmethod
    def download(root: str) -> None:
        """Download MIT-BIH via wfdb (requires `pip install wfdb`)."""
        import wfdb  # type: ignore
        dest = Path(root) / "mitdb"
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MIT-BIH Arrhythmia Database → %s", dest)
        wfdb.dl_database("mitdb", str(dest))
        logger.info("MIT-BIH download complete.")

    @classmethod
    def load(
        cls,
        root: str,
        window_s: float = 10.0,
        stride_s: float = 5.0,
        split: str = "train",
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load windowed MIT-BIH segments.

        Returns
        -------
        X       : (N, T, 2) float32 — ECG windows
        y       : (N,)      int64   — AAMI class per window (majority vote)
        subjects: (N,)      int64   — record number for each window
        metas   : list[dict]
        """
        import wfdb  # type: ignore
        db_path = Path(root) / "mitdb"
        _require(db_path, "MIT-BIH data not found. Run MitBih.download(root).")

        window = int(window_s * cls.FS)
        stride = int(stride_s * cls.FS)
        records = cls.TEST_RECORDS if split == "test" else (cls.ALL_RECORDS - cls.TEST_RECORDS)

        X_all, y_all, subj_all, meta_all = [], [], [], []
        for rec in sorted(records):
            path = str(db_path / str(rec))
            try:
                record = wfdb.rdrecord(path)
                ann = wfdb.rdann(path, "atr")
            except Exception:
                continue
            sig = record.p_signal.astype(np.float32)          # (T, 2)
            if normalize:
                sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
            # Build sample-level label trace (default N)
            label_trace = np.zeros(sig.shape[0], dtype=np.int64)
            for sample, sym in zip(ann.sample, ann.symbol):
                if sym in cls.AAMI_MAP and 0 <= sample < sig.shape[0]:
                    label_trace[sample] = cls.AAMI_MAP[sym]
            segs = _segment(sig, window, stride)
            labs = _window_labels(label_trace, window, stride)
            subjs = np.full(len(segs), rec, dtype=np.int64)
            metas = [{"record": rec, "fs": cls.FS, "modality": "ecg"} for _ in range(len(segs))]
            X_all.append(segs)
            y_all.append(labs)
            subj_all.append(subjs)
            meta_all.extend(metas)

        return (
            np.concatenate(X_all),
            np.concatenate(y_all),
            np.concatenate(subj_all),
            meta_all,
        )


# ---------------------------------------------------------------------------
# PTB-XL 12-Lead ECG
# ---------------------------------------------------------------------------

class PtbXl:
    """PTB-XL: large publicly available 12-lead ECG dataset.

    21 799 records, 500 Hz (or 100 Hz low-res version), 12 leads, 10 s each.
    Labels: 5 superclass tasks (NORM, MI, STTC, CD, HYP).

    Citation
    --------
    Wagner P et al. (2020). PTB-XL, a large publicly available
    electrocardiography dataset. Sci Data 7, 154.
    https://doi.org/10.13026/6zlx-na29
    """

    FS = 500
    FS_LR = 100          # low-resolution version
    N_CHANNELS = 12
    LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    SUPERCLASSES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}
    # Folds 9-10 reserved for test per original paper
    TEST_FOLDS = {9, 10}
    VAL_FOLDS = {8}

    @staticmethod
    def download(root: str) -> None:
        """Download PTB-XL via wfdb."""
        import wfdb  # type: ignore
        dest = Path(root) / "ptb-xl"
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading PTB-XL → %s  (≈ 1.8 GB)", dest)
        wfdb.dl_database("ptb-xl", str(dest))
        logger.info("PTB-XL download complete.")

    @classmethod
    def load(
        cls,
        root: str,
        split: str = "train",
        use_lr: bool = True,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load PTB-XL records.

        Returns
        -------
        X       : (N, T, 12) float32
        y       : (N, 5)     int64  — multi-hot AAMI superclass labels
        subjects: (N,)       int64  — patient_id
        metas   : list[dict]
        """
        import pandas as pd  # type: ignore
        import wfdb  # type: ignore

        db_path = Path(root) / "ptb-xl"
        _require(db_path / "ptbxl_database.csv", "PTB-XL data not found. Run PtbXl.download(root).")

        db = pd.read_csv(db_path / "ptbxl_database.csv", index_col="ecg_id")
        import ast
        db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

        scp = pd.read_csv(db_path / "scp_statements.csv", index_col=0)
        scp = scp[scp.diagnostic == 1]

        def _superclass(codes: dict) -> np.ndarray:
            label = np.zeros(5, dtype=np.int64)
            for code, conf in codes.items():
                if conf >= 50 and code in scp.index:
                    sc = scp.loc[code].diagnostic_class
                    idx = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}.get(sc, -1)
                    if idx >= 0:
                        label[idx] = 1
            return label

        if split == "test":
            fold_mask = db["strat_fold"].isin(cls.TEST_FOLDS)
        elif split == "val":
            fold_mask = db["strat_fold"].isin(cls.VAL_FOLDS)
        else:
            fold_mask = ~db["strat_fold"].isin(cls.TEST_FOLDS | cls.VAL_FOLDS)

        subset = db[fold_mask]
        fs = cls.FS_LR if use_lr else cls.FS
        sig_key = "filename_lr" if use_lr else "filename_hr"

        X_all, y_all, subj_all, meta_all = [], [], [], []
        for ecg_id, row in subset.iterrows():
            path = str(db_path / row[sig_key])
            try:
                record = wfdb.rdrecord(path)
            except Exception:
                continue
            sig = record.p_signal.astype(np.float32)          # (T, 12)
            if normalize:
                sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
            label = _superclass(row["scp_codes"])
            X_all.append(sig[None])
            y_all.append(label[None])
            subj_all.append(int(row.get("patient_id", ecg_id)))
            meta_all.append({"ecg_id": ecg_id, "fs": fs, "modality": "ecg", "leads": cls.LEADS})

        return (
            np.concatenate(X_all),
            np.concatenate(y_all),
            np.asarray(subj_all, dtype=np.int64),
            meta_all,
        )


# ---------------------------------------------------------------------------
# EEG Motor Movement / Imagery Dataset (EEGMMIDB)
# ---------------------------------------------------------------------------

class EegMmidb:
    """PhysioNet EEG Motor Movement/Imagery Dataset.

    109 subjects, 64-channel EEG, 160 Hz.
    Tasks: baseline eyes-open/closed, real/imagined movements (hands, feet).

    Citation
    --------
    Schalk G et al. (2004). BCI2000: A General-Purpose BCI System.
    IEEE TBME 51(6):1034-1043. https://doi.org/10.13026/C28G6P
    """

    FS = 160
    N_CHANNELS = 64
    N_SUBJECTS = 109
    # Run-to-task mapping
    TASK_RUNS = {
        "motor_execution_LR": [3, 7, 11],   # left(1)/right(2) hand execution
        "motor_imagery_LR":   [4, 8, 12],   # left(1)/right(2) hand imagery
        "motor_execution_HF": [5, 9, 13],   # hands(1)/feet(2) execution
        "motor_imagery_HF":   [6, 10, 14],  # hands(1)/feet(2) imagery
    }
    CLASSES = {0: "rest", 1: "left/hands", 2: "right/feet"}
    # Standard 80/20 subject split
    TEST_SUBJECTS = set(range(88, 110))

    @staticmethod
    def download(root: str) -> None:
        """Download EEGMMIDB via MNE (requires `pip install mne`)."""
        import mne  # type: ignore
        dest = Path(root) / "eegmmidb"
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading EEGMMIDB subjects 1-%d → %s", EegMmidb.N_SUBJECTS, dest)
        mne.datasets.eegbci.load_data(
            subjects=list(range(1, EegMmidb.N_SUBJECTS + 1)),
            runs=list(range(1, 15)),
            path=str(dest),
            update_path=False,
        )
        logger.info("EEGMMIDB download complete.")

    @classmethod
    def load(
        cls,
        root: str,
        task: str = "motor_imagery_LR",
        window_s: float = 2.0,
        stride_s: float = 1.0,
        split: str = "train",
        bandpass: tuple[float, float] = (4.0, 40.0),
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load windowed EEG segments for a given task.

        Returns
        -------
        X       : (N, T, 64) float32
        y       : (N,)       int64   — 0=rest, 1=left/hands, 2=right/feet
        subjects: (N,)       int64
        metas   : list[dict]
        """
        import mne  # type: ignore
        mne.set_log_level("WARNING")

        db_path = Path(root) / "eegmmidb"
        _require(db_path, "EEGMMIDB data not found. Run EegMmidb.download(root).")

        runs = cls.TASK_RUNS[task]
        subjects = cls.TEST_SUBJECTS if split == "test" else (set(range(1, cls.N_SUBJECTS + 1)) - cls.TEST_SUBJECTS)
        window = int(window_s * cls.FS)
        stride = int(stride_s * cls.FS)

        X_all, y_all, subj_all, meta_all = [], [], [], []
        for subj in sorted(subjects):
            try:
                raw_files = mne.datasets.eegbci.load_data(
                    subjects=[subj], runs=runs, path=str(db_path), update_path=False
                )
                raws = [mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_files]
                raw = mne.concatenate_raws(raws, verbose=False)
            except Exception as exc:
                logger.warning("Skipping subject %d: %s", subj, exc)
                continue

            mne.datasets.eegbci.standardize(raw)
            if bandpass:
                raw.filter(*bandpass, method="iir", verbose=False)

            events, event_id = mne.events_from_annotations(raw, verbose=False)
            sig = raw.get_data().T.astype(np.float32)   # (T, 64)
            if normalize:
                sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)

            # Build label trace: 0=rest, event_id values map to 1/2
            label_trace = np.zeros(sig.shape[0], dtype=np.int64)
            for onset, _dur, eid in events:
                label_trace[onset] = int(eid)

            segs = _segment(sig, window, stride)
            labs = _window_labels(label_trace, window, stride)
            subjs = np.full(len(segs), subj, dtype=np.int64)
            metas = [{"subject": subj, "task": task, "fs": cls.FS, "modality": "eeg"} for _ in range(len(segs))]
            X_all.append(segs)
            y_all.append(labs)
            subj_all.append(subjs)
            meta_all.extend(metas)

        return (
            np.concatenate(X_all),
            np.concatenate(y_all),
            np.concatenate(subj_all),
            meta_all,
        )


# ---------------------------------------------------------------------------
# BCI Competition IV Dataset 2a
# ---------------------------------------------------------------------------

class BciIv2a:
    """BCI Competition IV Dataset 2a.

    9 subjects, 22-channel EEG, 250 Hz, 4-class motor imagery.
    Classes: left hand (0), right hand (1), feet (2), tongue (3).

    Access via MOABB: `pip install moabb`

    Citation
    --------
    Brunner C et al. (2008). BCI Competition 2008 – Graz data sets A and B.
    https://www.bbci.de/competition/iv/
    """

    FS = 250
    N_CHANNELS = 22
    N_SUBJECTS = 9
    CLASSES = {0: "left_hand", 1: "right_hand", 2: "feet", 3: "tongue"}
    TEST_SUBJECTS = {7, 8, 9}

    @staticmethod
    def download(root: str) -> None:
        """Download BCI-IV-2a via MOABB (requires `pip install moabb`)."""
        from moabb.datasets import BNCI2014_001  # type: ignore
        dest = Path(root) / "bcii v2a"
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading BCI-IV-2a via MOABB → %s", dest)
        dataset = BNCI2014_001()
        dataset.download(path=str(dest))
        logger.info("BCI-IV-2a download complete.")

    @classmethod
    def load(
        cls,
        root: str,
        window_s: float = 2.0,
        stride_s: float = 1.0,
        split: str = "train",
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load windowed BCI-IV-2a trials via MOABB.

        Returns
        -------
        X       : (N, T, 22) float32
        y       : (N,)       int64
        subjects: (N,)       int64
        metas   : list[dict]
        """
        from moabb.datasets import BNCI2014_001  # type: ignore
        import mne  # type: ignore
        mne.set_log_level("WARNING")

        dataset = BNCI2014_001()
        subjects = cls.TEST_SUBJECTS if split == "test" else (set(range(1, cls.N_SUBJECTS + 1)) - cls.TEST_SUBJECTS)
        window = int(window_s * cls.FS)
        stride = int(stride_s * cls.FS)

        X_all, y_all, subj_all, meta_all = [], [], [], []
        label_map = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}

        for subj in sorted(subjects):
            try:
                data = dataset.get_data(subjects=[subj])
            except Exception as exc:
                logger.warning("Skipping BCI subject %d: %s", subj, exc)
                continue
            for sess_data in data[subj].values():
                for run_raw in sess_data.values():
                    events, event_id = mne.events_from_annotations(run_raw, verbose=False)
                    inv = {v: k for k, v in event_id.items()}
                    sig = run_raw.get_data().T.astype(np.float32)[:, :cls.N_CHANNELS]
                    if normalize:
                        sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
                    label_trace = np.zeros(sig.shape[0], dtype=np.int64)
                    for onset, _dur, eid in events:
                        name = inv.get(eid, "")
                        if name in label_map:
                            label_trace[onset] = label_map[name]
                    segs = _segment(sig, window, stride)
                    labs = _window_labels(label_trace, window, stride)
                    subjs = np.full(len(segs), subj, dtype=np.int64)
                    metas = [{"subject": subj, "fs": cls.FS, "modality": "eeg"} for _ in range(len(segs))]
                    X_all.append(segs)
                    y_all.append(labs)
                    subj_all.append(subjs)
                    meta_all.extend(metas)

        return (
            np.concatenate(X_all),
            np.concatenate(y_all),
            np.concatenate(subj_all),
            meta_all,
        )


# ---------------------------------------------------------------------------
# UCI Human Activity Recognition
# ---------------------------------------------------------------------------

class UciHar:
    """UCI Human Activity Recognition Using Smartphones.

    30 subjects, waist-mounted smartphone, 50 Hz, 6 channels (3-acc + 3-gyro).
    Classes: WALKING(0), WALKING_UPSTAIRS(1), WALKING_DOWNSTAIRS(2),
             SITTING(3), STANDING(4), LAYING(5).

    Citation
    --------
    Anguita D et al. (2013). A Public Domain Dataset for Human Activity Recognition
    Using Smartphones. ESANN. https://archive.ics.uci.edu/dataset/240/
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    FS = 50
    N_CHANNELS = 9   # body-acc (3) + gyro (3) + total-acc (3)
    CLASSES = {0: "WALKING", 1: "WALKING_UPSTAIRS", 2: "WALKING_DOWNSTAIRS",
               3: "SITTING", 4: "STANDING", 5: "LAYING"}
    WINDOW = 128     # fixed 2.56 s windows (already segmented in dataset)

    @staticmethod
    def download(root: str) -> None:
        """Download UCI-HAR (≈ 60 MB)."""
        import urllib.request
        dest = Path(root) / "ucihar"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "ucihar.zip"
        logger.info("Downloading UCI-HAR → %s", dest)
        urllib.request.urlretrieve(UciHar.URL, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(dest))
        logger.info("UCI-HAR download complete.")

    @classmethod
    def _load_split(cls, base: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-segmented train or test split."""
        sub = base / split
        signal_dirs = ["Inertial Signals"]
        channels = [
            "body_acc_x", "body_acc_y", "body_acc_z",
            "body_gyro_x", "body_gyro_y", "body_gyro_z",
            "total_acc_x", "total_acc_y", "total_acc_z",
        ]
        parts = []
        for ch in channels:
            fname = sub / "Inertial Signals" / f"{ch}_{split}.txt"
            if fname.exists():
                parts.append(np.loadtxt(str(fname)).astype(np.float32))  # (N, 128)
        X = np.stack(parts, axis=-1)  # (N, 128, 9)
        y_path = sub / f"y_{split}.txt"
        y = np.loadtxt(str(y_path)).astype(np.int64) - 1  # 1-indexed → 0-indexed
        return X, y

    @classmethod
    def load(
        cls,
        root: str,
        split: str = "train",
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load UCI-HAR windows.

        Returns
        -------
        X       : (N, 128, 9) float32
        y       : (N,)        int64
        subjects: (N,)        int64   — subject id from subject_{split}.txt
        metas   : list[dict]
        """
        base = Path(root) / "ucihar" / "UCI HAR Dataset"
        _require(base, "UCI-HAR data not found. Run UciHar.download(root).")

        X, y = cls._load_split(base, split)
        subj_path = base / split / f"subject_{split}.txt"
        subj = np.loadtxt(str(subj_path)).astype(np.int64) if subj_path.exists() else np.zeros(len(X), dtype=np.int64)

        if normalize:
            mean = X.mean(axis=(0, 1), keepdims=True)
            std = X.std(axis=(0, 1), keepdims=True)
            X = (X - mean) / (std + 1e-6)

        metas = [{"subject": int(s), "fs": cls.FS, "modality": "imu"} for s in subj]
        return X, y, subj, metas


# ---------------------------------------------------------------------------
# PAMAP2 Physical Activity Monitoring
# ---------------------------------------------------------------------------

class Pamap2:
    """PAMAP2 Physical Activity Monitoring.

    9 subjects, 3 IMU sensors (wrist, chest, ankle), 100 Hz, 36 features.
    18 physical activities (protocol) + optional optional activities.

    Citation
    --------
    Reiss A & Stricker D (2012). Introducing a New Benchmarked Dataset for
    Activity Monitoring. ISWC. https://archive.ics.uci.edu/dataset/231/
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
    FS = 100
    # 36 IMU features per timestep (per raw data files)
    N_CHANNELS = 36
    CLASSES = {
        1: "lying", 2: "sitting", 3: "standing", 4: "walking",
        5: "running", 6: "cycling", 7: "nordic_walking",
        9: "watching_tv", 10: "computer_work", 11: "car_driving",
        12: "ascending_stairs", 13: "descending_stairs",
        16: "vacuum_cleaning", 17: "ironing", 18: "folding_laundry",
        19: "house_cleaning", 20: "playing_soccer", 24: "rope_jumping",
    }
    # Remap to 0-indexed
    LABEL_MAP = {v: i for i, v in enumerate(sorted(CLASSES.keys()))}
    TEST_SUBJECTS = {108, 109}  # subjects 8 and 9 (1-indexed: 101-109)

    @staticmethod
    def download(root: str) -> None:
        """Download PAMAP2 (≈ 400 MB)."""
        import urllib.request
        dest = Path(root) / "pamap2"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "pamap2.zip"
        logger.info("Downloading PAMAP2 → %s", dest)
        urllib.request.urlretrieve(Pamap2.URL, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(dest))
        logger.info("PAMAP2 download complete.")

    @classmethod
    def load(
        cls,
        root: str,
        window_s: float = 2.0,
        stride_s: float = 1.0,
        split: str = "train",
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load windowed PAMAP2 segments (drops transient activity_id=0).

        Returns
        -------
        X       : (N, T, 36) float32
        y       : (N,)       int64   — 0-indexed activity label
        subjects: (N,)       int64
        metas   : list[dict]
        """
        base = Path(root) / "pamap2" / "PAMAP2_Dataset" / "Protocol"
        _require(base, "PAMAP2 data not found. Run Pamap2.download(root).")

        window = int(window_s * cls.FS)
        stride = int(stride_s * cls.FS)
        all_subj = set(range(101, 110))
        subjects = cls.TEST_SUBJECTS if split == "test" else (all_subj - cls.TEST_SUBJECTS)

        X_all, y_all, subj_all, meta_all = [], [], [], []
        for subj in sorted(subjects):
            path = base / f"subject{subj}.dat"
            if not path.exists():
                continue
            data = np.loadtxt(str(path), dtype=np.float32)  # (T, 54)
            act_id = data[:, 1].astype(np.int64)
            # columns 3..38 are the 36 IMU features; skip timestamp(0), actID(1), heartrate(2)
            sig = data[:, 3:39]
            # Replace NaN with column mean
            col_means = np.nanmean(sig, axis=0)
            nan_mask = np.isnan(sig)
            sig[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            if normalize:
                sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
            # Only keep valid activity labels
            valid = act_id != 0
            sig, act_id = sig[valid], act_id[valid]
            segs = _segment(sig, window, stride)
            act_segs = _window_labels(act_id, window, stride)
            # Remap to 0-indexed
            remapped = np.asarray([cls.LABEL_MAP.get(int(a), -1) for a in act_segs], dtype=np.int64)
            valid_segs = remapped >= 0
            segs, remapped = segs[valid_segs], remapped[valid_segs]
            subjs = np.full(len(segs), subj, dtype=np.int64)
            metas = [{"subject": subj, "fs": cls.FS, "modality": "imu"} for _ in range(len(segs))]
            X_all.append(segs)
            y_all.append(remapped)
            subj_all.append(subjs)
            meta_all.extend(metas)

        return (
            np.concatenate(X_all),
            np.concatenate(y_all),
            np.concatenate(subj_all),
            meta_all,
        )


# ---------------------------------------------------------------------------
# WISDM Activity Recognition
# ---------------------------------------------------------------------------

class Wisdm:
    """WISDM (Wireless Sensor Data Mining) Activity Recognition Dataset.

    36 subjects, smartphone accelerometer, 20 Hz, 3 channels (x/y/z).
    Classes: walking(0), jogging(1), upstairs(2), downstairs(3),
             sitting(4), standing(5).

    Citation
    --------
    Kwapisz JR, Weiss GM & Moore SA (2011). Activity Recognition Using
    Cell Phone Accelerometers. SIGKDD Explor 12(2):74-82.
    https://archive.ics.uci.edu/ml/datasets/WISDM+Smartphone+and+Smartwatch+Activity+and+Biometrics+Dataset+
    """

    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00507/wisdm-dataset.zip"
    FS = 20
    N_CHANNELS = 3
    CLASSES = {0: "walking", 1: "jogging", 2: "upstairs", 3: "downstairs", 4: "sitting", 5: "standing"}
    LABEL_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    TEST_SUBJECTS = set(range(28, 37))  # subjects 28–36

    @staticmethod
    def download(root: str) -> None:
        """Download WISDM (≈ 20 MB)."""
        import urllib.request
        dest = Path(root) / "wisdm"
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / "wisdm.zip"
        logger.info("Downloading WISDM → %s", dest)
        urllib.request.urlretrieve(Wisdm.URL, str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(dest))
        logger.info("WISDM download complete.")

    @classmethod
    def load(
        cls,
        root: str,
        window_s: float = 2.0,
        stride_s: float = 1.0,
        split: str = "train",
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Load WISDM windowed segments.

        Returns
        -------
        X       : (N, T, 3) float32
        y       : (N,)      int64
        subjects: (N,)      int64
        metas   : list[dict]
        """
        # Try multiple known paths inside the zip
        base = Path(root) / "wisdm"
        candidates = [
            base / "wisdm-dataset" / "raw" / "phone" / "accel",
            base / "WISDM_ar_v1.1",
        ]
        raw_dir = next((p for p in candidates if p.exists()), None)
        if raw_dir is None:
            raise FileNotFoundError(
                "WISDM data not found. Run Wisdm.download(root). "
                f"Searched: {candidates}"
            )

        window = int(window_s * cls.FS)
        stride = int(stride_s * cls.FS)
        subjects = cls.TEST_SUBJECTS if split == "test" else (set(range(1, 37)) - cls.TEST_SUBJECTS)

        X_all, y_all, subj_all, meta_all = [], [], [], []
        for subj in sorted(subjects):
            # Phone accelerometer files
            fname = raw_dir / f"data_{subj}_accel_phone.txt"
            if not fname.exists():
                # Fallback to v1.1 format
                fname = raw_dir / "WISDM_ar_v1.1_raw.txt"
            if not fname.exists():
                continue
            rows = []
            with open(fname) as f:
                for line in f:
                    line = line.strip().rstrip(";")
                    parts = line.split(",")
                    if len(parts) < 6:
                        continue
                    try:
                        uid = int(parts[0])
                        act = parts[1].strip()
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        if uid == subj and act in cls.LABEL_MAP:
                            rows.append((cls.LABEL_MAP[act], x, y, z))
                    except ValueError:
                        continue
            if not rows:
                continue
            rows_arr = np.asarray(rows, dtype=np.float32)
            sig = rows_arr[:, 1:]                  # (T, 3)
            act_trace = rows_arr[:, 0].astype(np.int64)
            if normalize:
                sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-6)
            segs = _segment(sig, window, stride)
            labs = _window_labels(act_trace, window, stride)
            subjs = np.full(len(segs), subj, dtype=np.int64)
            metas = [{"subject": subj, "fs": cls.FS, "modality": "imu"} for _ in range(len(segs))]
            X_all.append(segs)
            y_all.append(labs)
            subj_all.append(subjs)
            meta_all.extend(metas)

        if not X_all:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64), []
        return (
            np.concatenate(X_all),
            np.concatenate(y_all),
            np.concatenate(subj_all),
            meta_all,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, type] = {
    "mitbih":   MitBih,
    "ptbxl":    PtbXl,
    "eegmmidb": EegMmidb,
    "bciiv2a":  BciIv2a,
    "ucihar":   UciHar,
    "pamap2":   Pamap2,
    "wisdm":    Wisdm,
}


def get_dataset(name: str) -> type:
    """Return dataset class by registry key (case-insensitive)."""
    key = name.lower().replace("-", "").replace("_", "")
    for k, cls in DATASETS.items():
        if k.replace("-", "").replace("_", "") == key:
            return cls
    raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
