"""
main.py  —  Multimodal AI Framework for Early Cardiovascular Disease Detection
================================================================================
Combines the ECG (ECGNet, 1D CNN) and Chest X-ray (DenseNet) models behind a
single desktop app. The user loads one ECG recording and one chest X-ray,
presses one button, and gets AI-predicted findings with confidence for both.

No clinical/ground-truth comparison is shown here — predictions only.

Master's Thesis  ·  Neelam Fatimah

Run:  python main.py
"""

import io
import json
import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

import customtkinter as ctk
from tkinter import filedialog, messagebox

BASE = Path(__file__).parent
FINAL_DATASET_DIR = BASE / "Final_Dataset"

ECG_CHECKPOINT = BASE / "ecg_model" / "best_model.pt"
ECG_THRESH_JSON = BASE / "ecg_model" / "thresholds.json"

XRAY_MODEL_DIR = BASE / "xray_model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# ECG — labels
# ─────────────────────────────────────────────────────────────────────────────
ECG_N_LEADS = 12
ECG_N_SAMPLES = 5000
ECG_LEAD_NAMES = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_SAMPLES_PER_COL = ECG_N_SAMPLES // 4
ECG_WAVE_3x4 = [[0, 3, 6, 9], [1, 5, 7, 10], [2, 4, 8, 11]]

ECG_LABEL_COLS = [
    "ST_Ischemia", "Atrial_Fibrillation", "Left_Bundle_Branch", "Prolonged_QT",
    "Anterior_Infarct", "Inferior_Infarct", "LV_Hypertrophy", "Right_Bundle_Branch",
    "Sinus_Tachycardia",
]

# Only the 7 validated (EXCELLENT/GOOD) conditions are surfaced to the user
ECG_DISPLAY_GROUPS = [
    ("Rhythm", ["Atrial_Fibrillation", "Sinus_Tachycardia"]),
    ("Conduction", ["Left_Bundle_Branch", "Right_Bundle_Branch"]),
    ("Ischaemia / Infarct", ["ST_Ischemia", "Anterior_Infarct"]),
    ("Interval", ["Prolonged_QT"]),
]
ECG_LABEL_DISPLAY = {
    "ST_Ischemia": "ST Ischaemia",
    "Atrial_Fibrillation": "Atrial Fibrillation",
    "Left_Bundle_Branch": "Left Bundle Branch Block",
    "Prolonged_QT": "Prolonged QT Interval",
    "Anterior_Infarct": "Anterior Infarct",
    "Right_Bundle_Branch": "Right Bundle Branch Block",
    "Sinus_Tachycardia": "Sinus Tachycardia",
}

# Fixed colour per condition for the clinical-region highlighting plot
# (render_clinical_regions_plot) — each colour also names the waveform
# region it is drawn over, for reference:
#   ST_Ischemia         red     -> ST segment (QRS offset to T onset)
#   Anterior_Infarct     maroon  -> first third of QRS (Q-wave portion)
#   Left_Bundle_Branch   purple  -> full QRS window
#   Right_Bundle_Branch  teal    -> full QRS window
#   Prolonged_QT         amber   -> QRS onset to T offset (full span)
#   Atrial_Fibrillation  blue    -> expected P-wave window
#   Sinus_Tachycardia    green   -> each full beat (QRS to next QRS)
CONDITION_COLORS = {
    "ST_Ischemia": "#E24B4A",
    "Anterior_Infarct": "#B8452F",
    "Left_Bundle_Branch": "#7B4FA6",
    "Right_Bundle_Branch": "#2E8B8B",
    "Prolonged_QT": "#D89A2E",
    "Atrial_Fibrillation": "#378ADD",
    "Sinus_Tachycardia": "#4C9A4C",
}

DEFAULT_ECG_THRESH = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# X-ray — labels
# ─────────────────────────────────────────────────────────────────────────────
XRAY_DISPLAY_COLS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
XRAY_LABEL_INFO = {
    "Cardiomegaly": "Enlarged heart silhouette",
    "Edema": "Fluid build-up in the lungs",
    "Pleural Effusion": "Fluid around the lungs",
}
XRAY_MEAN = [0.485, 0.456, 0.406]
XRAY_STD = [0.229, 0.224, 0.225]
DEFAULT_XRAY_THRESH = 0.50

XRAY_CONDITION_COLORS = {
    "Cardiomegaly": "#C81E3A",       # red   — cardiac silhouette / CTR
    "Edema": "#2E8B8B",              # teal  — diffuse lung fields
    "Pleural Effusion": "#D89A2E",   # amber — costophrenic angles
}
# Warning colour (RGB, not hex — drawn directly via PIL) used when the AI's
# Cardiomegaly prediction and the independently measured CTR disagree.
CTR_DISAGREEMENT_COLOR = (184, 134, 11)

# Relative left/right mean-intensity difference above which a finding is
# reported as one-sided rather than bilateral (see _lateral_asymmetry).
LATERALITY_THRESHOLD = 0.12


# ═════════════════════════════════════════════════════════════════════════════
# ECG MODEL  (architecture identical to train_ecg.py / main_ecg.py)
# ═════════════════════════════════════════════════════════════════════════════
class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
                                 nn.Linear(mid, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.gap(x).squeeze(-1)).unsqueeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                    nn.BatchNorm1d(out_ch))
                     if in_ch != out_ch or stride != 1 else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.drop(self.se(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))))
        return self.act(h + self.skip(x))


class MultiScaleStem(nn.Module):
    def __init__(self, in_ch=ECG_N_LEADS, out_ch=64):
        super().__init__()
        mid = out_ch // 3

        def branch(k):
            return nn.Sequential(nn.Conv1d(in_ch, mid, k, stride=2, padding=k // 2, bias=False),
                                  nn.BatchNorm1d(mid), nn.ReLU(inplace=True))

        self.b_s = branch(7)
        self.b_m = branch(15)
        self.b_l = branch(31)
        self.proj = nn.Sequential(nn.Conv1d(mid * 3, out_ch, 1, bias=False),
                                   nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.proj(torch.cat([self.b_s(x), self.b_m(x), self.b_l(x)], dim=1)))


class ECGNet(nn.Module):
    def __init__(self, n_leads=ECG_N_LEADS, n_diag=len(ECG_LABEL_COLS)):
        super().__init__()
        dr = 0.08
        self.stem = MultiScaleStem(n_leads, 64)
        self.layer1 = nn.Sequential(ResBlock1D(64, 64, 7, dropout=dr), ResBlock1D(64, 64, 7, dropout=dr),
                                     ResBlock1D(64, 64, 7, dropout=dr), ResBlock1D(64, 64, 7, dropout=dr))
        self.layer2 = nn.Sequential(ResBlock1D(64, 128, 7, stride=2, dropout=dr), ResBlock1D(128, 128, 7, dropout=dr),
                                     ResBlock1D(128, 128, 7, dropout=dr), ResBlock1D(128, 128, 7, dropout=dr))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, 7, stride=2, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr),
                                     ResBlock1D(256, 256, 7, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr),
                                     ResBlock1D(256, 256, 7, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr),
                                     ResBlock1D(256, 256, 7, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr))
        self.layer4 = nn.Sequential(ResBlock1D(256, 512, 7, stride=2, dropout=dr), ResBlock1D(512, 512, 7, dropout=dr),
                                     ResBlock1D(512, 512, 7, dropout=dr), ResBlock1D(512, 512, 7, dropout=dr))
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(0.50),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(256, n_diag))

    def forward(self, x):
        x = self.stem(x)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.head(torch.cat([self.pool_avg(x), self.pool_max(x)], dim=1).squeeze(-1))


def read_ecg(hea_path: Path) -> np.ndarray:
    with open(hea_path) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    p = lines[0].split()
    nl = int(p[1])
    ns = int(p[3])
    gains, baselines = [], []
    for line in lines[1: 1 + nl]:
        tok = line.split()
        gt = tok[2]
        gains.append(float(re.split(r"[(/]", gt)[0]))
        baselines.append(int(gt.split("(")[1].split(")")[0]) if "(" in gt else 0)
    raw = np.fromfile(str(hea_path.with_suffix(".dat")), dtype=np.int16)
    raw = raw[: ns * nl].reshape(ns, nl)
    sig = np.zeros((nl, ns), dtype=np.float32)
    for i in range(nl):
        g = gains[i] if gains[i] != 0 else 1.0
        sig[i] = (raw[:, i].astype(np.float32) - baselines[i]) / g
    return sig


def normalize_ecg(sig: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sig)
    for i in range(len(sig)):
        s = sig[i].copy()
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        if iqr > 1e-6:
            s = np.clip(s, q25 - 5 * iqr, q75 + 5 * iqr)
        std = s.std()
        if std > 1e-6:
            out[i] = (s - s.mean()) / std
    return out


def pad_crop_ecg(sig: np.ndarray, length: int = ECG_N_SAMPLES) -> np.ndarray:
    n = sig.shape[1]
    if n >= length:
        return sig[:, :length]
    return np.concatenate([sig, np.zeros((sig.shape[0], length - n), dtype=np.float32)], axis=1)


def load_ecg_model():
    if not ECG_CHECKPOINT.exists():
        return None, None, None
    ck = torch.load(ECG_CHECKPOINT, map_location=DEVICE, weights_only=False)
    cols = ck.get("label_cols", ECG_LABEL_COLS)
    model = ECGNet(n_leads=ck.get("n_leads", ECG_N_LEADS), n_diag=len(cols)).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    if ECG_THRESH_JSON.exists():
        d = json.loads(ECG_THRESH_JSON.read_text())
        thresholds = [d.get(c, DEFAULT_ECG_THRESH) for c in cols]
    else:
        thresholds = ck.get("thresholds") or [DEFAULT_ECG_THRESH] * len(cols)
    return model, thresholds, ck


@torch.no_grad()
def run_ecg_inference(model, sig, thresholds, ck):
    s = pad_crop_ecg(normalize_ecg(sig))
    x = torch.from_numpy(s).unsqueeze(0).to(DEVICE)
    pr = torch.sigmoid(model(x)).cpu().numpy()[0]
    cols = (ck or {}).get("label_cols", ECG_LABEL_COLS)
    thr = thresholds or [DEFAULT_ECG_THRESH] * len(cols)
    preds = {c: bool(pr[i] >= thr[i]) for i, c in enumerate(cols)}
    confs = {c: float(pr[i]) for i, c in enumerate(cols)}
    return preds, confs


def _style_ecg_axis(ax, name):
    """Shared clinical look for every lead panel: pink-tinted ECG-paper grid,
    dark-red lead label — used by both the plain and region-highlighted plots."""
    ax.set_facecolor("#FFF5F6")
    for sp in ax.spines.values():
        sp.set_color("#F3C4CC")
        sp.set_linewidth(0.6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(0.015, 0.92, name, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color="#7A0C1E", va="top", ha="left", zorder=5)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.04))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.grid(True, which="minor", color="#FBDCE0", lw=0.25, zorder=0)
    ax.grid(True, which="major", color="#F3B9C2", lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def render_ecg_plot(sig: np.ndarray) -> Image.Image:
    """12-lead grid, rendered headless, returned as a PIL Image."""
    fig = plt.figure(figsize=(11, 6.4), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.55],
                  hspace=0.05, wspace=0.02, left=0.01, right=0.99, top=0.96, bottom=0.02)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]
    rhythm_ax = fig.add_subplot(gs[3, :])

    DS = 2
    t_c = np.arange(0, ECG_SAMPLES_PER_COL, DS) / 500.0
    for r in range(3):
        for c in range(4):
            ax = axes[r][c]
            lead = ECG_WAVE_3x4[r][c]
            y = sig[lead, c * ECG_SAMPLES_PER_COL:(c + 1) * ECG_SAMPLES_PER_COL:DS].copy()
            std = y.std()
            if std > 1e-6:
                y = np.clip(y / std, -2.8, 2.8)
            _style_ecg_axis(ax, ECG_LEAD_NAMES[lead])
            ax.plot(t_c, y, color="#C81E3A", lw=0.9, zorder=2)
            ax.set_xlim(0, t_c[-1])
            ax.set_ylim(-3.2, 3.2)

    DS_r = 4
    t_r = np.arange(0, ECG_N_SAMPLES, DS_r) / 500.0
    y = sig[1, ::DS_r].copy()
    std = y.std()
    if std > 1e-6:
        y = np.clip(y / std, -2.8, 2.8)
    _style_ecg_axis(rhythm_ax, "II — Rhythm Strip (10s)")
    rhythm_ax.plot(t_r, y, color="#C81E3A", lw=0.9, zorder=2)
    rhythm_ax.set_xlim(0, t_r[-1])
    rhythm_ax.set_ylim(-3.2, 3.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ═════════════════════════════════════════════════════════════════════════════
# CLINICAL WAVE SEGMENTATION  —  independent, non-AI wave delineation via
# neurokit2 (classical signal processing, not a learned model), used to
# locate the medically expected region for each detected condition. This is
# NOT expert-annotated ground truth (neurokit2 is itself an algorithmic
# detector, not a cardiologist), but it is a deterministic, independent
# reference — see `render_clinical_regions_plot`.
# ═════════════════════════════════════════════════════════════════════════════
def _clean_wave_index(v):
    """neurokit2 returns a mix of int, numpy int64, and float('nan') for
    landmarks it could not detect — normalise to a plain int or None."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def segment_ecg_waves(signal_lead2: np.ndarray, fs: int = 500) -> list:
    """
    Delineates P, QRS, and T wave boundaries for every beat in a single-lead
    ECG strip using neurokit2 (classical signal processing, not a learned
    model), to serve as an independent reference for
    `expected_windows_for_condition` / `render_clinical_regions_plot`.

    signal_lead2: 1D array, raw (un-normalised) Lead II signal in mV.
    fs:           sampling rate in Hz.

    Returns a list of per-beat dicts (sample indices, or None where a
    landmark could not be cleanly detected for that beat):
        p_onset, p_offset,
        qrs_onset, qrs_offset, q_peak, s_peak,
        t_onset, t_offset,
        st_onset, st_offset   (= qrs_offset, t_onset; None if t_onset missing)
    Beats where neurokit2 cannot resolve a valid QRS onset/offset are
    skipped entirely (logged, not raised) since every other window in this
    module is anchored to the QRS complex.
    """
    import neurokit2 as nk

    try:
        cleaned = nk.ecg_clean(signal_lead2, sampling_rate=fs)
        _, rpeaks_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        if len(rpeaks_info.get("ECG_R_Peaks", [])) == 0:
            print("segment_ecg_waves: no R-peaks detected in this strip, skipping")
            return []
        _, waves_info = nk.ecg_delineate(cleaned, rpeaks_info, sampling_rate=fs, method="dwt")
    except Exception as exc:
        print(f"segment_ecg_waves: neurokit2 delineation failed ({exc}), skipping strip")
        return []

    n_beats = len(waves_info.get("ECG_R_Onsets", []))
    beats, skipped = [], 0

    for i in range(n_beats):
        qrs_onset = _clean_wave_index(waves_info["ECG_R_Onsets"][i])
        qrs_offset = _clean_wave_index(waves_info["ECG_R_Offsets"][i])
        if qrs_onset is None or qrs_offset is None or qrs_offset <= qrs_onset:
            skipped += 1
            continue

        p_onset = _clean_wave_index(waves_info["ECG_P_Onsets"][i])
        p_offset = _clean_wave_index(waves_info["ECG_P_Offsets"][i])
        q_peak = _clean_wave_index(waves_info["ECG_Q_Peaks"][i])
        s_peak = _clean_wave_index(waves_info["ECG_S_Peaks"][i])
        t_onset = _clean_wave_index(waves_info["ECG_T_Onsets"][i])
        t_offset = _clean_wave_index(waves_info["ECG_T_Offsets"][i])

        st_onset, st_offset = (qrs_offset, t_onset) if (t_onset is not None and t_onset > qrs_offset) \
            else (None, None)

        beats.append({
            "p_onset": p_onset, "p_offset": p_offset,
            "qrs_onset": qrs_onset, "qrs_offset": qrs_offset,
            "q_peak": q_peak, "s_peak": s_peak,
            "t_onset": t_onset, "t_offset": t_offset,
            "st_onset": st_onset, "st_offset": st_offset,
        })

    if skipped:
        print(f"segment_ecg_waves: skipped {skipped}/{n_beats} beat(s) with incomplete QRS delineation")

    return beats


def expected_windows_for_condition(condition_name: str, wave_segments: list) -> list:
    """
    Returns the list of (start, end) sample-index windows medically expected
    for `condition_name`, given per-beat `wave_segments` from
    `segment_ecg_waves`. The single source of truth for what
    `render_clinical_regions_plot` shades on the waveform.

    Sinus_Tachycardia is rate-based rather than tied to a fixed sub-segment,
    so its "window" is each full beat — QRS onset to the next beat's QRS
    onset (the RR interval) — rather than a portion of one beat.
    """
    if condition_name == "Sinus_Tachycardia":
        windows = []
        for a, b in zip(wave_segments, wave_segments[1:]):
            if a.get("qrs_onset") is not None and b.get("qrs_onset") is not None:
                windows.append((a["qrs_onset"], b["qrs_onset"]))
        return windows

    windows = []
    for b in wave_segments:
        if condition_name == "ST_Ischemia":
            if b.get("st_onset") is not None and b.get("st_offset") is not None:
                windows.append((b["st_onset"], b["st_offset"]))
        elif condition_name in ("Left_Bundle_Branch", "Right_Bundle_Branch"):
            if b.get("qrs_onset") is not None and b.get("qrs_offset") is not None:
                windows.append((b["qrs_onset"], b["qrs_offset"]))
        elif condition_name == "Prolonged_QT":
            if b.get("qrs_onset") is not None and b.get("t_offset") is not None:
                windows.append((b["qrs_onset"], b["t_offset"]))
        elif condition_name == "Anterior_Infarct":
            # Q-wave portion specifically: the first third of the QRS complex.
            if b.get("qrs_onset") is not None and b.get("qrs_offset") is not None:
                third = max(1, (b["qrs_offset"] - b["qrs_onset"]) // 3)
                windows.append((b["qrs_onset"], b["qrs_onset"] + third))
        elif condition_name == "Atrial_Fibrillation":
            if b.get("p_onset") is not None and b.get("p_offset") is not None:
                windows.append((b["p_onset"], b["p_offset"]))
    return windows


def _read_lead_with_trailing_context(hea_path: Path, lead_idx: int, base_samples: int,
                                      extra_samples: int = 750) -> np.ndarray:
    """
    Reads one lead with up to `extra_samples` extra samples appended beyond
    `base_samples` (the displayed window), so neurokit2's delineator gets
    enough lookahead to confidently close out the last beat — it needs to
    see the start of a following beat to finalise a QRS/T-wave offset,
    which the last beat in an exact-length window never has.

    Uses genuine extra samples from the .dat file when the file actually
    contains more than `base_samples` (some WFDB sources do). For this
    project's Final_Dataset, .dat files are sized to exactly match the
    header — verified: no extra real samples exist — so in practice this
    falls back to reflect-padding the tail of the loaded signal instead.
    Reflect-padding is a standard technique for giving an edge-detection
    algorithm local lookahead context without fabricating a fake beat; it
    is not real trailing signal, which is exactly why every boundary it
    might produce beyond `base_samples` is discarded by
    `_clip_beat_to_window` before use — the padding only helps delineate
    the *last real beat*, it is never itself treated as data.
    """
    with open(hea_path) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    p = lines[0].split()
    nl = int(p[1])
    gains, baselines = [], []
    for line in lines[1: 1 + nl]:
        tok = line.split()
        gt = tok[2]
        gains.append(float(re.split(r"[(/]", gt)[0]))
        baselines.append(int(gt.split("(")[1].split(")")[0]) if "(" in gt else 0)

    dat_path = Path(hea_path).with_suffix(".dat")
    available = dat_path.stat().st_size // (2 * nl)  # int16 = 2 bytes/sample/lead
    want = base_samples + extra_samples
    read_n = max(1, min(want, available))

    raw = np.fromfile(str(dat_path), dtype=np.int16, count=read_n * nl).reshape(-1, nl)
    g = gains[lead_idx] if gains[lead_idx] != 0 else 1.0
    lead = (raw[:, lead_idx].astype(np.float32) - baselines[lead_idx]) / g

    if len(lead) < want and len(lead) > 1:
        lead = np.pad(lead, (0, want - len(lead)), mode="reflect")
    return lead


def _clip_beat_to_window(beat: dict, max_sample: int) -> dict:
    """Discards (sets to None) any wave boundary at or beyond `max_sample` —
    used after delineating a signal extended with trailing context beyond
    the window actually being displayed/analysed, so nothing derived from
    the extra context (real or reflect-padded) leaks into the result."""
    return {k: (v if (v is None or v < max_sample) else None) for k, v in beat.items()}


def render_clinical_regions_plot(sig: np.ndarray, detected_keys: list, hea_path: Path = None) -> Image.Image:
    """
    Same 12-lead clinical grid as `render_ecg_plot`, with the medically
    expected region for each *detected* condition shaded in that
    condition's own fixed colour (CONDITION_COLORS). The region for every
    condition comes solely from `segment_ecg_waves` (neurokit2 wave
    delineation, classical signal processing) via
    `expected_windows_for_condition` — no model attention is
    involved in this plot.

    Where two or more detected conditions' regions overlap in time, the
    axvspan calls are simply drawn one after another and matplotlib
    alpha-blends them naturally into a mixed tone — no manual colour
    splitting.

    sig:            (12, N) raw (un-normalised) signal — this is what gets
                    displayed; N is the analysis window (5,000 samples).
    detected_keys:  condition keys (ECG_LABEL_COLS names) flagged DETECTED
                    for this ECG.
    hea_path:       optional path to the source .hea file. When given, wave
                    segmentation runs on Lead II extended with trailing
                    context beyond `sig`'s window (see
                    `_read_lead_with_trailing_context`), so the last beat
                    near the end of the strip still gets delineated instead
                    of being left blank; any boundary at or beyond the
                    original window is then discarded
                    (`_clip_beat_to_window`) before it is used. If omitted,
                    segmentation runs on `sig` alone as before.
    """
    n_samples = sig.shape[1]
    if hea_path is not None:
        try:
            extended_lead_ii = _read_lead_with_trailing_context(hea_path, lead_idx=1, base_samples=n_samples)
            wave_segments = segment_ecg_waves(extended_lead_ii, fs=500)
            wave_segments = [_clip_beat_to_window(b, n_samples) for b in wave_segments]
        except Exception as exc:
            print(f"render_clinical_regions_plot: extended-context read failed ({exc}), "
                  f"falling back to the displayed window only")
            wave_segments = segment_ecg_waves(sig[1], fs=500)
    else:
        wave_segments = segment_ecg_waves(sig[1], fs=500)  # Lead II

    condition_windows = []  # list of (key, color, windows)
    if wave_segments:
        for key in detected_keys:
            windows = expected_windows_for_condition(key, wave_segments)
            if not windows:
                print(f"render_clinical_regions_plot: no clean region found for {key}, skipping")
                continue
            condition_windows.append((key, CONDITION_COLORS.get(key, "#888888"), windows))
    elif detected_keys:
        print("render_clinical_regions_plot: no beats delineated, showing plain waveform")

    fig = plt.figure(figsize=(11, 6.4), dpi=140)
    fig.patch.set_facecolor("#FFFFFF")
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.55],
                  hspace=0.05, wspace=0.02, left=0.01, right=0.99, top=0.94, bottom=0.10)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(3)]
    rhythm_ax = fig.add_subplot(gs[3, :])

    def mark_panel(ax, seg_start, seg_end, fs=500.0):
        for _, color, windows in condition_windows:
            for s, e in windows:
                cs, ce = max(s, seg_start), min(e, seg_end)
                if ce <= cs:
                    continue
                ax.axvspan((cs - seg_start) / fs, (ce - seg_start) / fs,
                           color=color, alpha=0.35, zorder=1, lw=0)

    DS = 2
    t_c = np.arange(0, ECG_SAMPLES_PER_COL, DS) / 500.0
    for r in range(3):
        for c in range(4):
            ax = axes[r][c]
            lead = ECG_WAVE_3x4[r][c]
            seg_start, seg_end = c * ECG_SAMPLES_PER_COL, (c + 1) * ECG_SAMPLES_PER_COL
            y = sig[lead, seg_start:seg_end:DS].copy()
            std = y.std()
            if std > 1e-6:
                y = np.clip(y / std, -2.8, 2.8)
            _style_ecg_axis(ax, ECG_LEAD_NAMES[lead])
            mark_panel(ax, seg_start, seg_end)
            ax.plot(t_c, y, color="#C81E3A", lw=0.9, zorder=3)
            ax.set_xlim(0, t_c[-1])
            ax.set_ylim(-3.2, 3.2)

    DS_r = 4
    t_r = np.arange(0, ECG_N_SAMPLES, DS_r) / 500.0
    y_r = sig[1, ::DS_r].copy()
    std_r = y_r.std()
    if std_r > 1e-6:
        y_r = np.clip(y_r / std_r, -2.8, 2.8)
    _style_ecg_axis(rhythm_ax, "II — Rhythm Strip (10s)")
    mark_panel(rhythm_ax, 0, ECG_N_SAMPLES)
    rhythm_ax.plot(t_r, y_r, color="#C81E3A", lw=0.9, zorder=3)
    rhythm_ax.set_xlim(0, t_r[-1])
    rhythm_ax.set_ylim(-3.2, 3.2)

    if condition_windows:
        handles = [Patch(facecolor=color, alpha=0.6, edgecolor="none",
                          label=ECG_LABEL_DISPLAY.get(key, key))
                   for key, color, _ in condition_windows]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ═════════════════════════════════════════════════════════════════════════════
# X-RAY MODEL  (architecture identical to xray_train.py / xray_main.py)
# ═════════════════════════════════════════════════════════════════════════════
class XrayNet(nn.Module):
    def __init__(self, backbone_name="densenet121-res224-all", num_classes=len(XRAY_DISPLAY_COLS)):
        super().__init__()
        self._is_densenet = "densenet" in backbone_name.lower()

        if self._is_densenet:
            import torchxrayvision as txrv
            self.backbone = txrv.models.DenseNet(weights=backbone_name)
            in_feat = 1024
        elif backbone_name == "efficientnet_b4":
            from torchvision.models import efficientnet_b4
            self.backbone = efficientnet_b4(weights=None)
            in_feat = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(weights=None)
            in_feat = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(in_feat, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if self._is_densenet:
            feat_map = self.backbone.features(x)
            feat_map = F.relu(feat_map, inplace=True)
            pooled = F.adaptive_avg_pool2d(feat_map, (1, 1))
            feats = pooled.view(pooled.size(0), -1)
        else:
            feats = self.backbone(x)
        return self.head(feats)


def preprocess_xray(img: Image.Image, img_size=224, backbone_name="densenet121-res224-all"):
    if "densenet" in backbone_name.lower():
        arr = np.array(img.convert("L").resize((img_size, img_size)), dtype=np.float32)
        arr = (arr / 255.0) * 2048.0 - 1024.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(XRAY_MEAN, XRAY_STD),
    ])(img.convert("RGB")).unsqueeze(0)


def load_xray_model():
    best_pt = XRAY_MODEL_DIR / "best_model.pt"
    if not best_pt.exists():
        return None, {}, 224, "densenet121-res224-all"

    state = torch.load(best_pt, map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    cfg_path = XRAY_MODEL_DIR / "config.json"
    backbone, img_size = "densenet121-res224-all", 224
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        backbone = cfg.get("backbone", backbone)
        img_size = cfg.get("img_size", img_size)

    label_cols = XRAY_DISPLAY_COLS
    lbl_path = XRAY_MODEL_DIR / "label_classes.json"
    if lbl_path.exists():
        label_cols = json.loads(lbl_path.read_text(encoding="utf-8"))

    thresholds = {}
    thr_path = XRAY_MODEL_DIR / "thresholds.json"
    if thr_path.exists():
        thresholds = json.loads(thr_path.read_text(encoding="utf-8"))

    model = XrayNet(backbone_name=backbone, num_classes=len(label_cols))
    model.load_state_dict(state)
    model.eval()
    return model, thresholds, img_size, backbone


def xray_thumbnail(img: Image.Image, max_dim: int = 460) -> Image.Image:
    im = img.convert("L").copy()
    im.thumbnail((max_dim, max_dim))
    return im.convert("RGB")


# ═════════════════════════════════════════════════════════════════════════════
# X-RAY CLINICAL REGION HIGHLIGHTING  —  independent, non-AI image geometry
# (Otsu thresholding + connected-component analysis via scipy/scikit-image),
# used to locate the medically expected region or measurement for each
# *detected* condition. This does not use the classifier's weights,
# gradients, or attention in any way — it is pure post-hoc image processing
# run on the pixel data after the model's decision is already made.
# ═════════════════════════════════════════════════════════════════════════════
def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Keeps only the largest 4-connected True region of a boolean mask."""
    from scipy import ndimage as ndi
    labeled, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, labeled, index=np.arange(1, n + 1))
    largest = int(np.argmax(sizes)) + 1
    return labeled == largest


def _fill_and_clean(mask: np.ndarray):
    from scipy import ndimage as ndi
    mask = _largest_component_mask(mask)
    if not mask.any():
        return None
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, structure=np.ones((9, 9)))
    return mask


def segment_thoracic_cavity(gray: np.ndarray):
    """
    Segments the outer thoracic (body) silhouette from a grayscale chest
    X-ray using Otsu thresholding: separates the patient's body from the
    dark background outside it, keeps the largest connected foreground
    region, and fills internal holes so the result is one solid cavity
    mask. Classical image processing only — no learned model.
    """
    from skimage.filters import threshold_otsu

    h, w = gray.shape
    if gray.max() <= gray.min():
        return None
    try:
        mask = _fill_and_clean(gray > threshold_otsu(gray))
    except ValueError:
        mask = None

    # A plain Otsu split sometimes separates bone/mediastinum from lung
    # instead of body from background, leaving a mask far too small to be
    # a thoracic outline — fall back to a background-relative cut using the
    # image corners (typically pure background) as the reference level.
    if mask is None or mask.sum() < 0.12 * h * w:
        c = max(4, min(h, w) // 20)
        bg_patches = np.concatenate([
            gray[:c, :c].ravel(), gray[:c, -c:].ravel(),
            gray[-c:, :c].ravel(), gray[-c:, -c:].ravel(),
        ])
        bg_level = float(np.median(bg_patches))
        margin = 0.06 * (gray.max() - gray.min())
        mask = _fill_and_clean(gray > (bg_level + margin))
    return mask


def segment_cardiac_silhouette(gray: np.ndarray, thoracic_mask: np.ndarray):
    """
    Segments the cardiac silhouette using intensity-based contour detection
    restricted to the central-lower thoracic region (where the heart sits
    between the lungs, above the diaphragm) — an Otsu split of that region
    only, independent of the classifier.
    """
    from skimage.filters import threshold_otsu

    ys, xs = np.where(thoracic_mask)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None

    rx0, rx1 = int(x0 + 0.20 * w), int(x0 + 0.80 * w)
    ry0, ry1 = int(y0 + 0.30 * h), int(y0 + 0.78 * h)

    roi_mask = np.zeros_like(thoracic_mask)
    roi_mask[ry0:ry1, rx0:rx1] = True
    roi_mask &= thoracic_mask
    if roi_mask.sum() < 40:
        return None

    try:
        heart_thresh = threshold_otsu(gray[roi_mask])
    except ValueError:
        return None

    heart_mask = np.zeros_like(thoracic_mask)
    heart_mask[roi_mask] = gray[roi_mask] > heart_thresh
    heart_mask = _largest_component_mask(heart_mask)
    if not heart_mask.any() or heart_mask.sum() < 40:
        return None
    from scipy import ndimage as ndi
    return ndi.binary_fill_holes(heart_mask)


def _max_row_width(mask: np.ndarray, rows=None):
    """Returns (row, left, right, width) for the widest True run of any row."""
    rows = range(mask.shape[0]) if rows is None else rows
    best = None
    for r in rows:
        cols = np.where(mask[r])[0]
        if len(cols) == 0:
            continue
        left, right = int(cols.min()), int(cols.max())
        width = right - left
        if best is None or width > best[3]:
            best = (r, left, right, width)
    return best


def _hex_to_rgba(hex_color: str, alpha: int) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def _paint_mask(overlay: Image.Image, mask: np.ndarray, fill_rgba: tuple, edge_rgba: tuple = None):
    """Stencils `mask` (bool array) onto `overlay` with `fill_rgba`, so the
    painted region follows the exact segmented shape rather than a bounding
    box. `edge_rgba`, if given, is painted onto the mask's boundary ring."""
    if not mask.any():
        return
    from scipy import ndimage as ndi
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    fill_layer = Image.new("RGBA", overlay.size, fill_rgba)
    overlay.paste(fill_layer, (0, 0), mask_img)
    if edge_rgba is not None:
        edge = mask & ~ndi.binary_erosion(mask, iterations=2)
        if edge.any():
            edge_img = Image.fromarray((edge * 255).astype(np.uint8), mode="L")
            edge_layer = Image.new("RGBA", overlay.size, edge_rgba)
            overlay.paste(edge_layer, (0, 0), edge_img)


def _rect_mask(shape: tuple, rect: list) -> np.ndarray:
    """Boolean mask that is True inside axis-aligned [x0, y0, x1, y1] `rect`."""
    m = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = [int(v) for v in rect]
    h, w = shape
    m[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return m


def _lateral_asymmetry(gray: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray):
    """
    Compares mean pixel intensity between a left-side and right-side mask
    (lung field or costophrenic angle). Fluid/consolidation attenuates more
    X-ray and so appears brighter (denser) than normal aerated lung — the
    side with a meaningfully higher mean intensity is reported as more
    affected. Pure pixel statistics, independent of the classifier.

    Returns (side, left_mean, right_mean): side is "left" or "right" when
    one side's mean is more than LATERALITY_THRESHOLD (relative) brighter
    than the other, otherwise "bilateral". Means are None if either mask is
    empty (asymmetry can't be judged, so "bilateral" is returned).
    """
    if not left_mask.any() or not right_mask.any():
        return "bilateral", None, None
    left_mean = float(gray[left_mask].mean())
    right_mean = float(gray[right_mask].mean())
    denom = max(left_mean, right_mean, 1e-6)
    diff_ratio = abs(left_mean - right_mean) / denom
    if diff_ratio > LATERALITY_THRESHOLD:
        side = "left" if left_mean > right_mean else "right"
    else:
        side = "bilateral"
    return side, left_mean, right_mean


def render_clinical_regions_xray(image: Image.Image, detected: list, max_dim: int = 460) -> Image.Image:
    """
    Draws independent, non-AI overlays on the chest X-ray for whichever of
    the 3 conditions are actually DETECTED for this patient (never all 3 —
    same "only what's detected" pattern as the ECG legend):

      - Edema             : labelled region hugging the segmented lung field
                             boundary. Left/right mean pixel intensity is
                             compared (`_lateral_asymmetry`) — the denser
                             (brighter) side is highlighted at full strength
                             and labelled "predominantly left/right"; if both
                             sides are similarly dense it's labelled
                             "bilateral" and both are shown at full strength.
      - Pleural Effusion  : same left/right density comparison, applied to
                             the costophrenic-angle corner boxes.
      - Cardiomegaly      : measured Cardiothoracic Ratio (CTR), with the
                             heart-width and thoracic-width caliper lines —
                             drawn last, on top of every other overlay, so
                             they are always visible.

    All regions and the left/right comparison come from Otsu thresholding +
    pixel-intensity statistics on the image itself (`segment_thoracic_cavity`
    / `segment_cardiac_silhouette` / `_lateral_asymmetry`) — not from the
    classifier's predictions, weights, or gradients. The model is used only
    beforehand, to decide which conditions are detected.
    """
    base = image.convert("L").copy()
    base.thumbnail((max_dim, max_dim))
    gray = np.array(base, dtype=np.float32)
    h, w = gray.shape
    rgb = Image.merge("RGB", (base, base, base))

    if not detected:
        return rgb

    thoracic_mask = segment_thoracic_cavity(gray)
    if thoracic_mask is None or not thoracic_mask.any():
        return rgb  # segmentation failed on this image — nothing reliable to draw

    ys, xs = np.where(thoracic_mask)
    tx0, tx1, ty0, ty1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    tw, th = tx1 - tx0, ty1 - ty0
    cx0, cx1 = tx0 + 0.38 * tw, tx0 + 0.62 * tw  # central mediastinal strip, excluded from both lungs

    overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()
    legend_by_condition = {}
    pending_labels = []  # drawn in a final pass, after every fill/line, so text is never crossed out

    def queue_label(xy, text, color_rgb, align="left"):
        pending_labels.append((xy, text, color_rgb, align))

    def draw_label(xy, text, color_rgb, align="left"):
        tb = draw.textbbox((0, 0), text, font=font)
        tw_txt, th_txt = tb[2] - tb[0], tb[3] - tb[1]
        x, y = xy
        if align == "right":
            x -= tw_txt
        pad = 3
        draw.rectangle([x - pad, y - pad, x + tw_txt + pad, y + th_txt + pad], fill=(255, 255, 255, 225))
        draw.text((x, y), text, fill=color_rgb + (255,), font=font)

    # ── Edema: hug the actual segmented lung-field shape, not a fixed box ──
    if "Edema" in detected and tw > 0 and th > 0:
        row_lo, row_hi = int(ty0 + 0.12 * th), int(ty0 + 0.80 * th)  # below shoulders, above diaphragm
        row_band = np.zeros_like(thoracic_mask)
        row_band[row_lo:row_hi, :] = True
        lung_field = thoracic_mask & row_band

        left_lung = lung_field.copy()
        left_lung[:, int(cx0):] = False
        right_lung = lung_field.copy()
        right_lung[:, :int(cx1)] = False
        left_lung = _largest_component_mask(left_lung)
        right_lung = _largest_component_mask(right_lung)

        if left_lung.any() or right_lung.any():
            side, _, _ = _lateral_asymmetry(gray, left_lung, right_lung)
            color = XRAY_CONDITION_COLORS["Edema"]
            full_fill, full_edge = _hex_to_rgba(color, 70), _hex_to_rgba(color, 220)
            faint_fill, faint_edge = _hex_to_rgba(color, 22), _hex_to_rgba(color, 80)

            if side == "bilateral":
                _paint_mask(overlay, left_lung, full_fill, full_edge)
                _paint_mask(overlay, right_lung, full_fill, full_edge)
                label_text, legend_text = "Edema - bilateral", "Bilateral diffuse lung fields (density-based)"
            else:
                strong, faint = (left_lung, right_lung) if side == "left" else (right_lung, left_lung)
                _paint_mask(overlay, strong, full_fill, full_edge)
                _paint_mask(overlay, faint, faint_fill, faint_edge)
                label_text = f"Edema - predominantly {side}"
                legend_text = f"Predominantly {side} (density-based)"

            queue_label((tx0 + 6, row_lo + 4), label_text, (46, 90, 90))
            legend_by_condition["Edema"] = (legend_text, color)

    # ── Pleural Effusion: costophrenic-angle corners, weighted by density ──
    if "Pleural Effusion" in detected and tw > 0 and th > 0:
        by0, by1 = ty0 + 0.80 * th, ty1
        left_rect = [tx0 + 0.04 * tw, by0, tx0 + 0.32 * tw, by1]
        right_rect = [tx1 - 0.32 * tw, by0, tx1 - 0.04 * tw, by1]
        left_pe_mask = _rect_mask(gray.shape, left_rect) & thoracic_mask
        right_pe_mask = _rect_mask(gray.shape, right_rect) & thoracic_mask
        side, _, _ = _lateral_asymmetry(gray, left_pe_mask, right_pe_mask)

        color = XRAY_CONDITION_COLORS["Pleural Effusion"]
        full_fill, full_edge = _hex_to_rgba(color, 70), _hex_to_rgba(color, 220)
        faint_fill, faint_edge = _hex_to_rgba(color, 22), _hex_to_rgba(color, 80)

        if side == "bilateral":
            draw.rectangle(left_rect, fill=full_fill, outline=full_edge, width=2)
            draw.rectangle(right_rect, fill=full_fill, outline=full_edge, width=2)
            label_text = "Pleural Effusion - bilateral"
            legend_text = "Bilateral costophrenic angles (density-based)"
        else:
            strong_rect, faint_rect = (left_rect, right_rect) if side == "left" else (right_rect, left_rect)
            draw.rectangle(strong_rect, fill=full_fill, outline=full_edge, width=2)
            draw.rectangle(faint_rect, fill=faint_fill, outline=faint_edge, width=1)
            label_text = f"Pleural Effusion - {side}"
            legend_text = f"{side.capitalize()} costophrenic angle (density-based)"

        queue_label((left_rect[0] + 4, by0 - 16), label_text, (140, 95, 20))
        legend_by_condition["Pleural Effusion"] = (legend_text, color)

    # ── Cardiomegaly: CTR caliper lines ─────────────────────────────────────
    if "Cardiomegaly" in detected:
        heart_mask = segment_cardiac_silhouette(gray, thoracic_mask)
        heart_best = _max_row_width(heart_mask) if heart_mask is not None else None
        if heart_best is not None:
            hr, hl, hrt, hwid = heart_best
            thor_at_row = _max_row_width(thoracic_mask, rows=[hr])
            if thor_at_row is not None:
                _, tl, trt, twid = thor_at_row
                heart_color = _hex_to_rgba("#FF1E3A", 255)
                thor_color = _hex_to_rgba("#8F001C", 255)
                thor_row = min(h - 1, hr + 12)

                def caliper(y, x0, x1, color, width=3, tick=6):
                    draw.line([(x0, y), (x1, y)], fill=color, width=width)
                    draw.line([(x0, y - tick), (x0, y + tick)], fill=color, width=width)
                    draw.line([(x1, y - tick), (x1, y + tick)], fill=color, width=width)

                caliper(hr, hl, hrt, heart_color)
                caliper(thor_row, tl, trt, thor_color)

                ctr = hwid / twid if twid > 0 else None
                if ctr is not None:
                    disagrees = ctr <= 0.5  # AI flagged Cardiomegaly, but the measured CTR doesn't support it
                    note = "  (>0.50, enlarged)" if ctr > 0.5 else ""
                    # Fixed top-right corner: guaranteed clear of the Edema
                    # (top-left) and Pleural Effusion (bottom-left) labels.
                    queue_label((w - 6, 6), f"CTR: {ctr:.2f}{note}", heart_color[:3], align="right")
                    legend_text = f"CTR {ctr:.2f} - heart / thoracic width"
                    if disagrees:
                        queue_label((w - 6, 24), "AI positive, but CTR is below 0.50",
                                    CTR_DISAGREEMENT_COLOR, align="right")
                        legend_text += "  (below 0.50 - disagrees with AI)"
                    legend_by_condition["Cardiomegaly"] = (legend_text, XRAY_CONDITION_COLORS["Cardiomegaly"])

    # Final pass: every text label on top of every fill/line, so no
    # measurement line or region box ever crosses through readable text.
    for xy, text, color_rgb, align in pending_labels:
        draw_label(xy, text, color_rgb, align)

    legend_entries = [(name, *legend_by_condition[name]) for name in XRAY_DISPLAY_COLS
                       if name in legend_by_condition]

    composited = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")

    if not legend_entries:
        return composited

    legend_h = 20 * len(legend_entries) + 10
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = max((measure.textbbox((0, 0), f"{name} - {desc}", font=font)[2]
                       for name, desc, _ in legend_entries), default=0)
    final_w = max(w, max_text_w + 34)
    final_img = Image.new("RGB", (final_w, h + legend_h), "white")
    final_img.paste(composited, (0, 0))
    ld = ImageDraw.Draw(final_img)
    ly = h + 6
    for name, desc, color in legend_entries:
        ld.rectangle([8, ly + 3, 20, ly + 13], fill=color)
        ld.text((26, ly), f"{name} - {desc}", fill="#3A3A3A", font=font)
        ly += 20
    return final_img


# ═════════════════════════════════════════════════════════════════════════════
# FAIRNESS EVALUATION  —  gender-based AUC gap check on the held-out test set.
#
# X-ray: test-set membership is reconstructed with the exact patient-level
# 70/15/15 split (seed=42) that xray_train.py's make_splits() computes from
# Final_Dataset/ + final_patient_dataset_summary.xlsx — the same split that
# actually produced xray_model/best_model.pt, so this reconstruction is
# verified correct.
#
# ECG: an earlier version of this module reconstructed a split from
# Final_Dataset/ + final_patient_dataset_summary.xlsx (the same source as
# X-ray) and got AUC == 1.000 (zero-width bootstrap CI) on every condition,
# on BOTH the "test" and "train/val" partitions alike — a sign of train/test
# leakage, since the checkpoint's own training-time metric
# (ck['mean_auc'] = 0.907) matches the thesis's realistic 0.886-0.994 range.
# The root cause: ecg_model/best_model.pt was actually trained from
# Test/patient_dataset_summary.xlsx + Test/sorted_patients/ (train_ecg.py's
# original, hardcoded data source — 11,622 rows, a superset of the later,
# trimmed Final_Dataset/), NOT from Final_Dataset/. Calling train_ecg.py's
# own build_loaders() against that original source (see
# _load_ecg_fairness_cohort) reproduces the exact recovered test split;
# re-running inference on it gives realistic 0.786-0.994 AUCs, confirming
# it as the genuine held-out set. If Test/sorted_patients/ or
# Test/patient_dataset_summary.xlsx are ever moved or deleted, ECG fairness
# degrades gracefully to "unavailable" rather than silently reporting
# in-sample numbers again.
#
# Neither model has a saved test-set predictions file on disk, so
# predictions are re-computed here by re-running each model's own inference
# path (read_ecg/run_ecg_inference, preprocess_xray) over its cohort. This
# is independent of, and does not alter, the single-file Analyze workflow
# above — it only reuses the already-loaded model objects.
# ═════════════════════════════════════════════════════════════════════════════
GENDER_XLSX = BASE / "final_patient_dataset_summary.xlsx"
ECG_SPLIT_XLSX = BASE / "Test" / "patient_dataset_summary.xlsx"
ECG_SORTED_DIR = BASE / "Test" / "sorted_patients"
FAIRNESS_SEED = 42
FAIRNESS_MIN_POS = 25       # positives required in EACH gender group, else "small sample"
FAIRNESS_N_BOOT = 1000      # bootstrap resamples for CIs / the difference test
FAIRNESS_ALPHA = 0.05
FAIRNESS_ECG_COLS = [c for c in ECG_LABEL_COLS if c in ECG_LABEL_DISPLAY]
FAIRNESS_XRAY_COLS = list(XRAY_DISPLAY_COLS)
FAIRNESS_CACHE_PATH = BASE / "fairness_results_cache.json"


def _save_fairness_cache(result: dict) -> dict:
    """Writes `result` (plus a fresh timestamp) to FAIRNESS_CACHE_PATH and
    returns the exact payload written, so the caller can display the same
    timestamp it just saved. Caching is a convenience only — a write
    failure is swallowed rather than blocking the results from showing."""
    payload = dict(result)
    payload["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        FAIRNESS_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass
    return payload


def _load_fairness_cache():
    """Returns the cached fairness result dict, or None if no cache exists
    or it's unreadable/corrupt (treated as a cache miss, not an error)."""
    if not FAIRNESS_CACHE_PATH.exists():
        return None
    try:
        return json.loads(FAIRNESS_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_ecg_fairness_cohort():
    """
    Recovers the genuine, verified-correct ECGNet test split by calling
    train_ecg.py's own build_loaders() against its original data source
    (Test/patient_dataset_summary.xlsx + Test/sorted_patients/) — see the
    module comment above for why this, and not Final_Dataset/, is the
    right source. Returns a DataFrame with subject_id, gender, hea_path,
    and the 9 raw ECG label columns, or None if the original source files
    are unavailable (graceful degradation, no exception raised).
    """
    if not ECG_SPLIT_XLSX.exists() or not ECG_SORTED_DIR.exists():
        return None
    import pandas as pd
    import train_ecg as _te

    _te.SUMMARY_XL = ECG_SPLIT_XLSX
    _te.SORTED_DIR = ECG_SORTED_DIR
    _, _, _, _, test_df = _te.build_loaders(batch_size=8, min_samples=30)
    if test_df is None or len(test_df) == 0:
        return None

    gender_df = pd.read_excel(ECG_SPLIT_XLSX, engine="openpyxl", usecols=["subject_id", "gender"])
    gender_df["subject_id"] = gender_df["subject_id"].astype(str)
    gender_df["gender"] = gender_df["gender"].astype(str).str.strip().str.upper()
    gender_df = gender_df.drop_duplicates(subset="subject_id")

    out = test_df.merge(gender_df, on="subject_id", how="left")
    out = out[out["gender"].isin(["M", "F"])].reset_index(drop=True)
    return out if len(out) else None


def _run_ecg_fairness_inference(app, cohort, progress_cb=None):
    import pandas as pd
    rows = []
    n = len(cohort)
    for i, (_, r) in enumerate(cohort.iterrows()):
        try:
            raw = read_ecg(Path(r["hea_path"]))
            _, confs = run_ecg_inference(app.ecg_model, raw, app.ecg_thresholds, app.ecg_ck)
        except Exception:
            continue
        row = {"subject_id": r["subject_id"], "gender": r["gender"]}
        for col in FAIRNESS_ECG_COLS:
            row[col] = int(r[col])
            row[f"prob_{col}"] = confs.get(col, 0.0)
        rows.append(row)
        if progress_cb and (i % 25 == 0 or i == n - 1):
            progress_cb(f"Fairness Check — ECG inference {i + 1}/{n} …")
    return pd.DataFrame(rows)


def _load_fairness_split():
    """Loads gender + diagnosis metadata and reconstructs the test-patient
    set. Raises FileNotFoundError if the metadata file is missing, so the
    caller can surface a clean error instead of crashing."""
    import pandas as pd
    if not GENDER_XLSX.exists():
        raise FileNotFoundError(f"Gender/diagnosis metadata file not found: {GENDER_XLSX.name}")
    if not FINAL_DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {FINAL_DATASET_DIR.name}")

    df = pd.read_excel(GENDER_XLSX, engine="openpyxl")
    df["subject_id"] = df["subject_id"].astype(str)

    # Same split arithmetic as xray_train.py's make_splits(): patient-level,
    # seed=42, computed on the full unique subject_id list before any
    # gender filtering, so it matches the actual training-time split.
    patients = df["subject_id"].unique()
    rng = np.random.default_rng(FAIRNESS_SEED)
    rng.shuffle(patients)
    n = len(patients)
    test_patients = set(patients[int(0.85 * n):])

    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    return df, test_patients


def _build_xray_fairness_cohort(df_meta, test_patients):
    import pandas as pd
    sub = df_meta[df_meta["subject_id"].isin(test_patients) & df_meta["gender"].isin(["M", "F"])]
    sub = sub.dropna(subset=["cxr_study_id"]).copy()
    sub["cxr_study_id"] = sub["cxr_study_id"].astype(float).astype(int).astype(str)

    rows = []
    for _, r in sub.iterrows():
        png = FINAL_DATASET_DIR / f"patient_{r['subject_id']}" / "xray" / f"cxr_{r['cxr_study_id']}.png"
        if png.exists():
            row = {"subject_id": r["subject_id"], "gender": r["gender"], "png_path": png}
            for col in FAIRNESS_XRAY_COLS:
                v = r.get(col)
                row[col] = int(v) if (pd.notna(v) and float(v) >= 0) else -1  # -1 = uncertain/unmentioned
            rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _run_xray_fairness_inference(app, cohort, progress_cb=None):
    import pandas as pd
    lbl_path = XRAY_MODEL_DIR / "label_classes.json"
    model_cols = json.loads(lbl_path.read_text(encoding="utf-8")) if lbl_path.exists() else XRAY_DISPLAY_COLS
    rows = []
    n = len(cohort)
    with torch.no_grad():
        for i, (_, r) in enumerate(cohort.iterrows()):
            try:
                img = Image.open(r["png_path"])
                tensor = preprocess_xray(img, app.xray_img_size, app.xray_backbone)
                probs = torch.sigmoid(app.xray_model(tensor)).squeeze(0).tolist()
            except Exception:
                continue
            by_label = dict(zip(model_cols, probs))
            row = {"subject_id": r["subject_id"], "gender": r["gender"]}
            for col in FAIRNESS_XRAY_COLS:
                row[col] = int(r[col])
                row[f"prob_{col}"] = by_label.get(col, 0.0)
            rows.append(row)
            if progress_cb and (i % 25 == 0 or i == n - 1):
                progress_cb(f"Fairness Check — X-ray inference {i + 1}/{n} …")
    return pd.DataFrame(rows)


def _bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray,
                       n_boot: int = FAIRNESS_N_BOOT, seed: int = 0, alpha: float = FAIRNESS_ALPHA):
    """Point AUC + percentile bootstrap CI. Returns (auc, lo, hi); lo/hi are
    None if there weren't enough valid resamples (both classes present) to
    form a stable CI, and all three are None if AUC itself isn't computable
    (fewer than 2 classes present)."""
    from sklearn.metrics import roc_auc_score
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return None, None, None
    try:
        point = float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None, None, None

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boots.append(roc_auc_score(yt, y_score[idx]))
        except ValueError:
            continue
    if len(boots) < 50:
        return point, None, None
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def _bootstrap_auc_diff_pvalue(y_true_m, y_score_m, y_true_f, y_score_f,
                                n_boot: int = FAIRNESS_N_BOOT, seed: int = 0):
    """Two-sided bootstrap difference test for H0: AUC_male == AUC_female
    (percentile-bootstrap fallback in place of a DeLong test, as permitted
    when a full DeLong covariance implementation isn't warranted here).
    Independently resamples each gender group with replacement, rebuilds
    the AUC-gap distribution, and reads off the two-sided empirical p-value.
    Returns None if either group never has both classes present."""
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true_m)) < 2 or len(np.unique(y_true_f)) < 2:
        return None
    rng = np.random.default_rng(seed)
    nm, nf = len(y_true_m), len(y_true_f)
    diffs = []
    for _ in range(n_boot):
        im = rng.integers(0, nm, nm)
        jf = rng.integers(0, nf, nf)
        ytm, ytf = y_true_m[im], y_true_f[jf]
        if len(np.unique(ytm)) < 2 or len(np.unique(ytf)) < 2:
            continue
        try:
            diffs.append(roc_auc_score(ytm, y_score_m[im]) - roc_auc_score(ytf, y_score_f[jf]))
        except ValueError:
            continue
    if len(diffs) < 50:
        return None
    diffs = np.array(diffs)
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return min(p, 1.0)


def _evaluate_condition_fairness(condition: str, y_true, y_score, gender, seed_base: int):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    gender = np.asarray(gender)
    m, f = gender == "M", gender == "F"
    ytm, ysm = y_true[m], y_score[m]
    ytf, ysf = y_true[f], y_score[f]
    n_pos_m, n_pos_f = int((ytm == 1).sum()), int((ytf == 1).sum())

    auc_m, lo_m, hi_m = _bootstrap_auc_ci(ytm, ysm, seed=seed_base)
    auc_f, lo_f, hi_f = _bootstrap_auc_ci(ytf, ysf, seed=seed_base + 1)
    p_value = (_bootstrap_auc_diff_pvalue(ytm, ysm, ytf, ysf, seed=seed_base + 2)
               if (auc_m is not None and auc_f is not None) else None)

    small_sample = n_pos_m < FAIRNESS_MIN_POS or n_pos_f < FAIRNESS_MIN_POS
    return {
        "condition": condition,
        "n_male": int(m.sum()), "n_female": int(f.sum()),
        "n_pos_male": n_pos_m, "n_pos_female": n_pos_f,
        "auc_male": auc_m, "ci_male": (lo_m, hi_m),
        "auc_female": auc_f, "ci_female": (lo_f, hi_f),
        "gap": (auc_m - auc_f) if (auc_m is not None and auc_f is not None) else None,
        "p_value": p_value,
        "small_sample": small_sample,
        "significant": p_value is not None and p_value < FAIRNESS_ALPHA,
    }


def _fairness_table_for_cohort(pred_df, cols, seed_base):
    results = []
    for i, col in enumerate(cols):
        sub = pred_df[pred_df[col].isin([0, 1])]
        if len(sub) == 0:
            continue
        results.append(_evaluate_condition_fairness(
            col, sub[col].values.astype(int), sub[f"prob_{col}"].values.astype(float),
            sub["gender"].values, seed_base + i * 10))
    return results


def run_fairness_evaluation(app, progress_cb=None):
    """Orchestrates the gender-fairness check for both models and returns
    the results dict consumed by App._show_fairness_results. ECG uses the
    recovered original test split (see module comment above); if that
    source is unavailable, ECG degrades gracefully to an empty result
    rather than raising or falling back to an untrustworthy split."""
    ecg_results = []
    if app.ecg_model is not None:
        if progress_cb:
            progress_cb("Fairness Check — loading recovered ECG test split …")
        ecg_cohort = _load_ecg_fairness_cohort()
        if ecg_cohort is not None and len(ecg_cohort):
            ecg_pred = _run_ecg_fairness_inference(app, ecg_cohort, progress_cb)
            if len(ecg_pred):
                ecg_results = _fairness_table_for_cohort(ecg_pred, FAIRNESS_ECG_COLS, seed_base=100)

    if progress_cb:
        progress_cb("Fairness Check — loading gender & diagnosis metadata …")
    df_meta, test_patients = _load_fairness_split()

    xray_results = []
    if app.xray_model is not None:
        if progress_cb:
            progress_cb("Fairness Check — building X-ray test cohort …")
        xray_cohort = _build_xray_fairness_cohort(df_meta, test_patients)
        if xray_cohort is not None and len(xray_cohort):
            xray_pred = _run_xray_fairness_inference(app, xray_cohort, progress_cb)
            if len(xray_pred):
                xray_results = _fairness_table_for_cohort(xray_pred, FAIRNESS_XRAY_COLS, seed_base=200)

    if not ecg_results and not xray_results:
        raise RuntimeError("No evaluable test-set cases found (check Final_Dataset, "
                            f"{GENDER_XLSX.name}, and {ECG_SPLIT_XLSX}).")

    all_results = ecg_results + xray_results
    n_total = len(all_results)
    n_no_gap = sum(1 for r in all_results if not r["significant"])
    return {"ecg_results": ecg_results, "xray_results": xray_results,
            "n_total": n_total, "n_no_gap": n_no_gap}


# ═════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE  (red / white, professional clinical theme)
# ═════════════════════════════════════════════════════════════════════════════
RED = "#C8102E"
RED_DARK = "#8F001C"
RED_LIGHT = "#FDECEF"
RED_SOFT = "#F3B9C2"
INK = "#18181B"
MUTED = "#6B7280"
MUTED_LIGHT = "#9CA3AF"
BORDER = "#E7E5E4"
BG = "#FBFAFA"
CARD = "#FFFFFF"
BAR_TRACK = "#E9E7E7"

# Fairness-table row colours: teal = no significant gap, amber = small
# sample (interpret cautiously), red = significant gap.
FAIRNESS_OK_COLOR = "#2E8B8B"
FAIRNESS_SMALL_SAMPLE_COLOR = "#B8860B"
FAIRNESS_SIGNIFICANT_COLOR = "#8F001C"

APP_TITLE = "Multimodal AI Framework for Early Cardiovascular Disease Detection"
APP_SUBTITLE = "Combined ECG & Chest X-ray Analysis  ·  Interpretability & Fairness-Aware Design"
CREDIT_TEXT = "Neelam Fatimah   ·   Master's Thesis"
DISCLAIMER = ("AI-assisted predictions generated for research purposes only — "
              "not a substitute for professional clinical diagnosis.")


def _icon_canvas(parent, draw_fn, size=56, ring_color=RED_LIGHT, bg=CARD):
    """A small round icon badge drawn with plain Canvas primitives (no emoji/font dependency)."""
    c = ctk.CTkCanvas(parent, width=size, height=size, bg=bg, highlightthickness=0, bd=0)
    c.create_oval(2, 2, size - 2, size - 2, fill=ring_color, outline="")
    draw_fn(c, size)
    return c


def _draw_ecg_icon(c, size):
    pts = [0.10, 0.55, 0.28, 0.55, 0.37, 0.30, 0.46, 0.75, 0.55, 0.20, 0.64, 0.55, 0.90, 0.55]
    coords = []
    for i in range(0, len(pts), 2):
        coords.append(pts[i] * size)
        coords.append(pts[i + 1] * size)
    c.create_line(*coords, fill=RED, width=max(2, size // 20), smooth=False,
                  joinstyle="round", capstyle="round")


def _draw_xray_icon(c, size):
    m = size * 0.24
    c.create_rectangle(m, m * 0.85, size - m, size - m * 0.85, outline=RED,
                        width=max(2, size // 22))
    c.create_arc(m * 1.5, m * 1.3, size - m * 1.5, size - m * 1.3,
                 start=200, extent=140, style="arc", outline=RED, width=max(2, size // 26))


class ConditionRow(ctk.CTkFrame):
    """One clinical finding: name, DETECTED/NOT DETECTED badge, % confidence, progress bar."""

    def __init__(self, parent, name, detected, confidence, description=None, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        self.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(self, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", pady=(10, 2))
        top.grid_columnconfigure(0, weight=1)

        name_lbl = ctk.CTkLabel(top, text=name, anchor="w",
                                 font=ctk.CTkFont(size=16, weight="bold"), text_color=INK)
        name_lbl.grid(row=0, column=0, sticky="w")

        meta = ctk.CTkFrame(top, fg_color="transparent")
        meta.grid(row=0, column=1, sticky="e")

        badge_text = "DETECTED" if detected else "NOT DETECTED"
        badge = ctk.CTkLabel(meta, text=badge_text,
                              font=ctk.CTkFont(size=12, weight="bold"),
                              fg_color=RED if detected else BAR_TRACK,
                              text_color="#FFFFFF" if detected else MUTED,
                              corner_radius=999, width=118, height=26)
        badge.pack(side="left", padx=(0, 12))

        conf_lbl = ctk.CTkLabel(meta, text=f"{confidence:.1f}%",
                                 font=ctk.CTkFont(size=17, weight="bold"),
                                 text_color=RED if detected else MUTED, width=58, anchor="e")
        conf_lbl.pack(side="left")

        if description:
            desc = ctk.CTkLabel(self, text=description, anchor="w",
                                 font=ctk.CTkFont(size=13), text_color=MUTED)
            desc.grid(row=1, column=0, sticky="w", pady=(0, 2))

        bar = ctk.CTkProgressBar(self, height=7, corner_radius=999,
                                  fg_color=BAR_TRACK,
                                  progress_color=RED if detected else BAR_TRACK)
        bar.grid(row=2, column=0, sticky="ew", pady=(4, 10))
        bar.set(max(0.02, confidence / 100))

        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1)
        sep.grid(row=3, column=0, sticky="ew")


class FairnessConditionRow(ctk.CTkFrame):
    """One row of the fairness-check table: Condition | AUC(M) | AUC(F) |
    Gap | p-value | Flag. `r` is a result dict from _evaluate_condition_fairness."""

    _COL_WEIGHTS = (3, 2, 2, 1, 1, 2)

    def __init__(self, parent, r, **kw):
        super().__init__(parent, fg_color="transparent", **kw)
        for i, w in enumerate(self._COL_WEIGHTS):
            self.grid_columnconfigure(i, weight=w, uniform="fc")

        if r["small_sample"]:
            color, flag = FAIRNESS_SMALL_SAMPLE_COLOR, "Small sample — interpret cautiously"
        elif r["significant"]:
            color, flag = FAIRNESS_SIGNIFICANT_COLOR, "Significant gender gap"
        else:
            color, flag = FAIRNESS_OK_COLOR, "No significant gap"

        def auc_text(auc, ci):
            if auc is None:
                return "n/a"
            lo, hi = ci
            return f"{auc:.3f}" if lo is None else f"{auc:.3f} [{lo:.3f}-{hi:.3f}]"

        name = ECG_LABEL_DISPLAY.get(r["condition"], r["condition"])
        gap_text = f"{r['gap']:+.3f}" if r["gap"] is not None else "n/a"
        p_text = f"{r['p_value']:.3f}" if r["p_value"] is not None else "n/a"

        cells = [
            (name, INK, "bold"), (auc_text(r["auc_male"], r["ci_male"]), INK, "normal"),
            (auc_text(r["auc_female"], r["ci_female"]), INK, "normal"),
            (gap_text, INK, "normal"), (p_text, INK, "normal"),
        ]
        for i, (text, tc, weight) in enumerate(cells):
            ctk.CTkLabel(self, text=text, font=ctk.CTkFont(size=13, weight=weight),
                         text_color=tc, anchor="w").grid(row=0, column=i, sticky="w", padx=(0, 8), pady=6)
        ctk.CTkLabel(self, text=flag, font=ctk.CTkFont(size=12, weight="bold"), text_color=color,
                     anchor="w", wraplength=190).grid(row=0, column=5, sticky="w", pady=6)

        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1)
        sep.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(4, 0))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE + "  —  Neelam Fatimah")
        self.configure(fg_color=BG)
        self.geometry("1440x920")
        self.minsize(1180, 760)
        self.after(50, lambda: self.state("zoomed"))
        self.bind("<F11>", lambda e: self.attributes(
            "-fullscreen", not self.attributes("-fullscreen")))
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        self.ecg_path = None
        self.xray_path = None
        self.results_frame = None
        self.fairness_frame = None
        self.methodology_frame = None
        self._methodology_prev_visible = None

        self.ecg_model, self.ecg_thresholds, self.ecg_ck = load_ecg_model()
        self.xray_model, self.xray_thresholds, self.xray_img_size, self.xray_backbone = load_xray_model()

        self._build_header()
        self._build_body()

    # ── header ──────────────────────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=RED, corner_radius=0, height=104)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=32, pady=14, fill="y")

        mark = _icon_canvas(left, _draw_ecg_icon, size=48, ring_color="#FFFFFF", bg=RED)
        mark.pack(side="left", padx=(0, 16))

        text_col = ctk.CTkFrame(left, fg_color="transparent")
        text_col.pack(side="left", fill="y")
        ctk.CTkLabel(text_col, text=APP_TITLE, anchor="w", justify="left",
                     font=ctk.CTkFont(size=23, weight="bold"),
                     text_color="#FFFFFF", wraplength=1000).pack(anchor="w")
        ctk.CTkLabel(text_col, text=APP_SUBTITLE, anchor="w",
                     font=ctk.CTkFont(size=14),
                     text_color="#FBD7DC").pack(anchor="w", pady=(3, 0))

        right = ctk.CTkFrame(header, fg_color="#DE4A62", corner_radius=999)
        right.pack(side="right", padx=32, pady=34)
        ctk.CTkLabel(right, text=CREDIT_TEXT, font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#FFFFFF").pack(padx=18, pady=8)

    # ── body ────────────────────────────────────────────────────────────────
    def _build_body(self):
        self.scroll = ctk.CTkScrollableFrame(self, fg_color=BG,
                                              scrollbar_button_color=RED_SOFT,
                                              scrollbar_button_hover_color=RED)
        self.scroll.pack(fill="both", expand=True)
        self.content = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=48, pady=32)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_upload_section(self.content)
        self._build_analyze_section(self.content)

        footer = ctk.CTkLabel(self.content, text=DISCLAIMER, font=ctk.CTkFont(size=12),
                               text_color=MUTED_LIGHT)
        footer.grid(row=99, column=0, pady=(30, 6))

    # ── upload cards ────────────────────────────────────────────────────────
    def _build_upload_section(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        row.grid_columnconfigure((0, 1), weight=1, uniform="card")

        self.ecg_card, self.ecg_select_btn, self.ecg_chip = self._upload_card(
            row, col=0, icon_fn=_draw_ecg_icon, title="ECG Recording",
            hint="12-lead recording  (.hea + .dat)",
            btn_text="Browse ECG File", command=self._on_select_ecg,
            enabled=self.ecg_model is not None)

        self.xray_card, self.xray_select_btn, self.xray_chip = self._upload_card(
            row, col=1, icon_fn=_draw_xray_icon, title="Chest X-ray",
            hint="Frontal chest radiograph  (.png / .jpg)",
            btn_text="Browse X-ray Image", command=self._on_select_xray,
            enabled=self.xray_model is not None)

    def _upload_card(self, parent, col, icon_fn, title, hint, btn_text, command, enabled):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                             border_width=1.5, border_color=RED_SOFT)
        card.grid(row=0, column=col, sticky="nsew", padx=10, pady=10, ipady=8)

        icon = _icon_canvas(card, icon_fn, size=56, ring_color=RED_LIGHT)
        icon.pack(pady=(26, 12))

        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=INK).pack()
        ctk.CTkLabel(card, text=hint, font=ctk.CTkFont(size=13), text_color=MUTED).pack(
            pady=(2, 18))

        btn = ctk.CTkButton(card, text=btn_text, command=command,
                             fg_color=CARD, hover_color=RED_LIGHT,
                             text_color=RED, border_width=1.6, border_color=RED,
                             font=ctk.CTkFont(size=14, weight="bold"),
                             corner_radius=999, width=210, height=40,
                             state="normal" if enabled else "disabled")
        btn.pack(pady=(0, 6))

        chip = ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=13, weight="bold"),
                             fg_color=RED_LIGHT, text_color=RED_DARK,
                             corner_radius=999, height=30)
        # chip is packed on demand once a file is chosen

        if not enabled:
            ctk.CTkLabel(card, text="Model files missing — train this model first.",
                         font=ctk.CTkFont(size=12, weight="bold"), text_color=RED_DARK,
                         wraplength=260).pack(pady=(4, 20))
        else:
            ctk.CTkLabel(card, text=" ", height=20).pack()

        return card, btn, chip

    # ── analyze section ─────────────────────────────────────────────────────
    def _build_analyze_section(self, parent):
        sec = ctk.CTkFrame(parent, fg_color="transparent")
        sec.grid(row=1, column=0, pady=(18, 30))

        btn_row = ctk.CTkFrame(sec, fg_color="transparent")
        btn_row.pack()

        self.analyze_btn = ctk.CTkButton(
            btn_row, text="Analyze", command=self._on_analyze,
            fg_color=RED, hover_color=RED_DARK, text_color="#FFFFFF",
            font=ctk.CTkFont(size=18, weight="bold"),
            corner_radius=999, width=260, height=54, state="disabled")
        self.analyze_btn.pack(side="left", padx=(0, 14))

        self.fairness_btn = ctk.CTkButton(
            btn_row, text="Fairness Check", command=self._on_fairness_check,
            fg_color=CARD, hover_color=RED_LIGHT, text_color=RED,
            border_width=1.6, border_color=RED,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=999, width=220, height=54,
            state="normal" if (self.ecg_model is not None or self.xray_model is not None) else "disabled")
        self.fairness_btn.pack(side="left")

        self.methodology_btn = ctk.CTkButton(
            btn_row, text="Novelty & Methodology", command=self._show_methodology,
            fg_color=CARD, hover_color=RED_LIGHT, text_color=RED,
            border_width=1.6, border_color=RED,
            font=ctk.CTkFont(size=16, weight="bold"),
            corner_radius=999, width=220, height=54)
        self.methodology_btn.pack(side="left", padx=(14, 0))

        self.status_lbl = ctk.CTkLabel(
            sec, text="Select an ECG file and a chest X-ray image to begin.",
            font=ctk.CTkFont(size=14), text_color=MUTED)
        self.status_lbl.pack(pady=(12, 0))

    # ── file selection handlers ─────────────────────────────────────────────
    def _initial_dir(self):
        return str(FINAL_DATASET_DIR) if FINAL_DATASET_DIR.exists() else str(BASE)

    def _on_select_ecg(self):
        path = filedialog.askopenfilename(
            title="Select ECG Header File",
            initialdir=self._initial_dir(),
            filetypes=[("ECG Header Files", "*.hea"), ("All files", "*.*")])
        if not path:
            return
        p = Path(path)
        if not p.with_suffix(".dat").exists():
            messagebox.showerror("Missing signal file",
                                  "No matching .dat file was found next to this .hea file.")
            return
        self.ecg_path = p
        self.ecg_chip.configure(text=f"✓  {p.name}")
        self.ecg_chip.pack(pady=(0, 20))
        self._update_analyze_state()

    def _on_select_xray(self):
        path = filedialog.askopenfilename(
            title="Select Chest X-ray Image",
            initialdir=self._initial_dir(),
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")])
        if not path:
            return
        p = Path(path)
        try:
            Image.open(p).verify()
        except Exception:
            messagebox.showerror("Invalid image", "This file could not be read as an image.")
            return
        self.xray_path = p
        self.xray_chip.configure(text=f"✓  {p.name}")
        self.xray_chip.pack(pady=(0, 20))
        self._update_analyze_state()

    def _update_analyze_state(self):
        if self.ecg_path and self.xray_path:
            self.analyze_btn.configure(state="normal")
            self.status_lbl.configure(text="Ready to analyze.", text_color=MUTED)

    # ── analysis ─────────────────────────────────────────────────────────────
    def _on_analyze(self):
        self.analyze_btn.configure(state="disabled", text="Analyzing …")
        self.ecg_select_btn.configure(state="disabled")
        self.xray_select_btn.configure(state="disabled")
        self.fairness_btn.configure(state="disabled")
        self.status_lbl.configure(text="Running inference on both models …", text_color=MUTED)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            ecg_result = self._analyze_ecg()
            xray_result = self._analyze_xray()
            self.after(0, lambda: self._show_results(ecg_result, xray_result))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda: self._analysis_failed(msg))

    def _analysis_failed(self, msg):
        self.status_lbl.configure(text=f"Analysis failed: {msg}", text_color=RED_DARK)
        self._restore_buttons()

    def _restore_buttons(self):
        self.analyze_btn.configure(state="normal", text="Analyze")
        self.ecg_select_btn.configure(state="normal")
        self.xray_select_btn.configure(state="normal")
        if self.ecg_model is not None or self.xray_model is not None:
            self.fairness_btn.configure(state="normal")

    def _analyze_ecg(self):
        raw = read_ecg(self.ecg_path)
        preds, confs = run_ecg_inference(self.ecg_model, raw, self.ecg_thresholds, self.ecg_ck)
        groups = []
        detected_keys, detected_names = [], []
        for grp_name, cols in ECG_DISPLAY_GROUPS:
            items = []
            for c in cols:
                det = bool(preds.get(c, False))
                items.append((ECG_LABEL_DISPLAY[c], det, confs.get(c, 0.0) * 100))
                if det:
                    detected_keys.append(c)
                    detected_names.append(ECG_LABEL_DISPLAY[c])
            groups.append((grp_name, items))

        # Highlight the clinically expected region for each detected
        # condition, located purely via independent neurokit2 wave
        # segmentation — no model attention involved in this plot.
        plot_img = render_clinical_regions_plot(raw, detected_keys, hea_path=self.ecg_path) \
            if detected_keys else render_ecg_plot(raw)

        return {"plot": plot_img, "groups": groups, "detected_count": len(detected_keys),
                "detected_names": detected_names}

    def _analyze_xray(self):
        img = Image.open(self.xray_path)
        tensor = preprocess_xray(img, self.xray_img_size, self.xray_backbone)
        with torch.no_grad():
            probs = torch.sigmoid(self.xray_model(tensor)).squeeze(0).tolist()

        lbl_path = XRAY_MODEL_DIR / "label_classes.json"
        model_cols = json.loads(lbl_path.read_text(encoding="utf-8")) if lbl_path.exists() else XRAY_DISPLAY_COLS
        by_label = dict(zip(model_cols, probs))

        items = []
        detected_n = 0
        detected_conditions = []
        for col in XRAY_DISPLAY_COLS:
            prob = by_label.get(col, 0.0)
            thr = self.xray_thresholds.get(col, DEFAULT_XRAY_THRESH)
            positive = prob >= thr
            if positive:
                detected_n += 1
                detected_conditions.append(col)
            items.append((col, XRAY_LABEL_INFO.get(col, ""), positive, prob * 100))

        display_img = render_clinical_regions_xray(img, detected_conditions) \
            if detected_conditions else xray_thumbnail(img)

        return {"image": display_img, "items": items, "detected_count": detected_n,
                "detected_conditions": detected_conditions}

    # ── results ─────────────────────────────────────────────────────────────
    def _show_results(self, ecg_result, xray_result):
        self._restore_buttons()
        self.status_lbl.configure(text="Analysis complete.", text_color=MUTED)

        if self.fairness_frame is not None:
            self.fairness_frame.grid_remove()
        if self.methodology_frame is not None:
            self.methodology_frame.destroy()
            self.methodology_frame = None
        if self.results_frame is not None:
            self.results_frame.destroy()

        total = ecg_result["detected_count"] + xray_result["detected_count"]
        self.results_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.results_frame.grid(row=2, column=0, sticky="ew")
        self.results_frame.grid_columnconfigure(0, weight=1)

        self._build_summary_banner(self.results_frame, total)

        grid = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        grid.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        grid.grid_columnconfigure((0, 1), weight=1, uniform="panel")

        self._build_ecg_panel(grid, ecg_result)
        self._build_xray_panel(grid, xray_result)

        reset_btn = ctk.CTkButton(
            self.results_frame, text="Analyze Another Case", command=self._reset,
            fg_color=CARD, hover_color=RED_LIGHT, text_color=RED,
            border_width=1.6, border_color=RED, font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=999, width=240, height=42)
        reset_btn.grid(row=2, column=0, pady=(26, 0))

        self.update_idletasks()
        self.scroll._parent_canvas.yview_moveto(1.0)

    def _build_summary_banner(self, parent, total):
        clear = total == 0
        banner = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                               border_width=1.5, border_color=BORDER)
        banner.grid(row=0, column=0, sticky="ew")
        accent = ctk.CTkFrame(banner, fg_color=MUTED_LIGHT if clear else RED,
                               width=7, corner_radius=0)
        accent.pack(side="left", fill="y")

        inner = ctk.CTkFrame(banner, fg_color="transparent")
        inner.pack(side="left", fill="both", expand=True, padx=24, pady=20)

        num = ctk.CTkLabel(inner, text=str(total), font=ctk.CTkFont(size=42, weight="bold"),
                            text_color=MUTED if clear else RED)
        num.pack(side="left", padx=(0, 20))

        text_col = ctk.CTkFrame(inner, fg_color="transparent")
        text_col.pack(side="left", fill="y")
        label = "No Findings Detected" if clear else \
            f"{total} Condition{'s' if total > 1 else ''} Detected"
        ctk.CTkLabel(text_col, text=label, font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w")
        ctk.CTkLabel(text_col, text="Across ECG and chest X-ray analysis",
                     font=ctk.CTkFont(size=13), text_color=MUTED, anchor="w").pack(anchor="w")

    def _build_ecg_panel(self, parent, result):
        panel = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                              border_width=1.5, border_color=BORDER)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        inner = ctk.CTkFrame(panel, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=20)

        ctk.CTkLabel(inner, text="ECG Findings", font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w")
        ctk.CTkLabel(inner, text="ECGNet  ·  1D Convolutional Neural Network",
                     font=ctk.CTkFont(size=12), text_color=MUTED, anchor="w").pack(anchor="w")

        if result["detected_names"]:
            ctk.CTkLabel(inner, text="Highlighted regions: " + ", ".join(result["detected_names"]),
                         font=ctk.CTkFont(size=12, weight="bold"), text_color=INK,
                         anchor="w", wraplength=480).pack(anchor="w", pady=(6, 0))
            ctk.CTkLabel(inner, text="Each condition's expected waveform region is shaded in its "
                                      "own colour (see legend on the plot below), located via "
                                      "independent neurokit2 wave segmentation — not model attention.",
                         font=ctk.CTkFont(size=11), text_color=MUTED, anchor="w",
                         wraplength=480).pack(anchor="w")

        img = result["plot"]
        w, h = img.size
        disp_w = 520
        disp_h = int(h * disp_w / w)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(disp_w, disp_h))
        img_lbl = ctk.CTkLabel(inner, image=ctk_img, text="")
        img_lbl.pack(pady=(8, 12))

        for grp_name, items in result["groups"]:
            ctk.CTkLabel(inner, text=grp_name.upper(), font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=MUTED, anchor="w").pack(anchor="w", pady=(10, 0))
            for name, detected, confidence in items:
                ConditionRow(inner, name, detected, confidence).pack(fill="x")

    def _build_xray_panel(self, parent, result):
        panel = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                              border_width=1.5, border_color=BORDER)
        panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        inner = ctk.CTkFrame(panel, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=20)

        ctk.CTkLabel(inner, text="Chest X-ray Findings", font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w")
        ctk.CTkLabel(inner, text="DenseNet-121  ·  Chest Radiograph Classifier",
                     font=ctk.CTkFont(size=12), text_color=MUTED, anchor="w").pack(anchor="w")

        if result.get("detected_conditions"):
            ctk.CTkLabel(inner, text="Highlighted regions: " + ", ".join(result["detected_conditions"]),
                         font=ctk.CTkFont(size=12, weight="bold"), text_color=INK,
                         anchor="w", wraplength=420).pack(anchor="w", pady=(6, 0))
            ctk.CTkLabel(inner, text="Each condition's expected region/measurement is drawn from "
                                      "Otsu thresholding and contour geometry on the image itself "
                                      "(see legend below the image) — not model attention.",
                         font=ctk.CTkFont(size=11), text_color=MUTED, anchor="w",
                         wraplength=420).pack(anchor="w", pady=(0, 12))

        img = result["image"]
        w, h = img.size
        disp_w = 380
        disp_h = int(h * disp_w / w)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(disp_w, disp_h))
        img_lbl = ctk.CTkLabel(inner, image=ctk_img, text="")
        img_lbl.pack(pady=(0, 12))

        for name, description, detected, confidence in result["items"]:
            ConditionRow(inner, name, detected, confidence, description).pack(fill="x")

    def _reset(self):
        self.ecg_path = None
        self.xray_path = None
        self.ecg_chip.configure(text="")
        self.ecg_chip.pack_forget()
        self.xray_chip.configure(text="")
        self.xray_chip.pack_forget()
        self.analyze_btn.configure(state="disabled")
        self.status_lbl.configure(text="Select an ECG file and a chest X-ray image to begin.",
                                   text_color=MUTED)
        if self.results_frame is not None:
            self.results_frame.destroy()
            self.results_frame = None
        if self.fairness_frame is not None:
            self.fairness_frame.destroy()
            self.fairness_frame = None
        if self.methodology_frame is not None:
            self.methodology_frame.destroy()
            self.methodology_frame = None
        self.update_idletasks()
        self.scroll._parent_canvas.yview_moveto(0.0)

    # ── fairness check ──────────────────────────────────────────────────────
    def _on_fairness_check(self, force_recompute=False):
        if not force_recompute:
            cached = _load_fairness_cache()
            if cached is not None:
                self.status_lbl.configure(
                    text=f"Loaded cached fairness results (last computed {cached.get('timestamp', '-')}).",
                    text_color=MUTED)
                self._show_fairness_results(cached)
                return

        self.fairness_btn.configure(state="disabled", text="Running Fairness Check …")
        self.analyze_btn.configure(state="disabled")
        self.status_lbl.configure(text="Fairness Check — starting …", text_color=MUTED)
        threading.Thread(target=self._run_fairness_check, daemon=True).start()

    def _run_fairness_check(self):
        def progress(msg):
            self.after(0, lambda: self.status_lbl.configure(text=msg, text_color=MUTED))
        try:
            result = run_fairness_evaluation(self, progress_cb=progress)
            payload = _save_fairness_cache(result)
            self.after(0, lambda: self._show_fairness_results(payload))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda: self._fairness_failed(msg))

    def _fairness_failed(self, msg):
        self.status_lbl.configure(text=f"Fairness evaluation failed: {msg}", text_color=RED_DARK)
        self._restore_fairness_button()

    def _restore_fairness_button(self):
        self.fairness_btn.configure(state="normal", text="Fairness Check")
        self.analyze_btn.configure(state="normal" if (self.ecg_path and self.xray_path) else "disabled")

    def _show_fairness_results(self, result):
        self._restore_fairness_button()
        ts = result.get("timestamp")
        self.status_lbl.configure(
            text=f"Fairness evaluation complete. (Last computed: {ts})" if ts else "Fairness evaluation complete.",
            text_color=MUTED)

        if self.results_frame is not None:
            self.results_frame.grid_remove()
        if self.methodology_frame is not None:
            self.methodology_frame.destroy()
            self.methodology_frame = None
        if self.fairness_frame is not None:
            self.fairness_frame.destroy()

        self.fairness_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.fairness_frame.grid(row=2, column=0, sticky="ew")
        self.fairness_frame.grid_columnconfigure(0, weight=1)

        self._build_fairness_summary_banner(self.fairness_frame, result)

        next_row = 1
        if not result["ecg_results"]:
            note = ctk.CTkLabel(
                self.fairness_frame,
                text="ECGNet is not included above: its recovered original test-split source "
                     "(Test/patient_dataset_summary.xlsx + Test/sorted_patients/) is unavailable "
                     "right now, so an ECG fairness result would not be trustworthy. Only the "
                     "X-ray model's fairness result is shown.",
                font=ctk.CTkFont(size=12), text_color=MUTED, anchor="w",
                justify="left", wraplength=900)
            note.grid(row=next_row, column=0, sticky="w", pady=(10, 0))
            next_row += 1

        if result["ecg_results"]:
            self._build_fairness_table(self.fairness_frame, "ECG Conditions  ·  ECGNet",
                                        result["ecg_results"], next_row)
            next_row += 1
        if result["xray_results"]:
            self._build_fairness_table(self.fairness_frame, "Chest X-ray Conditions  ·  DenseNet-121",
                                        result["xray_results"], next_row)
            next_row += 1

        footer = ctk.CTkFrame(self.fairness_frame, fg_color="transparent")
        footer.grid(row=next_row, column=0, pady=(20, 0))

        back_btn = ctk.CTkButton(
            footer, text="Back to Analysis", command=self._back_to_analysis,
            fg_color=CARD, hover_color=RED_LIGHT, text_color=RED,
            border_width=1.6, border_color=RED, font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=999, width=220, height=42)
        back_btn.pack(side="left")

        recompute_btn = ctk.CTkButton(
            footer, text="Recompute", command=lambda: self._on_fairness_check(force_recompute=True),
            fg_color=CARD, hover_color=RED_LIGHT, text_color=MUTED,
            border_width=1.2, border_color=BORDER, font=ctk.CTkFont(size=13, weight="bold"),
            corner_radius=999, width=150, height=42)
        recompute_btn.pack(side="left", padx=(12, 0))

        self.update_idletasks()
        self.scroll._parent_canvas.yview_moveto(1.0)

    def _back_to_analysis(self):
        if self.fairness_frame is not None:
            self.fairness_frame.destroy()
            self.fairness_frame = None
        if self.results_frame is not None:
            self.results_frame.grid()
        self.status_lbl.configure(
            text="Ready to analyze." if (self.ecg_path and self.xray_path)
            else "Select an ECG file and a chest X-ray image to begin.", text_color=MUTED)
        self.update_idletasks()
        self.scroll._parent_canvas.yview_moveto(0.0)

    def _build_fairness_summary_banner(self, parent, result):
        n_total, n_no_gap = result["n_total"], result["n_no_gap"]
        all_clear = n_total > 0 and n_no_gap == n_total
        banner = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                               border_width=1.5, border_color=BORDER)
        banner.grid(row=0, column=0, sticky="ew")
        accent = ctk.CTkFrame(banner, fg_color=FAIRNESS_OK_COLOR if all_clear else RED,
                               width=7, corner_radius=0)
        accent.pack(side="left", fill="y")

        inner = ctk.CTkFrame(banner, fg_color="transparent")
        inner.pack(side="left", fill="both", expand=True, padx=24, pady=20)

        num = ctk.CTkLabel(inner, text=f"{n_no_gap}/{n_total}",
                            font=ctk.CTkFont(size=36, weight="bold"),
                            text_color=FAIRNESS_OK_COLOR if all_clear else RED)
        num.pack(side="left", padx=(0, 20))

        text_col = ctk.CTkFrame(inner, fg_color="transparent")
        text_col.pack(side="left", fill="y")
        ctk.CTkLabel(text_col, text=f"{n_no_gap} of {n_total} conditions show no significant "
                                     "gender gap (p > 0.05)",
                     font=ctk.CTkFont(size=17, weight="bold"), text_color=INK,
                     anchor="w", wraplength=560).pack(anchor="w")
        ctk.CTkLabel(text_col, text="Male vs. Female AUC, held-out test-set patients  ·  "
                                     "bootstrap 95% CI  ·  bootstrap difference test",
                     font=ctk.CTkFont(size=13), text_color=MUTED, anchor="w").pack(anchor="w")
        ts = result.get("timestamp")
        if ts:
            ctk.CTkLabel(text_col, text=f"Last computed: {ts}",
                         font=ctk.CTkFont(size=12), text_color=MUTED_LIGHT, anchor="w").pack(anchor="w", pady=(2, 0))

    def _build_fairness_table(self, parent, title, results, row):
        panel = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                              border_width=1.5, border_color=BORDER)
        panel.grid(row=row, column=0, sticky="ew", pady=(16, 0))

        inner = ctk.CTkFrame(panel, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=20)

        ctk.CTkLabel(inner, text=title, font=ctk.CTkFont(size=17, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w", pady=(0, 12))

        header = ctk.CTkFrame(inner, fg_color="transparent")
        header.pack(fill="x")
        for i, w in enumerate(FairnessConditionRow._COL_WEIGHTS):
            header.grid_columnconfigure(i, weight=w, uniform="fc")
        for i, text in enumerate(("Condition", "AUC (Male)", "AUC (Female)", "Gap", "p-value", "Flag")):
            ctk.CTkLabel(header, text=text.upper(), font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=MUTED, anchor="w").grid(row=0, column=i, sticky="w", padx=(0, 8))

        sep = ctk.CTkFrame(inner, fg_color=BORDER, height=1)
        sep.pack(fill="x", pady=(6, 4))

        for r in results:
            FairnessConditionRow(inner, r).pack(fill="x")

    # ── novelty & methodology panel ─────────────────────────────────────────
    def _show_methodology(self):
        self._methodology_prev_visible = None
        if self.results_frame is not None and self.results_frame.winfo_ismapped():
            self._methodology_prev_visible = "results"
            self.results_frame.grid_remove()
        if self.fairness_frame is not None and self.fairness_frame.winfo_ismapped():
            self._methodology_prev_visible = "fairness"
            self.fairness_frame.grid_remove()

        if self.methodology_frame is not None:
            self.methodology_frame.destroy()
        self.methodology_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.methodology_frame.grid(row=2, column=0, sticky="ew")
        self.methodology_frame.grid_columnconfigure(0, weight=1)
        self._build_methodology_panel(self.methodology_frame)

        self.update_idletasks()
        self.scroll._parent_canvas.yview_moveto(0.0)

    def _close_methodology(self):
        if self.methodology_frame is not None:
            self.methodology_frame.destroy()
            self.methodology_frame = None
        if self._methodology_prev_visible == "results" and self.results_frame is not None:
            self.results_frame.grid()
        elif self._methodology_prev_visible == "fairness" and self.fairness_frame is not None:
            self.fairness_frame.grid()
        self.update_idletasks()
        self.scroll._parent_canvas.yview_moveto(0.0)

    def _build_methodology_panel(self, parent):
        # ── title card ──────────────────────────────────────────────────────
        title_card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                                   border_width=1.5, border_color=BORDER)
        title_card.grid(row=0, column=0, sticky="ew")
        t_accent = ctk.CTkFrame(title_card, fg_color=RED, width=7, corner_radius=0)
        t_accent.pack(side="left", fill="y")
        t_inner = ctk.CTkFrame(title_card, fg_color="transparent")
        t_inner.pack(side="left", fill="both", expand=True, padx=24, pady=20)
        ctk.CTkLabel(t_inner, text="Novelty & Methodology", font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w")
        ctk.CTkLabel(t_inner, text="A Multimodal AI Framework Using ECG and Chest X-ray for "
                                    "Early Cardiovascular Disease Detection, with Interpretability "
                                    "and Fairness Evaluation.",
                     font=ctk.CTkFont(size=14, weight="bold"), text_color=RED, anchor="w",
                     wraplength=900, justify="left").pack(anchor="w", pady=(6, 0))
        ctk.CTkLabel(t_inner, text="This page covers the three pillars behind that title: how "
                                    "the AI makes each prediction, how that prediction is then "
                                    "independently verified using non-AI signal and image "
                                    "processing, and how the whole system is evaluated for "
                                    "fairness between male and female patients.",
                     font=ctk.CTkFont(size=13), text_color=MUTED, anchor="w",
                     wraplength=900, justify="left").pack(anchor="w", pady=(6, 0))

        # ── novelty points ──────────────────────────────────────────────────
        novelty_card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                                     border_width=1.5, border_color=BORDER)
        novelty_card.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        n_inner = ctk.CTkFrame(novelty_card, fg_color="transparent")
        n_inner.pack(fill="both", expand=True, padx=24, pady=20)
        ctk.CTkLabel(n_inner, text="Novelty", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w", pady=(0, 12))

        novelty_points = [
            ("Multimodal fusion",
             "Combines ECG and chest X-ray in one diagnostic pipeline for cardiovascular disease, "
             "instead of the single-modality approach most comparable student and research projects use."),
            ("Independent, non-AI evidence verification",
             "Explanations are not drawn from the model's own internal attention. A separate, "
             "deterministic layer re-checks every detected condition using classical signal "
             "processing (neurokit2 wave delineation for ECG) and image geometry (Otsu "
             "thresholding for X-ray) — evidence "
             "that can be verified independently of the network's internal weights."),
            ("Real clinical measurements, not heatmaps",
             "Cardiomegaly is shown as an actual computed Cardiothoracic Ratio (CTR) with measurement "
             "lines, not a vague colour wash — the same measurement a radiologist would take."),
            ("Left/right laterality detection",
             "For Oedema and Pleural Effusion, real pixel-density is compared between the left and "
             "right lung fields, so the app reports which side is actually affected instead of "
             "highlighting both sides by default."),
            ("Built-in fairness auditing",
             "A gender-fairness check is a first-class feature of the app itself — per-condition AUC "
             "for Male vs. Female, 95% bootstrap confidence intervals, and a significance test — not "
             "an afterthought in a separate notebook."),
            ("Verified, leakage-free evaluation",
             "The held-out test split used for fairness evaluation was traced back to its original "
             "source and validated by confirming realistic (non-perfect) AUC on genuinely unseen "
             "patients — catching and fixing a real train/test leakage bug during development."),
        ]
        for i, (head, body) in enumerate(novelty_points):
            self._build_novelty_row(n_inner, i + 1, head, body)

        # ── two-pipeline methodology explainer ─────────────────────────────
        method_row = ctk.CTkFrame(parent, fg_color="transparent")
        method_row.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        method_row.grid_columnconfigure((0, 1), weight=1, uniform="method")

        self._build_pipeline_card(
            method_row, col=0, title="Pipeline A  ·  AI Diagnosis",
            subtitle="ECGNet (1D CNN)  +  DenseNet-121",
            body="The only AI part of the app. The trained model outputs a probability (0-100%) "
                 "for each condition; if it crosses a learned threshold, the condition is marked "
                 "\"detected.\" This pipeline answers one question only: is the condition present?",
            color=RED)

        self._build_pipeline_card(
            method_row, col=1, title="Pipeline B  ·  Independent Verification",
            subtitle="neurokit2 (ECG)  +  Otsu / image geometry (X-ray)",
            body="Runs completely separately, after Pipeline A, and never looks inside the neural "
                 "network. For each detected condition, it locates the medically expected region "
                 "using classical, deterministic rules — answering: here is where the evidence is.",
            color=FAIRNESS_OK_COLOR)

        # ── validated performance ───────────────────────────────────────────
        perf_card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                                  border_width=1.5, border_color=BORDER)
        perf_card.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        p_inner = ctk.CTkFrame(perf_card, fg_color="transparent")
        p_inner.pack(fill="both", expand=True, padx=24, pady=20)
        ctk.CTkLabel(p_inner, text="Validated Performance", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w", pady=(0, 10))

        stats = [
            ("ECG dataset", "5,926 recordings  ·  5,218 patients  ·  70/15/15 patient-level split"),
            ("ECG held-out test AUC", "0.907 mean  (range 0.886 - 0.994 across displayed conditions)"),
            ("Chest X-ray dataset", "6,318 radiographs  ·  5,218 patients  ·  70/15/15 patient-level split"),
            ("X-ray held-out test AUC", "0.826 macro average  (Cardiomegaly, Oedema, Pleural Effusion)"),
        ]
        for label, value in stats:
            row = ctk.CTkFrame(p_inner, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=13, weight="bold"),
                         text_color=INK, anchor="w", width=220).pack(side="left")
            ctk.CTkLabel(row, text=value, font=ctk.CTkFont(size=13), text_color=MUTED,
                         anchor="w").pack(side="left")

        close_btn = ctk.CTkButton(
            parent, text="Close", command=self._close_methodology,
            fg_color=CARD, hover_color=RED_LIGHT, text_color=RED,
            border_width=1.6, border_color=RED, font=ctk.CTkFont(size=14, weight="bold"),
            corner_radius=999, width=160, height=42)
        close_btn.grid(row=4, column=0, pady=(20, 0))

    def _build_novelty_row(self, parent, number, head, body):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=8)

        badge = ctk.CTkLabel(row, text=str(number), font=ctk.CTkFont(size=14, weight="bold"),
                              text_color="#FFFFFF", fg_color=RED, corner_radius=999,
                              width=26, height=26)
        badge.pack(side="left", anchor="n", padx=(0, 14))

        text_col = ctk.CTkFrame(row, fg_color="transparent")
        text_col.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(text_col, text=head, font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w")
        ctk.CTkLabel(text_col, text=body, font=ctk.CTkFont(size=13), text_color=MUTED,
                     anchor="w", justify="left", wraplength=820).pack(anchor="w", pady=(2, 0))

        sep = ctk.CTkFrame(parent, fg_color=BORDER, height=1)
        sep.pack(fill="x", pady=(8, 0))

    def _build_pipeline_card(self, parent, col, title, subtitle, body, color):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16,
                             border_width=1.5, border_color=BORDER)
        card.grid(row=0, column=col, sticky="nsew", padx=(0, 10) if col == 0 else (10, 0))
        accent = ctk.CTkFrame(card, fg_color=color, height=7, corner_radius=0)
        accent.pack(fill="x", side="top")
        inner = ctk.CTkFrame(card, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=22, pady=20)
        ctk.CTkLabel(inner, text=title, font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=INK, anchor="w").pack(anchor="w")
        ctk.CTkLabel(inner, text=subtitle, font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=color, anchor="w").pack(anchor="w", pady=(2, 10))
        ctk.CTkLabel(inner, text=body, font=ctk.CTkFont(size=13), text_color=MUTED, anchor="w",
                     justify="left", wraplength=460).pack(anchor="w")


def main():
    ctk.set_appearance_mode("light")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()