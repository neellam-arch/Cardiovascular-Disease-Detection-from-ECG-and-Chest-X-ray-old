"""
train.py  —  ECGNet  9-Label ECG Diagnosis Classifier
=======================================================
Architecture : MultiScaleStem  +  Deep SE-ResNet (20 blocks)  +  Dual-Pool Head
Labels       : 9 focused clinical ECG diagnoses
Loss         : Asymmetric Focal Loss  (gamma_neg=6, gamma_pos=1)
Sampler      : WeightedRandomSampler  (rare-class oversampling)
Augmentation : Mixup 35%  |  Time-shift  |  Amplitude  |  Noise  |
               Baseline wander  |  50 Hz powerline  |  Lead dropout  |  Masking
Scheduler    : OneCycleLR  (10% warm-up + cosine decay)
SWA          : Stochastic Weight Averaging in final 30% of training
Threshold    : Per-class grid search on validation set
CPU-safe     : num_workers=0, all-core threading

Data (priority order):
  1.  Final Dataset/       +  patient_details.csv            (ECG_Diagnosis column)
  2.  sorted_patients/     +  patient_dataset_summary.xlsx   (ECG_Machine_Report column)
  3.  Test/sorted_patients/+  patient_dataset_summary.xlsx   (if moved to Test/)
  4.  Final_Dataset/       +  patient_dataset_summary.xlsx   (7,684-ECG demo subset)

Startup behaviour:
  - Deletes  ecg_model/  to start fresh training

Output  ->  ecg_model/
  best_model.pt         PyTorch checkpoint
  thresholds.json       per-class optimal thresholds
  label_classes.json    ordered label list
  training_history.csv  epoch-level metrics
  test_predictions.csv  sample-level test-set predictions

Usage:
    python train.py
    python train.py --epochs 40 --batch 16 --lr 1e-3
    python train.py --keep-sorted          # skip sorted_patients deletion
"""

import argparse, csv, json, os, re, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    from torch.optim.swa_utils import AveragedModel, update_bn as swa_update_bn
    HAS_SWA = True
except ImportError:
    HAS_SWA = False

# UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# use every available CPU core
torch.set_num_threads(os.cpu_count() or 4)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
OUTPUT_DIR   = BASE / "ecg_model"
CHECKPOINT   = OUTPUT_DIR / "best_model.pt"
LOG_PATH     = OUTPUT_DIR / "training_history.csv"
THRESH_JSON  = OUTPUT_DIR / "thresholds.json"
CLASSES_JSON = OUTPUT_DIR / "label_classes.json"

# Primary source (CSV-based)
PATIENT_DIR  = BASE / "Final Dataset"
DETAILS_CSV  = BASE / "patient_details.csv"

# MIMIC-IV Excel summary (used by fallback sources)
SUMMARY_XL   = BASE / "patient_dataset_summary.xlsx"

# ECG data folder — auto-detected in priority order:
#   1. sorted_patients/        (original location at project root)
#   2. Test/sorted_patients/   (moved to Test folder)
#   3. Final_Dataset/          (7,684-ECG validated demo subset)
SORTED_DIR = next(
    (p for p in [
        BASE / "sorted_patients",
        BASE / "Test" / "sorted_patients",
        BASE / "Final_Dataset",
    ] if p.exists()),
    BASE / "sorted_patients"   # default path shown in error if none found
)

N_LEADS    = 12
N_SAMPLES  = 5000      # 10 s x 500 Hz
TARGET_FS  = 500
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  NINE CLINICAL LABELS
# ─────────────────────────────────────────────────────────────────────────────
ECG_LABEL_RULES = [
    ("ST_Ischemia", [
        "consider acute st elevation", "st elevation",
        "may be due to myocardial ischemia",
        "lateral st-t changes may be due to ischemia",
        "extensive st-t changes may be due",
        "st-t wave abnormality", "st-t changes", "st depression"]),

    ("Atrial_Fibrillation", [
        "atrial fibrillation", "a-fib", "afib", "af "]),

    ("Left_Bundle_Branch", [
        "left bundle branch", "lbbb"]),

    ("Prolonged_QT", [
        "prolonged qt", "long qt", "qt prolongation"]),

    ("Anterior_Infarct", [
        "anterior infarct", "anteroseptal infarct",
        "possible anterior infarct", "possible anteroseptal infarct",
        "septal infarct", "possible septal infarct",
        "anterior mi", "anteroseptal mi"]),

    ("Inferior_Infarct", [
        "inferior infarct", "inferior/lateral st",
        "inferior t wave", "inferior st elev",
        "inferior mi", "inferior wall mi",
        "inferolateral", "infero-lateral",
        "old inferior", "previous inferior mi",
        "inferior st segment", "inferior lead st",
        "inferior q wave", "q in iii", "q in avf"]),

    ("LV_Hypertrophy", [
        "left ventricular hypertrophy", "lvh",
        "possible left ventricular hypertrophy",
        "probable left ventricular hypertrophy",
        "voltage criteria for lvh",
        "lv hypertrophy", "increased voltage",
        "sokolow", "cornell criteria",
        "biventricular hypertrophy",
        "meets voltage criteria", "high voltage",
        "left ventricular enlargement"]),

    ("Right_Bundle_Branch", [
        "right bundle branch", "rbbb"]),

    ("Sinus_Tachycardia", [
        "sinus tachycardia"]),
]

LABEL_COLS     = [name for name, _ in ECG_LABEL_RULES]
N_CLASSES      = len(LABEL_COLS)
DEFAULT_THRESH = 0.40


def extract_labels_series(text_series: pd.Series) -> pd.DataFrame:
    lv = text_series.fillna("").str.lower()
    result = {}
    for col, kws in ECG_LABEL_RULES:
        pat = "|".join(re.escape(k) for k in kws)
        result[col] = lv.str.contains(pat, regex=True).astype(np.float32)
    return pd.DataFrame(result)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SIGNAL  I/O  (direct binary — no wfdb dependency)
# ─────────────────────────────────────────────────────────────────────────────
def read_hea_signal(hea_path: Path) -> np.ndarray:
    """Parse .hea header + read .dat binary -> (N_LEADS, samples) float32."""
    with open(hea_path) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    p       = lines[0].split()
    n_leads = int(p[1]); n_samp = int(p[3])
    gains, baselines = [], []
    for line in lines[1: 1 + n_leads]:
        tok = line.split(); gt = tok[2]
        gains.append(float(re.split(r"[(/]", gt)[0]))
        baselines.append(
            int(gt.split("(")[1].split(")")[0]) if "(" in gt else 0)
    raw = np.fromfile(str(hea_path.with_suffix(".dat")), dtype=np.int16)
    raw = raw[: n_samp * n_leads].reshape(n_samp, n_leads)
    sig = np.zeros((n_leads, n_samp), dtype=np.float32)
    for i in range(n_leads):
        g = gains[i] if gains[i] != 0 else 1.0
        sig[i] = (raw[:, i].astype(np.float32) - baselines[i]) / g
    return sig


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    out = np.zeros_like(sig)
    for i in range(len(sig)):
        s = sig[i].copy()
        q25, q75 = np.percentile(s, [25, 75]); iqr = q75 - q25
        if iqr > 1e-6:
            s = np.clip(s, q25 - 5 * iqr, q75 + 5 * iqr)
        std = s.std()
        out[i] = (s - s.mean()) / std if std > 1e-6 else s
    return out.astype(np.float32)


def pad_or_crop(sig: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    n = sig.shape[1]
    if n >= length:
        return sig[:, :length]
    return np.concatenate(
        [sig, np.zeros((sig.shape[0], length - n), dtype=np.float32)], axis=1)


def safe_load(hea_path: Path) -> np.ndarray:
    try:
        return pad_or_crop(normalize_signal(read_hea_signal(hea_path)))
    except Exception:
        return np.zeros((N_LEADS, N_SAMPLES), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def augment(sig: np.ndarray) -> np.ndarray:
    sig = np.roll(sig, np.random.randint(-300, 300), axis=1)
    sig = sig * np.random.uniform(0.75, 1.25, (sig.shape[0], 1)).astype(np.float32)
    sig = sig + np.random.normal(0, 0.03, sig.shape).astype(np.float32)
    if np.random.random() < 0.55:
        t   = np.linspace(0, 2 * np.pi, sig.shape[1], dtype=np.float32)
        sig = sig + (np.random.uniform(0, 0.15)
                     * np.sin(np.random.uniform(0.03, 0.6) * t
                               + np.random.uniform(0, 2 * np.pi)))[np.newaxis, :]
    if np.random.random() < 0.30:
        t   = np.linspace(0, 2 * np.pi * 50 * 10, sig.shape[1], dtype=np.float32)
        sig = sig + (np.random.uniform(0, 0.06) * np.sin(t))[np.newaxis, :]
    if np.random.random() < 0.20:
        sig[np.random.choice(sig.shape[0],
                              np.random.randint(1, 3), replace=False)] = 0.0
    if np.random.random() < 0.25:
        s = np.random.randint(0, sig.shape[1] - 500)
        sig[:, s: s + np.random.randint(100, 500)] = 0.0
    return sig.astype(np.float32)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.6):
    if np.random.random() > 0.50:
        return x, y
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], lam * y + (1 - lam) * y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, augment_data: bool = False):
        self.df  = df.reset_index(drop=True)
        self.aug = augment_data

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig = safe_load(Path(row["hea_path"]))
        if self.aug:
            sig = augment(sig)
        return (torch.from_numpy(sig),
                torch.tensor(row[LABEL_COLS].values.astype(np.float32)))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL  —  ECGNet
#     MultiScaleStem | 20 SE-ResBlocks | Dual-Pool Head (1024 -> 9)
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        mid = max(ch // r, 8)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.gap(x).squeeze(-1)).unsqueeze(-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.0):
        super().__init__()
        pad        = kernel // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = SEBlock(out_ch)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip  = (nn.Sequential(
                          nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch))
                      if in_ch != out_ch or stride != 1 else nn.Identity())
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.drop(self.se(self.bn2(self.conv2(
                self.act(self.bn1(self.conv1(x)))))))
        return self.act(h + self.skip(x))


class MultiScaleStem(nn.Module):
    """3 parallel conv branches at QRS/P-wave/slow-trend scales -> fuse."""
    def __init__(self, in_ch=N_LEADS, out_ch=64):
        super().__init__()
        mid = out_ch // 3
        def branch(k):
            return nn.Sequential(
                nn.Conv1d(in_ch, mid, k, stride=2, padding=k // 2, bias=False),
                nn.BatchNorm1d(mid), nn.ReLU(inplace=True))
        self.b_s  = branch(7)      # QRS fine detail
        self.b_m  = branch(15)     # P / T wave
        self.b_l  = branch(31)     # slow trends
        self.proj = nn.Sequential(
            nn.Conv1d(mid * 3, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(self.proj(
            torch.cat([self.b_s(x), self.b_m(x), self.b_l(x)], dim=1)))


class ECGNet(nn.Module):
    def __init__(self, n_leads=N_LEADS, n_diag=N_CLASSES):
        super().__init__()
        dr = 0.08
        self.stem   = MultiScaleStem(n_leads, 64)
        self.layer1 = nn.Sequential(
            ResBlock1D(64,  64,  7, dropout=dr), ResBlock1D(64,  64,  7, dropout=dr),
            ResBlock1D(64,  64,  7, dropout=dr), ResBlock1D(64,  64,  7, dropout=dr))
        self.layer2 = nn.Sequential(
            ResBlock1D(64,  128, 7, stride=2, dropout=dr),
            ResBlock1D(128, 128, 7, dropout=dr), ResBlock1D(128, 128, 7, dropout=dr),
            ResBlock1D(128, 128, 7, dropout=dr))
        self.layer3 = nn.Sequential(
            ResBlock1D(128, 256, 7, stride=2, dropout=dr),
            ResBlock1D(256, 256, 7, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr),
            ResBlock1D(256, 256, 7, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr),
            ResBlock1D(256, 256, 7, dropout=dr), ResBlock1D(256, 256, 7, dropout=dr),
            ResBlock1D(256, 256, 7, dropout=dr))
        self.layer4 = nn.Sequential(
            ResBlock1D(256, 512, 7, stride=2, dropout=dr),
            ResBlock1D(512, 512, 7, dropout=dr), ResBlock1D(512, 512, 7, dropout=dr),
            ResBlock1D(512, 512, 7, dropout=dr))
        self.pool_avg = nn.AdaptiveAvgPool1d(1)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(0.50),
            nn.Linear(512,  256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.30),
            nn.Linear(256, n_diag))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        f = torch.cat([self.pool_avg(x),
                       self.pool_max(x)], dim=1).squeeze(-1)
        return self.head(f)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  ASYMMETRIC FOCAL LOSS
# ─────────────────────────────────────────────────────────────────────────────
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg=6.0, gamma_pos=1.0,
                 clip=0.05, smooth=0.03, class_weights=None, eps=1e-8):
        super().__init__()
        self.gn, self.gp = gamma_neg, gamma_pos
        self.clip, self.eps = clip, eps
        self.smooth = smooth; self.cw = class_weights

    def forward(self, logits, targets):
        targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        p   = torch.sigmoid(logits)
        p_m = (p + self.clip).clamp(max=1)
        lp  = targets       * torch.log(p.clamp(min=self.eps))
        ln  = (1 - targets) * torch.log((1 - p_m).clamp(min=self.eps))
        pt  = p * targets + (1 - p_m) * (1 - targets)
        g   = self.gp * targets + self.gn * (1 - targets)
        loss = -((1 - pt) ** g) * (lp + ln)
        if self.cw is not None:
            loss = loss * self.cw
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  WEIGHTED SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
def make_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    labels = df[LABEL_COLS].values.astype(np.float32)
    cw     = 1.0 / (labels.sum(axis=0) + 1.0)
    sw     = np.array([
        cw[np.where(labels[i] > 0)[0]].max() if labels[i].sum() > 0 else cw.min()
        for i in range(len(df))])
    return WeightedRandomSampler(
        torch.FloatTensor(sw), num_samples=len(df), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  METRICS  &  THRESHOLD SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(probs, labels, thresholds=None):
    t     = thresholds or [DEFAULT_THRESH] * N_CLASSES
    preds = np.stack(
        [(probs[:, i] >= t[i]).astype(np.float32) for i in range(N_CLASSES)],
        axis=1)
    per_acc = (preds == labels).mean(axis=0)
    per_f1  = []
    for i in range(N_CLASSES):
        tp = int(((preds[:, i] == 1) & (labels[:, i] == 1)).sum())
        fp = int(((preds[:, i] == 1) & (labels[:, i] == 0)).sum())
        fn = int(((preds[:, i] == 0) & (labels[:, i] == 1)).sum())
        per_f1.append(2 * tp / (2 * tp + fp + fn + 1e-8))
    per_auc = {}
    for i, col in enumerate(LABEL_COLS):
        if 0 < labels[:, i].sum() < len(labels):
            try:
                per_auc[col] = float(roc_auc_score(labels[:, i], probs[:, i]))
            except Exception:
                pass
    mean_auc = float(np.mean(list(per_auc.values()))) if per_auc else 0.0
    return float(per_acc.mean()), per_acc, float(np.mean(per_f1)), per_f1, mean_auc, per_auc


def find_thresholds(probs, labels, min_prec=0.60):
    cands = np.round(np.arange(0.05, 0.95, 0.01), 2)
    out   = []
    print(f"\n  {'Label':<25} {'Thr':>5}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*53}")
    for i, col in enumerate(LABEL_COLS):
        if labels[:, i].sum() == 0:
            out.append(DEFAULT_THRESH)
            print(f"  {col:<25}  (no positives — using {DEFAULT_THRESH:.2f})")
            continue
        best_f1, best_t   = 0.0, DEFAULT_THRESH
        best_pf1, best_pt, found = 0.0, DEFAULT_THRESH, False
        for thr in cands:
            p  = (probs[:, i] >= thr).astype(np.float32)
            tp = int(((p==1) & (labels[:,i]==1)).sum())
            fp = int(((p==1) & (labels[:,i]==0)).sum())
            fn = int(((p==0) & (labels[:,i]==1)).sum())
            pr = tp / (tp + fp + 1e-8)
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1: best_f1, best_t = f1, float(thr)
            if pr >= min_prec and f1 > best_pf1:
                best_pf1, best_pt, found = f1, float(thr), True
        chosen = best_pt if found else best_t
        out.append(chosen)
        p  = (probs[:, i] >= chosen).astype(np.float32)
        tp = int(((p==1) & (labels[:,i]==1)).sum())
        fp = int(((p==1) & (labels[:,i]==0)).sum())
        fn = int(((p==0) & (labels[:,i]==1)).sum())
        print(f"  {col:<25} {chosen:>5.2f}  "
              f"{2*tp/(2*tp+fp+fn+1e-8):>6.4f}  "
              f"{tp/(tp+fp+1e-8):>6.4f}  "
              f"{tp/(tp+fn+1e-8):>6.4f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 10. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, opt, sched, epoch, total):
    model.train()
    total_loss = 0.0
    n = len(loader)
    for step, (x, y) in enumerate(loader, 1):
        x, y   = x.to(DEVICE), y.to(DEVICE)
        x, y   = mixup(x, y)
        opt.zero_grad(set_to_none=True)
        loss   = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        total_loss += loss.item()
        if step % 5 == 0 or step == n:
            print(f"  Ep{epoch}/{total} [{step:>4}/{n}]  "
                  f"loss={total_loss/step:.4f}", end="\r", flush=True)
    print()
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    return (total_loss / len(loader),
            np.concatenate(all_probs),
            np.concatenate(all_labels))


# ─────────────────────────────────────────────────────────────────────────────
# 11. DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _load_from_csv() -> pd.DataFrame:
    df = pd.read_csv(DETAILS_CSV)
    df.columns = df.columns.str.strip()

    for cand in ("ECG_Diagnosis", "ECG_Machine_Report", "Diagnosis", "ecg_diagnosis"):
        if cand in df.columns:
            diag_col = cand; break
    else:
        raise SystemExit(f"No diagnosis column in {DETAILS_CSV}. Columns: {list(df.columns)}")

    label_df = extract_labels_series(df[diag_col])
    for col in LABEL_COLS:
        df[col] = label_df[col]

    # find folder / file columns
    f_col = next((c for c in ("Patient_Folder","patient_folder","Folder") if c in df.columns), None)
    e_col = next((c for c in ("ECG_File","ecg_file","File") if c in df.columns), None)
    if not f_col or not e_col:
        raise SystemExit(f"Cannot find Patient_Folder/ECG_File columns in {DETAILS_CSV}")

    df["hea_path"] = df.apply(
        lambda r: str(PATIENT_DIR / str(r[f_col])
                      / r[e_col].replace(".dat", ".hea")), axis=1)
    df = df[df["hea_path"].map(
        lambda p: Path(p).exists() and Path(p).with_suffix(".dat").exists()
    )].reset_index(drop=True)

    s_col = next((c for c in ("Subject_ID","subject_id","SubjectID") if c in df.columns), None)
    df["subject_id"] = df[s_col].astype(str) if s_col else df.index.astype(str)
    return df


def _load_from_xlsx() -> pd.DataFrame:
    xl = pd.read_excel(SUMMARY_XL, engine="openpyxl",
                       usecols=["subject_id","ecg_study_id","ECG_Machine_Report"])
    xl["subject_id"]   = xl["subject_id"].astype(str)
    xl["ecg_study_id"] = xl["ecg_study_id"].astype(str)

    label_df = extract_labels_series(xl["ECG_Machine_Report"])
    for col in LABEL_COLS:
        xl[col] = label_df[col]

    records = []
    for _, row in xl.iterrows():
        hea = SORTED_DIR / f"patient_{row['subject_id']}" / "ecg" / f"{row['ecg_study_id']}.hea"
        if hea.exists() and hea.with_suffix(".dat").exists():
            r = row.to_dict(); r["hea_path"] = str(hea); records.append(r)
    return pd.DataFrame(records).reset_index(drop=True)


def build_loaders(batch_size: int, min_samples: int):
    # pick source
    if DETAILS_CSV.exists() and PATIENT_DIR.exists():
        print(f"  Data source : {PATIENT_DIR.name}/  +  {DETAILS_CSV.name}")
        df = _load_from_csv()
    elif SUMMARY_XL.exists() and SORTED_DIR.exists():
        print(f"  Data source : {SORTED_DIR.name}/  +  {SUMMARY_XL.name}  [{SORTED_DIR}]")
        df = _load_from_xlsx()
    else:
        raise SystemExit(
            "No ECG data found. Checked these locations:\n"
            f"  {PATIENT_DIR}  +  {DETAILS_CSV}\n"
            f"  {BASE / 'sorted_patients'}  +  {SUMMARY_XL}\n"
            f"  {BASE / 'Test' / 'sorted_patients'}  +  {SUMMARY_XL}\n"
            f"  {BASE / 'Final_Dataset'}  +  {SUMMARY_XL}\n"
            "Make sure patient_dataset_summary.xlsx is in the project root.")

    print(f"  ECG files matched : {len(df):,}")
    if len(df) == 0:
        raise SystemExit("No matched ECG files. Check data directory structure.")

    # label stats
    print(f"\n  {'Label':<25}  {'Count':>6}  {'%':>5}")
    print(f"  {'-'*40}")
    drop_cols = []
    for col in LABEL_COLS:
        cnt = int(df[col].sum()); pct = 100 * cnt / len(df)
        flag = "  [TOO RARE — zeroed]" if cnt < min_samples else ""
        print(f"  {col:<25}  {cnt:>6}  {pct:>4.1f}%{flag}")
        if cnt < min_samples:
            drop_cols.append(col)
    if drop_cols:
        for col in drop_cols:
            df[col] = 0.0

    # keep positives + 40% negatives (more negatives → better specificity)
    has_lbl = df[LABEL_COLS].sum(axis=1) > 0
    df_neg  = df[~has_lbl].sample(frac=0.40, random_state=42)
    df      = pd.concat([df[has_lbl], df_neg], ignore_index=True)
    print(f"\n  Dataset : {len(df):,}  (pos={int(has_lbl.sum())}  neg_30%={len(df_neg)})")

    # 70/15/15 patient-level split
    pats = list(df["subject_id"].unique())   # list() avoids ArrowStringArray shuffle warning
    np.random.seed(42); np.random.shuffle(pats)
    n_test = max(1, int(len(pats) * 0.15))
    n_val  = max(1, int(len(pats) * 0.15))
    p_test = set(pats[:n_test])
    p_val  = set(pats[n_test: n_test + n_val])
    p_train= set(pats[n_test + n_val:])

    tr = df[df["subject_id"].isin(p_train)].reset_index(drop=True)
    vl = df[df["subject_id"].isin(p_val)  ].reset_index(drop=True)
    te = df[df["subject_id"].isin(p_test) ].reset_index(drop=True)
    print(f"  Split 70/15/15  train:{len(tr):,}  val:{len(vl):,}  test:{len(te):,}")

    freq = tr[LABEL_COLS].values.sum(0) / max(len(tr), 1)
    cw   = (1.0 / np.sqrt(freq + 0.01)).astype(np.float32)
    cw  /= cw.mean()
    # Manually amplify the two POOR-performing labels so loss focuses on them
    LABEL_BOOST = {
        "Inferior_Infarct": 5.0,
        "LV_Hypertrophy":   5.0,
        "Anterior_Infarct": 2.5,
        "ST_Ischemia":      1.8,
        "Prolonged_QT":     1.8,
    }
    for i, col in enumerate(LABEL_COLS):
        if col in LABEL_BOOST:
            cw[i] *= LABEL_BOOST[col]
    cw  /= cw.mean()   # re-normalise after boost
    cw_t = torch.tensor(cw, device=DEVICE).unsqueeze(0)
    print(f"\n  Loss weights : " + "  ".join(f"{c:.2f}" for c in cw))

    tdl  = DataLoader(ECGDataset(tr, True),  batch_size,
                      sampler=make_sampler(tr), num_workers=0, drop_last=True)
    vdl  = DataLoader(ECGDataset(vl, False), batch_size,
                      shuffle=False, num_workers=0)
    tsdl = DataLoader(ECGDataset(te, False), batch_size,
                      shuffle=False, num_workers=0)
    return tdl, vdl, tsdl, cw_t, te


# ─────────────────────────────────────────────────────────────────────────────
# 12. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="ECGNet 9-class trainer (CPU-safe)")
    ap.add_argument("--epochs",      type=int,   default=120)
    ap.add_argument("--batch",       type=int,   default=8,
                    help="8 for CPU; 32+ for GPU")
    ap.add_argument("--lr",          type=float, default=8e-4)
    ap.add_argument("--min-samples", type=int,   default=30)
    ap.add_argument("--min-prec",    type=float, default=0.45)
    ap.add_argument("--keep-sorted", action="store_true",
                    help="(legacy flag — no longer deletes sorted_patients/)")
    args = ap.parse_args()

    print("=" * 65)
    print("ECGNet  —  9-Label ECG Diagnosis Classifier")
    print(f"Device   : {DEVICE}  ({os.cpu_count()} CPU cores)")
    print(f"Epochs   : {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")
    print(f"SWA      : {HAS_SWA}  |  Mixup 35%  |  ASL gamma_neg=6  |  LS=0.03")
    print("=" * 65)

    # ── clean up old model directory ─────────────────────────────────────────
    if OUTPUT_DIR.exists():
        print(f"\nClearing old ecg_model/ for fresh training...")
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)

    with open(CLASSES_JSON, "w") as f:
        json.dump(LABEL_COLS, f, indent=2)

    # ── data ─────────────────────────────────────────────────────────────────
    print("\nBuilding dataset...")
    tdl, vdl, tsdl, cw, test_df = build_loaders(args.batch, args.min_samples)

    # ── model ─────────────────────────────────────────────────────────────────
    model = ECGNet(N_LEADS, N_CLASSES).to(DEVICE)
    print(f"\nModel    : ECGNet  |  "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    crit  = AsymmetricFocalLoss(8.0, 1.0, 0.05, 0.02, cw)
    opt   = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    sched = OneCycleLR(opt, max_lr=args.lr,
                       steps_per_epoch=len(tdl),
                       epochs=args.epochs, pct_start=0.10)

    use_swa   = HAS_SWA and args.epochs >= 6
    swa_m     = AveragedModel(model) if use_swa else None
    swa_start = max(1, int(args.epochs * 0.70))

    log_fields = (["epoch","train_loss","val_loss","mean_acc","mean_f1","mean_auc"]
                  + [f"f1_{c}"  for c in LABEL_COLS]
                  + [f"auc_{c}" for c in LABEL_COLS])
    with open(LOG_PATH, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_auc = 0.0; pat_cnt = 0; PATIENCE = 18
    best_thr = [DEFAULT_THRESH] * N_CLASSES

    print(f"\n{'='*65}")
    print(f"Training  {args.epochs} epochs  |  early-stop patience={PATIENCE}")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tl = train_epoch(model, tdl, crit, opt, sched, epoch, args.epochs)
        vl_loss, probs, labels = evaluate(model, vdl, crit)
        acc, per_acc, f1, per_f1, auc, per_auc = compute_metrics(
            probs, labels, best_thr)

        if use_swa and epoch >= swa_start:
            swa_m.update_parameters(model)

        print(f"\nEpoch {epoch:02d}/{args.epochs}  ({int(time.time()-t0)}s)")
        print(f"  train={tl:.4f}  val={vl_loss:.4f}  "
              f"Acc={acc*100:.1f}%  F1={f1:.4f}  AUC={auc:.4f}")
        print(f"\n  {'Label':<25} {'Acc%':>5}  {'F1':>6}  {'AUC':>6}")
        print(f"  {'-'*44}")
        for i, col in enumerate(LABEL_COLS):
            auc_s = f"{per_auc[col]:.4f}" if col in per_auc else "   N/A"
            print(f"  {col:<25} {per_acc[i]*100:>4.1f}%  {per_f1[i]:>6.4f}  {auc_s}")

        row = {"epoch": epoch, "train_loss": tl, "val_loss": vl_loss,
               "mean_acc": acc, "mean_f1": f1, "mean_auc": auc,
               **{f"f1_{c}":  per_f1[i]           for i, c in enumerate(LABEL_COLS)},
               **{f"auc_{c}": per_auc.get(c, "")  for c     in LABEL_COLS}}
        with open(LOG_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        if auc > best_auc:
            best_auc = auc; pat_cnt = 0
            print(f"\n  Threshold search...")
            best_thr = find_thresholds(probs, labels, args.min_prec)
            torch.save({
                "epoch":      epoch,
                "model":      {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "mean_auc":   auc,
                "mean_f1":    f1,
                "label_cols": LABEL_COLS,
                "thresholds": best_thr,
                "n_leads":    N_LEADS,
                "n_samples":  N_SAMPLES,
            }, CHECKPOINT)
            with open(THRESH_JSON, "w") as f:
                json.dump(dict(zip(LABEL_COLS, best_thr)), f, indent=2)
            print(f"  [SAVED]  AUC={auc:.4f}  F1={f1:.4f}")
        else:
            pat_cnt += 1
            print(f"  [no improvement]  patience {pat_cnt}/{PATIENCE}")
            if pat_cnt >= PATIENCE:
                print(f"\nEarly stopping — best val AUC = {best_auc:.4f}")
                break

    # SWA
    if use_swa and swa_m is not None:
        print("\nFinalising SWA weights...")
        swa_update_bn(tdl, swa_m, device=DEVICE)
        _, p_s, l_s = evaluate(swa_m, vdl, crit)
        _, _, _, _, auc_s, _ = compute_metrics(p_s, l_s, best_thr)
        print(f"  SWA AUC={auc_s:.4f}  (single best={best_auc:.4f})")
        if auc_s > best_auc:
            best_thr = find_thresholds(p_s, l_s, args.min_prec)
            swa_sd   = {k.replace("module.", ""): v.cpu()
                        for k, v in swa_m.state_dict().items()
                        if k != "n_averaged"}
            torch.save({
                "epoch":      args.epochs, "model": swa_sd,
                "mean_auc":   auc_s,       "mean_f1": 0.0,
                "label_cols": LABEL_COLS,  "thresholds": best_thr,
                "n_leads":    N_LEADS,     "n_samples": N_SAMPLES,
            }, CHECKPOINT)
            with open(THRESH_JSON, "w") as f:
                json.dump(dict(zip(LABEL_COLS, best_thr)), f, indent=2)
            print(f"  [SAVED SWA]  AUC={auc_s:.4f}")
            best_auc = auc_s

    # test set
    print(f"\n{'='*65}\nFINAL TEST SET")
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    _, tp, tl2 = evaluate(model, tsdl, crit)
    _, _, f1, per_f1, auc, per_auc = compute_metrics(tp, tl2, best_thr)
    print(f"\n  {'Label':<25} {'F1':>6}  {'AUC':>6}")
    print(f"  {'-'*40}")
    for i, col in enumerate(LABEL_COLS):
        auc_s = f"{per_auc[col]:.4f}" if col in per_auc else "   N/A"
        print(f"  {col:<25} {per_f1[i]:>6.4f}  {auc_s}")
    print(f"\n  Mean F1={f1:.4f}   Mean AUC={auc:.4f}")

    rows = []
    for i in range(len(test_df)):
        r = {"subject_id": test_df.iloc[i]["subject_id"],
             "hea_path":   test_df.iloc[i]["hea_path"]}
        for j, col in enumerate(LABEL_COLS):
            r[f"actual_{col}"]    = int(test_df.iloc[i][col])
            r[f"predicted_{col}"] = int(tp[i, j] >= best_thr[j])
            r[f"prob_{col}"]      = round(float(tp[i, j]), 4)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print(f"\n{'='*65}")
    print(f"Done  ->  {OUTPUT_DIR}")
    print(f"Best AUC : {best_auc:.4f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
