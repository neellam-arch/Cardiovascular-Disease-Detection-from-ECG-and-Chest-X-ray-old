"""
xray_train.py  —  Fast CPU Chest X-ray Classifier
==================================================
Strategy : Frozen DenseNet-121 backbone pretrained on chest X-ray datasets
           (NIH, PadChest, CheXpert, MIMIC-CXR) via torchxrayvision.
           Only the small classification head is trained — no backbone backprop.

Phase 1  : Extract features from every image ONCE and cache to disk.
           (~20-30 min on CPU, never repeated unless you delete the cache)
Phase 2  : Train the 3-class head on cached features.
           Each epoch runs in seconds. 50 epochs ~ 5 minutes total.

Targets  : Cardiomegaly, Edema, Pleural Effusion
Split    : Patient-level 70 / 15 / 15  (seed=42)

Run      : python xray_train.py
Calibrate: python xray_train.py --calibrate

Outputs (xray_model/)
  best_model.pt        best val-AUC checkpoint (backbone + head)
  thresholds.json      per-class Youden-J thresholds (val set only)
  label_classes.json   ["Cardiomegaly","Edema","Pleural Effusion"]
  config.json          architecture info (read by xray_main.py)
  training_history.csv epoch metrics
"""

import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("Run:  pip install scikit-learn")

try:
    import torchxrayvision as txrv
except ImportError:
    sys.exit("Run:  pip install torchxrayvision")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
FINAL     = ROOT / "Final_Dataset"
SUMMARY   = ROOT / "final_patient_dataset_summary.xlsx"
_colab    = Path("/content/drive/MyDrive")
MODEL_DIR = (_colab / "xray_model") if _colab.exists() else (ROOT / "xray_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = MODEL_DIR / "feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── targets (3 best-performing conditions) ────────────────────────────────────
LABEL_COLS  = ["Cardiomegaly", "Edema", "Pleural Effusion"]
NUM_CLASSES = 3
IMG_SIZE    = 224
BACKBONE    = "densenet121-res224-all"
IN_FEAT     = 1024   # DenseNet-121 pooled feature size

# ── model ─────────────────────────────────────────────────────────────────────
class XrayNet(nn.Module):
    """Frozen DenseNet-121 backbone (chest X-ray pretrained) + trainable head."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = BACKBONE if pretrained else None
        self.backbone = txrv.models.DenseNet(weights=weights)

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(IN_FEAT, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(512, 256),     nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self._extract(x))

    def _extract(self, x):
        feat_map = self.backbone.features(x)
        feat_map = F.relu(feat_map, inplace=True)
        pooled   = F.adaptive_avg_pool2d(feat_map, (1, 1))
        return pooled.view(pooled.size(0), -1)


# ── patient split ─────────────────────────────────────────────────────────────
def make_splits(df):
    patients = df["subject_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(patients)
    n = len(patients)
    train_p = set(patients[:int(0.70 * n)])
    val_p   = set(patients[int(0.70 * n):int(0.85 * n)])
    test_p  = set(patients[int(0.85 * n):])
    return (df[df.subject_id.isin(train_p)].copy(),
            df[df.subject_id.isin(val_p)].copy(),
            df[df.subject_id.isin(test_p)].copy(),
            train_p, val_p, test_p)


# ── image dataset (used only during feature extraction) ───────────────────────
class XrayImageDS(Dataset):
    def __init__(self, df):
        self.samples = []
        missing = 0
        for _, r in df.iterrows():
            sid = str(int(r["subject_id"]))
            xray_dir = FINAL / f"patient_{sid}" / "xray"
            if not xray_dir.is_dir():
                missing += 1
                continue
            found = None
            for ext in ["png", "jpg", "jpeg"]:
                imgs = list(xray_dir.glob(f"*.{ext}"))
                if imgs:
                    found = imgs[0]
                    break
            if found is None:
                missing += 1
                continue
            lbl = []
            for col in LABEL_COLS:
                v = r[col]
                lbl.append(float(v) if (not pd.isna(v) and float(v) >= 0) else -1.0)
            self.samples.append((str(found), lbl))
        if missing:
            print(f"  (skipped {missing} rows with no image)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, lbl = self.samples[i]
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
        arr = (arr / 255.0) * 2048.0 - 1024.0   # torchxrayvision expected range
        tensor = torch.from_numpy(arr).unsqueeze(0)   # (1, H, W)
        return tensor, torch.tensor(lbl, dtype=torch.float32)


# ── phase 1: extract & cache features ────────────────────────────────────────
def extract_and_cache(backbone_model, df, split_name, device):
    cache_f = CACHE_DIR / f"{split_name}_features.npy"
    cache_t = CACHE_DIR / f"{split_name}_targets.npy"

    if cache_f.exists() and cache_t.exists():
        print(f"  [{split_name}] Loading cached features ...")
        return np.load(cache_f), np.load(cache_t)

    print(f"  [{split_name}] Extracting features from images ...")
    ds     = XrayImageDS(df)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_feats, all_tgts = [], []
    backbone_model.eval()
    t0 = time.time()
    for step, (imgs, tgts) in enumerate(loader, 1):
        imgs = imgs.to(device)
        with torch.no_grad():
            feats = backbone_model._extract(imgs).cpu().numpy()
        all_feats.append(feats)
        all_tgts.append(tgts.numpy())
        if step % max(1, len(loader) // 5) == 0 or step == len(loader):
            pct = 100 * step / len(loader)
            print(f"    {pct:5.1f}%  ({step}/{len(loader)} batches)  "
                  f"[{time.time()-t0:.0f}s elapsed]", end="\r", flush=True)
    print()

    feats = np.vstack(all_feats)
    tgts  = np.vstack(all_tgts)
    np.save(cache_f, feats)
    np.save(cache_t, tgts)
    print(f"  [{split_name}] Cached {feats.shape[0]} samples  "
          f"({feats.shape[1]}-dim features)  [{time.time()-t0:.0f}s]")
    return feats, tgts


# ── phase 2: head training on cached features ─────────────────────────────────
def make_feature_loader(feats, tgts, batch_size=256, shuffle=True):
    ft = torch.from_numpy(feats).float()
    tt = torch.from_numpy(tgts).float()
    return DataLoader(TensorDataset(ft, tt),
                      batch_size=batch_size, shuffle=shuffle)


def masked_bce(logits, targets, pos_weight=None):
    mask        = (targets >= 0).float()                  # (B, C)
    tgt_safe    = targets.clamp(min=0.0)                  # replace -1 with 0

    # per-element loss, shape (B, C)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, tgt_safe, reduction="none")

    # apply pos_weight manually so shape stays (B, C)
    if pos_weight is not None:
        pw = pos_weight.to(logits.device)                 # (C,)
        w  = torch.where(tgt_safe > 0.5, pw, torch.ones_like(pw))
        loss = loss * w

    loss = loss * mask                                    # zero masked entries
    n_valid = mask.sum()
    return loss.sum() / n_valid if n_valid > 0 else loss.sum() * 0.0


# ── Youden's J threshold search ───────────────────────────────────────────────
def youden_threshold(probs, targets):
    best_j, best_thr = -1.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        pred = (probs >= thr).astype(int)
        tp = ((pred == 1) & (targets == 1)).sum()
        fp = ((pred == 1) & (targets == 0)).sum()
        fn = ((pred == 0) & (targets == 1)).sum()
        tn = ((pred == 0) & (targets == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, round(float(thr), 3)
    return best_thr


# ── per-condition metrics ─────────────────────────────────────────────────────
def report(probs, tgts, thresholds, title):
    print(f"\n  {title}")
    print(f"  {'Condition':<22} {'AUC':>6} {'Acc%':>6} {'Sens%':>7} "
          f"{'Spec%':>7} {'F1%':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("  " + "-" * 80)
    aucs = []
    for i, col in enumerate(LABEL_COLS):
        mask = tgts[:, i] >= 0
        t    = tgts[mask, i].astype(int)
        p    = probs[mask, i]
        thr  = thresholds.get(col, 0.5)
        pred = (p >= thr).astype(int)
        tp = int(((pred == 1) & (t == 1)).sum())
        fp = int(((pred == 1) & (t == 0)).sum())
        fn = int(((pred == 0) & (t == 1)).sum())
        tn = int(((pred == 0) & (t == 0)).sum())
        sens = tp / (tp + fn + 1e-9) * 100
        spec = tn / (tn + fp + 1e-9) * 100
        prec = tp / (tp + fp + 1e-9) * 100
        f1   = 2 * prec * sens / (prec + sens + 1e-9)
        acc  = (tp + tn) / (tp + fp + fn + tn + 1e-9) * 100
        try:
            auc = roc_auc_score(t, p)
            aucs.append(auc)
            auc_s = f"{auc:.3f}"
        except Exception:
            auc_s = "  N/A"
        print(f"  {col:<22} {auc_s:>6} {acc:>6.1f} {sens:>7.1f} "
              f"{spec:>7.1f} {f1:>6.1f} {tp:>5} {fp:>5} {fn:>5} {tn:>5}")
    if aucs:
        print(f"  {'Macro avg':<22} {np.mean(aucs):.3f}")
    return aucs


# ── calibrate thresholds (val set only) ──────────────────────────────────────
def calibrate(val_probs, val_tgts):
    thresholds = {}
    print("\n  Threshold calibration (Youden's J on val set):")
    for i, col in enumerate(LABEL_COLS):
        mask = val_tgts[:, i] >= 0
        t    = val_tgts[mask, i]
        p    = val_probs[mask, i]
        thr  = youden_threshold(p, t)
        thresholds[col] = thr
        print(f"    {col:<22}  threshold = {thr:.3f}")
    return thresholds


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true",
                    help="Skip training; re-calibrate thresholds on val set")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--batch",  type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  XrayNet  —  {BACKBONE}  (frozen backbone)")
    print(f"  Device   : {device}")
    print(f"  Targets  : {LABEL_COLS}")
    print(f"  Save to  : {MODEL_DIR}")
    print(f"{'='*60}\n")

    # ── load dataset ──────────────────────────────────────────────────────────
    print("  Loading dataset ...")
    df = pd.read_excel(SUMMARY, engine="openpyxl")
    df = df[df["cxr_data_available"] == True].copy()
    df_train, df_val, df_test, *_ = make_splits(df)
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  "
          f"Test: {len(df_test):,}\n")

    # ── build model ───────────────────────────────────────────────────────────
    model = XrayNet(num_classes=NUM_CLASSES, pretrained=True).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,} total  |  {trainable:,} trainable (head only)\n")

    # ── phase 1: feature extraction ───────────────────────────────────────────
    print("  PHASE 1 — Feature extraction (runs once, cached after)")
    print("  This takes ~20-30 min on CPU the first time.\n")
    train_f, train_t = extract_and_cache(model, df_train, "train", device)
    val_f,   val_t   = extract_and_cache(model, df_val,   "val",   device)
    test_f,  test_t  = extract_and_cache(model, df_test,  "test",  device)

    if args.calibrate:
        print("\n  --calibrate mode: loading existing checkpoint ...")
        ckpt = MODEL_DIR / "best_model.pt"
        if not ckpt.exists():
            sys.exit("  ERROR: best_model.pt not found. Run training first.")
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()
        val_probs  = _infer_from_features(model, val_f,  device)
        test_probs = _infer_from_features(model, test_f, device)
        thresholds = calibrate(val_probs, val_t)
        report(test_probs, test_t, thresholds, "TEST SET (after calibration)")
        _save_outputs(thresholds)
        return

    # ── phase 2: head training ─────────────────────────────────────────────────
    print(f"\n  PHASE 2 — Head training  ({args.epochs} epochs, lr={args.lr})")
    print("  Each epoch takes seconds. Total ~5-10 minutes.\n")

    # pos_weight per condition
    pos_w = torch.ones(NUM_CLASSES)
    for i, col in enumerate(LABEL_COLS):
        vals  = df_train[col].dropna()
        n_pos = int((vals == 1).sum())
        n_neg = int((vals == 0).sum())
        if n_pos > 0 and n_neg > 0:
            pos_w[i] = float(np.clip(np.sqrt(float(n_pos) / float(n_neg)), 1.0, 4.0))
    print(f"  pos_weight: { {c: round(float(pos_w[i]),2) for i,c in enumerate(LABEL_COLS)} }\n")

    optimizer = Adam(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    train_loader = make_feature_loader(train_f, train_t, args.batch, shuffle=True)
    val_loader   = make_feature_loader(val_f,   val_t,   args.batch, shuffle=False)

    best_auc  = 0.0
    best_pt   = MODEL_DIR / "best_model.pt"
    csv_path  = MODEL_DIR / "training_history.csv"

    # use a temp name if the file is locked (e.g. open in Excel)
    try:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_auc","lr"])
    except PermissionError:
        csv_path = MODEL_DIR / "training_history_new.csv"
        print(f"  training_history.csv is locked — writing to {csv_path.name}")
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_auc","lr"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for feats, tgts in train_loader:
            feats, tgts = feats.to(device), tgts.to(device)
            optimizer.zero_grad()
            logits = model.head(feats)
            loss   = masked_bce(logits, tgts, pos_w)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * feats.size(0)
            n        += feats.size(0)
        scheduler.step()

        # validation
        model.eval()
        val_loss, vn = 0.0, 0
        all_p, all_t = [], []
        with torch.no_grad():
            for feats, tgts in val_loader:
                feats, tgts = feats.to(device), tgts.to(device)
                logits = model.head(feats)
                val_loss += masked_bce(logits, tgts).item() * feats.size(0)
                vn       += feats.size(0)
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_t.append(tgts.cpu().numpy())

        vp = np.vstack(all_p); vt = np.vstack(all_t)
        aucs = []
        for i in range(NUM_CLASSES):
            mask = vt[:, i] >= 0
            try:
                aucs.append(roc_auc_score(vt[mask, i], vp[mask, i]))
            except Exception:
                pass
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:>3}/{args.epochs}  "
              f"train={run_loss/n:.4f}  val={val_loss/vn:.4f}  "
              f"AUC={val_auc:.4f}  [{time.time()-t0:.1f}s]  lr={lr_now:.1e}")

        try:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{run_loss/n:.4f}",
                                        f"{val_loss/vn:.4f}", f"{val_auc:.4f}",
                                        f"{lr_now:.2e}"])
        except PermissionError:
            pass

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_pt)
            print(f"  *** saved best_model.pt  (AUC {best_auc:.4f})")

    # ── load best, calibrate, evaluate ────────────────────────────────────────
    print(f"\n  Loading best checkpoint (AUC {best_auc:.4f}) ...")
    model.load_state_dict(
        torch.load(best_pt, map_location=device, weights_only=False))
    model.eval()

    val_probs  = _infer_from_features(model, val_f,  device)
    test_probs = _infer_from_features(model, test_f, device)

    thresholds = calibrate(val_probs, val_t)
    report(test_probs, test_t, thresholds, "TEST SET RESULTS")
    _save_outputs(thresholds)

    print(f"\n{'='*60}")
    print(f"  Training complete.  Saved to: {MODEL_DIR}")
    print(f"{'='*60}\n")


def _infer_from_features(model, feats, device, batch=512):
    loader = make_feature_loader(feats,
                                 np.zeros((len(feats), NUM_CLASSES)),
                                 batch, shuffle=False)
    all_p = []
    model.eval()
    with torch.no_grad():
        for f, _ in loader:
            all_p.append(torch.sigmoid(model.head(f.to(device))).cpu().numpy())
    return np.vstack(all_p)


def _save_outputs(thresholds):
    (MODEL_DIR / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8")
    (MODEL_DIR / "label_classes.json").write_text(
        json.dumps(LABEL_COLS, indent=2), encoding="utf-8")
    (MODEL_DIR / "config.json").write_text(
        json.dumps({"backbone": BACKBONE,
                    "img_size": IMG_SIZE,
                    "num_classes": NUM_CLASSES}, indent=2),
        encoding="utf-8")
    print("\n  Saved: thresholds.json  label_classes.json  config.json")


if __name__ == "__main__":
    main()
