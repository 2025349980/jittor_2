#!/usr/bin/env python
"""
PCT (Point Cloud Transformer) for ModelNet40 Classification - 示例代码

本代码提供了基于 Jittor 框架的 PCT 模型，用于 ModelNet40 三维形状分类任务。
选手需要完成标注为 TODO 的部分，训练模型并生成测试集预测结果 result.json。

Usage:
    python pct.py

依赖安装:
    pip install jittor
"""

import os
import json
import math
import time
import argparse
import zipfile
import numpy as np

# Jittor CUDA 12.2 does not support the system default GCC 13.
if 'cc_path' not in os.environ and os.path.exists('/usr/bin/g++-12'):
    os.environ['cc_path'] = '/usr/bin/g++-12'

import jittor as jt
from jittor import nn
from jittor.dataset import Dataset


# ============================================================
# 数据集
# ============================================================

class ModelNet40Dataset(Dataset):
    """ModelNet40 点云数据集。

    加载预处理好的 npy 文件。
    - 训练集: data/train_points.npy + data/train_labels.npy
    - 测试集: data/test_points.npy (无标签)
    """

    def __init__(self, data_dir='./data', split='train', n_points=1024,
                 augment=False, batch_size=32, shuffle=False, num_workers=4,
                 indices=None, random_sample=True, augment_level='light'):
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.n_points = n_points
        self.augment = augment
        self.split = split
        self.random_sample = random_sample
        self.augment_level = augment_level

        data_split = 'train' if split in ('train', 'val') else split
        pts_path = os.path.join(data_dir, f'{data_split}_points.npy')
        assert os.path.exists(pts_path), f"{pts_path} not found."

        self.point_clouds = np.load(pts_path)  # (N, 2048, 3)
        self.n_cached = self.point_clouds.shape[1]

        if data_split == 'train':
            lbl_path = os.path.join(data_dir, f'{data_split}_labels.npy')
            assert os.path.exists(lbl_path), f"{lbl_path} not found."
            self.labels = np.load(lbl_path)  # (N,)
        else:
            self.labels = None  # 测试集不提供标签

        if indices is None:
            self.indices = np.arange(len(self.point_clouds), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        self.total_len = len(self.indices)
        self.set_attrs(total_len=self.total_len)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        pts = self.point_clouds[real_idx]
        replace = self.n_cached < self.n_points
        if self.random_sample or replace:
            choice = np.random.choice(self.n_cached, self.n_points, replace=replace)
        else:
            choice = np.linspace(0, self.n_cached - 1, self.n_points, dtype=np.int64)
        points = pts[choice].copy()

        if self.augment and self.augment_level != 'off':
            # 数据增强策略
            if self.augment_level == 'strong':
                # 1. 随机旋转（绕Y轴）
                theta = np.random.uniform(0, 2 * np.pi)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R = np.array([[cos_t, 0, sin_t],
                              [0, 1, 0],
                              [-sin_t, 0, cos_t]], dtype=np.float32)
                points = points @ R.T

            # 2. 随机缩放
            scale_low, scale_high = (0.8, 1.2) if self.augment_level == 'strong' else (0.9, 1.1)
            scale = np.random.uniform(scale_low, scale_high)
            points = points * scale

            # 3. 随机抖动（添加高斯噪声）
            jitter_std = 0.01 if self.augment_level == 'strong' else 0.005
            jitter = np.random.normal(0, jitter_std, points.shape).astype(np.float32)
            points = points + jitter

            # 4. 随机平移
            shift_range = 0.1 if self.augment_level == 'strong' else 0.05
            shift = np.random.uniform(-shift_range, shift_range, (1, 3)).astype(np.float32)
            points = points + shift

            # 5. 强增强才启用随机dropout点，避免早期训练被压得太慢。
            if self.augment_level == 'strong' and np.random.random() > 0.5:
                dropout_ratio = np.random.uniform(0, 0.5)  # 4GB显存训练优先稳，不做过强dropout
                drop_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0]
                if len(drop_idx) > 0:
                    points[drop_idx] = points[0]  # 用第一个点替代

        if self.labels is not None:
            return points.astype(np.float32), self.labels[real_idx]
        else:
            return points.astype(np.float32), real_idx  # 测试集返回样本编号


# ============================================================
# PCT 模型
# ============================================================

class SA_Layer(nn.Module):
    """Self-Attention layer for PCT."""

    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def execute(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # (B, N, C//4)
        x_k = self.k_conv(x)                     # (B, C//4, N)
        x_v = self.v_conv(x)                     # (B, C, N)
        energy = jt.nn.bmm(x_q, x_k)            # (B, N, N)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = jt.nn.bmm(x_v, attention)          # (B, C, N)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class PCT(nn.Module):
    """Point Cloud Transformer for classification.

    Input:  (B, 3, N) point cloud
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes=40):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 128, 1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = SA_Layer(128)
        self.sa2 = SA_Layer(128)
        self.sa3 = SA_Layer(128)
        self.sa4 = SA_Layer(128)

        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(scale=0.2))

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def execute(self, x):
        B, _, N = x.shape
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))

        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)

        x = jt.concat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x = self.conv_fuse(x)                      # (B, 1024, N)
        x = jt.max(x, dim=2)                       # (B, 1024)

        x = nn.relu(self.bn_fc1(self.fc1(x)))
        x = self.dp1(x)
        x = nn.relu(self.bn_fc2(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)
        return x


class PointNet(nn.Module):
    """轻量 PointNet 分类器，训练更快，适合作为 80% 过线稳态基线。"""

    def __init__(self, num_classes=40):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, 1, bias=False)
        self.conv5 = nn.Conv1d(256, 1024, 1, bias=False)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=0.3)
        self.dp2 = nn.Dropout(p=0.3)

    def execute(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = nn.relu(self.bn2(self.conv2(x)))
        x = nn.relu(self.bn3(self.conv3(x)))
        x = nn.relu(self.bn4(self.conv4(x)))
        x = nn.relu(self.bn5(self.conv5(x)))
        x = jt.max(x, dim=2)
        x = nn.relu(self.bn_fc1(self.fc1(x)))
        x = self.dp1(x)
        x = nn.relu(self.bn_fc2(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)
        return x


# ============================================================
# 学习率调度器
# ============================================================

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=1e-5):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
        self.optimizer.lr = lr
        return lr


# ============================================================
# 训练与推理
# ============================================================

def configure_cuda(mode):
    """配置 Jittor CUDA；返回实际是否启用 CUDA。"""
    if mode == 'off':
        jt.flags.use_cuda = 0
        return False

    try:
        jt.flags.use_cuda = 1
        x = jt.ones((2, 3))
        y = (x * x).sum()
        y.sync()
        print("CUDA check: OK")
        return True
    except Exception as exc:
        if mode == 'on':
            raise RuntimeError(
                "CUDA was requested but Jittor could not enable it. "
                "Please install a CUDA toolkit that Jittor can find."
            ) from exc
        print(f"CUDA check: unavailable ({exc})")
        print("Falling back to CPU. Use this only for smoke tests; full training will be slow.")
        jt.flags.use_cuda = 0
        return False


def check_data(data_dir):
    train_points = np.load(os.path.join(data_dir, 'train_points.npy'), mmap_mode='r')
    train_labels = np.load(os.path.join(data_dir, 'train_labels.npy'), mmap_mode='r')
    test_points = np.load(os.path.join(data_dir, 'test_points.npy'), mmap_mode='r')
    categories_path = os.path.join(data_dir, 'categories.txt')

    with open(categories_path, 'r') as f:
        categories = [line.strip() for line in f if line.strip()]

    assert train_points.shape == (9843, 2048, 3), train_points.shape
    assert train_labels.shape == (9843,), train_labels.shape
    assert test_points.shape == (2468, 2048, 3), test_points.shape
    assert len(categories) == 40, len(categories)
    print("Data check: OK")
    print(f"  Train points: {train_points.shape}, labels: {train_labels.shape}")
    print(f"  Test points:  {test_points.shape}, classes: {len(categories)}")


def stratified_train_val_indices(labels, val_ratio=0.1, seed=42, num_classes=40):
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    train_indices = []
    val_indices = []

    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0].copy()
        rng.shuffle(class_indices)
        if val_ratio > 0:
            val_count = int(round(len(class_indices) * val_ratio))
            val_count = max(1, val_count)
            if len(class_indices) > 1:
                val_count = min(val_count, len(class_indices) - 1)
        else:
            val_count = 0
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(val_indices, dtype=np.int64)


def classification_loss(logits, labels, smoothing=0.0):
    if smoothing <= 0:
        return nn.cross_entropy_loss(logits, labels)

    nll_loss = nn.cross_entropy_loss(logits, labels)
    log_probs = nn.log_softmax(logits, dim=1)
    smooth_loss = -log_probs.mean(dim=1).mean()
    return (1.0 - smoothing) * nll_loss + smoothing * smooth_loss


def train_one_epoch(model, train_loader, optimizer, epoch, label_smoothing=0.0,
                    log_interval=20):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    t0 = time.time()

    for batch_idx, (points, labels) in enumerate(train_loader):
        points = jt.array(points).permute(0, 2, 1)  # (B, 3, N)
        labels = jt.array(labels).reshape(-1)

        logits = model(points)
        loss = classification_loss(logits, labels, smoothing=label_smoothing)

        optimizer.step(loss)

        preds = logits.argmax(dim=1)[0]
        total_correct += (preds == labels).sum().item()
        total_count += labels.shape[0]
        total_loss += loss.item() * labels.shape[0]

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}] "
                  f"Loss: {total_loss/total_count:.4f}  "
                  f"Acc: {total_correct/total_count*100:.2f}%  "
                  f"Time: {time.time()-t0:.1f}s")

    return total_loss / total_count, total_correct / total_count * 100


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with jt.no_grad():
        for points, labels in val_loader:
            points = jt.array(points).permute(0, 2, 1)
            labels = jt.array(labels).reshape(-1)

            logits = model(points)
            loss = nn.cross_entropy_loss(logits, labels)
            preds = logits.argmax(dim=1)[0]

            total_correct += (preds == labels).sum().item()
            total_count += labels.shape[0]
            total_loss += loss.item() * labels.shape[0]

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count * 100


def predict(model, test_loader, vote_num=1):
    """对测试集进行预测，返回 {样本编号: 预测类别} 字典。"""
    model.eval()
    logit_sums = {}
    counts = {}

    with jt.no_grad():
        for vote_idx in range(vote_num):
            t0 = time.time()
            for points, indices in test_loader:
                points = jt.array(points).permute(0, 2, 1)
                indices = jt.array(indices).reshape(-1)

                logits = model(points)
                logits_np = logits.numpy()

                for i in range(logits_np.shape[0]):
                    sample_id = int(indices[i].item())
                    if sample_id not in logit_sums:
                        logit_sums[sample_id] = logits_np[i].copy()
                        counts[sample_id] = 1
                    else:
                        logit_sums[sample_id] += logits_np[i]
                        counts[sample_id] += 1
            print(f"  Vote [{vote_idx + 1}/{vote_num}] done in {time.time() - t0:.1f}s")

    results = {}
    for sample_id in sorted(logit_sums.keys()):
        avg_logits = logit_sums[sample_id] / max(1, counts[sample_id])
        results[str(sample_id)] = int(np.argmax(avg_logits))

    return results


def validate_results(results, expected_count=2468, num_classes=40):
    assert len(results) == expected_count, f"Expected {expected_count}, got {len(results)}"
    for idx in range(expected_count):
        key = str(idx)
        assert key in results, f"Missing prediction for sample {key}"
        value = results[key]
        assert isinstance(value, int), f"Prediction for {key} is not int: {value}"
        assert 0 <= value < num_classes, f"Prediction for {key} out of range: {value}"
    print("Result JSON check: OK")


def save_result_zip(result_path, zip_path):
    if not zip_path:
        return
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(result_path, arcname='result.json')
    print(f"Packed {result_path} to {zip_path}")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--mode', type=str, default='train_predict',
                        choices=['train', 'predict', 'train_predict'])
    parser.add_argument('--model', type=str, default='pointnet',
                        choices=['pointnet', 'pct'])
    parser.add_argument('--cuda', type=str, default='auto',
                        choices=['auto', 'on', 'off'])
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--vote_num', type=int, default=10)
    parser.add_argument('--augment_level', type=str, default='light',
                        choices=['off', 'light', 'strong'])
    parser.add_argument('--save_path', type=str, default='pct_best.pkl')
    parser.add_argument('--load_path', type=str, default='pct_best.pkl')
    parser.add_argument('--result_path', type=str, default='result.json')
    parser.add_argument('--zip_path', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--max_train_samples', type=int, default=0)
    parser.add_argument('--max_val_samples', type=int, default=0)
    parser.add_argument('--max_test_samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    jt.set_global_seed(args.seed)
    cuda_enabled = configure_cuda(args.cuda)

    print("=" * 60)
    print("ModelNet40 Classification - PCT")
    print(f"Points: {args.n_points}  Batch: {args.batch_size}  "
          f"Epochs: {args.epochs}  LR: {args.lr}")
    print(f"Model: {args.model}  Mode: {args.mode}  CUDA: {'on' if cuda_enabled else 'off'}  "
          f"Votes: {args.vote_num}")
    print("=" * 60)

    check_data(args.data_dir)

    # --------------------------------------------------
    # 加载数据
    # --------------------------------------------------
    train_loader = None
    val_loader = None
    test_loader = None
    if args.mode in ('train', 'train_predict'):
        labels = np.load(os.path.join(args.data_dir, 'train_labels.npy'))
        train_indices, val_indices = stratified_train_val_indices(
            labels, val_ratio=args.val_ratio, seed=args.seed, num_classes=40)
        if args.max_train_samples > 0:
            train_indices = train_indices[:args.max_train_samples]
        if args.max_val_samples > 0:
            val_indices = val_indices[:args.max_val_samples]
        train_loader = ModelNet40Dataset(
            data_dir=args.data_dir, split='train', n_points=args.n_points,
            augment=True, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, indices=train_indices,
            random_sample=True, augment_level=args.augment_level)
        val_loader = ModelNet40Dataset(
            data_dir=args.data_dir, split='val', n_points=args.n_points,
            augment=False, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, indices=val_indices,
            random_sample=False)
        print(f"Train: {train_loader.total_len} samples")
        print(f"Val:   {val_loader.total_len} samples")

    if args.mode in ('predict', 'train_predict'):
        test_indices = None
        if args.max_test_samples > 0:
            test_indices = np.arange(args.max_test_samples, dtype=np.int64)
        test_loader = ModelNet40Dataset(
            data_dir=args.data_dir, split='test', n_points=args.n_points,
            augment=False, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, indices=test_indices,
            random_sample=True)
        print(f"Test:  {test_loader.total_len} samples")

    # --------------------------------------------------
    # 构建模型
    # --------------------------------------------------
    model = PointNet(num_classes=40) if args.model == 'pointnet' else PCT(num_classes=40)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    if args.mode in ('train', 'train_predict'):
        optimizer = nn.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        best_val_acc = -1.0

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, epoch,
                label_smoothing=args.label_smoothing,
                log_interval=args.log_interval)
            val_loss, val_acc = evaluate(model, val_loader)
            lr = scheduler.step()
            print(f"Epoch [{epoch}/{args.epochs}]  "
                  f"Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
                  f"LR: {optimizer.lr:.6f}  Time: {time.time()-t0:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ensure_parent_dir(args.save_path)
                model.save(args.save_path)
                print(f"  Best model saved to {args.save_path} (Val Acc: {best_val_acc:.2f}%)")

        print(f"Training done. Best Val Acc: {best_val_acc:.2f}%")

    if args.mode == 'predict':
        assert os.path.exists(args.load_path), f"{args.load_path} not found."
        model.load(args.load_path)
        print(f"Loaded model from {args.load_path}")
    elif args.mode == 'train_predict':
        assert os.path.exists(args.save_path), f"{args.save_path} not found."
        model.load(args.save_path)
        print(f"Loaded best model from {args.save_path}")

    if args.mode in ('predict', 'train_predict'):
        print("Generating predictions on test set...")
        results = predict(model, test_loader, vote_num=args.vote_num)
        validate_results(results, expected_count=test_loader.total_len, num_classes=40)
        ensure_parent_dir(args.result_path)
        with open(args.result_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {args.result_path}")
        save_result_zip(args.result_path, args.zip_path)

    print("Done!")


if __name__ == '__main__':
    main()
