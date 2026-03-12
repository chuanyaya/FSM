import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.models._utils')
import pandas as pd
import cv2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from models_freq_perturb_fixed import MeanTeacherWithFSM
from losses import TotalLoss
from augmentations import DualAugmentation, get_validation_augmentation
SRC_TRAIN_CSV = os.path.join(PROJECT_ROOT, 'data/csv/Kvasir_SEG/train.csv')
TGT_LABELED_CSV = os.path.join(PROJECT_ROOT, 'data/csv/EndoScene_SSDA_50p/train_labeled.csv')
TGT_UNLABELED_CSV = os.path.join(PROJECT_ROOT, 'data/csv/EndoScene_SSDA_50p/train_unlabeled.csv')
TGT_TEST_CSV = os.path.join(PROJECT_ROOT, 'data/csv/EndoScene/test.csv')
SRC_ROOT = os.path.join(PROJECT_ROOT, 'Kvasir-SEG')
TGT_ROOT = os.path.join(PROJECT_ROOT, 'EndoScene/MergedDataset')
TGT_TEST_ROOT = os.path.join(PROJECT_ROOT, 'EndoScene/TestDataset')
IMG_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 100
LR = 0.0001
WEIGHT_DECAY = 0.0001
EMA_DECAY = 0.999
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 0.5
CONFIDENCE_THRESHOLD = 0.15
RAMPUP_EPOCHS = 50
USE_FREQ_PERTURB = True
FREQ_PERTURB_EPS = 0.2
FREQ_PERTURB_WEIGHT = 0.3
EXP_NAME = 'meanteacher_nomixstyle_freqperturb_resnet18_50p'
SAVE_DIR = os.path.join(SCRIPT_DIR, 'checkpoints', EXP_NAME)
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs', EXP_NAME)

class MixedBatchDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform, img_size=256, normalize=True, has_mask=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.normalize = normalize
        self.has_mask = has_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image'])
        if image is None:
            raise ValueError(f"Cannot read image: {row['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.has_mask:
            mask = cv2.imread(row['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask: {row['mask']}")
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        augmented = self.transform(image, mask)
        image_weak = augmented['image_weak'].astype(np.float32) / 255.0
        image_strong = augmented['image_strong'].astype(np.float32) / 255.0
        if self.normalize:
            image_weak = (image_weak - 0.5) / 0.22
            image_strong = (image_strong - 0.5) / 0.22
        result = {'image_weak': torch.from_numpy(image_weak.transpose(2, 0, 1)), 'image_strong': torch.from_numpy(image_strong.transpose(2, 0, 1))}
        if self.has_mask:
            result['mask'] = torch.from_numpy(augmented['mask']).float()
        return result

class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform, img_size=256, normalize=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.normalize = normalize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image'])
        if image is None:
            raise ValueError(f"Cannot read image: {row['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot read mask: {row['mask']}")
        mask = (mask > 127).astype(np.uint8)
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image'].astype(np.float32) / 255.0
        if self.normalize:
            image = (image - 0.5) / 0.22
        return {'image': torch.from_numpy(image.transpose(2, 0, 1)), 'mask': torch.from_numpy(augmented['mask']).float()}

@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_inter = 0
    total_union = 0
    total_pred = 0
    total_target = 0
    for batch in tqdm(val_loader, desc='Validating', leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        preds = model.student_forward(images)
        preds = torch.sigmoid(preds)
        preds_bin = (preds > 0.5).float()
        inter = (preds_bin.squeeze(1) * masks).sum()
        union = preds_bin.squeeze(1).sum() + masks.sum() - inter
        total_inter += inter.item()
        total_union += union.item()
        total_pred += preds_bin.sum().item()
        total_target += masks.sum().item()
    dice = 2 * total_inter / (total_pred + total_target + 1e-08)
    iou = total_inter / (total_union + 1e-08)
    return {'dice': dice, 'iou': iou}

def train_one_epoch(model, src_loader, tgt_labeled_loader, tgt_unlabeled_loader, optimizer, loss_fn, epoch, device, writer):
    model.train()
    src_iter = iter(src_loader)
    tgt_l_iter = iter(tgt_labeled_loader)
    tgt_u_iter = iter(tgt_unlabeled_loader)
    num_batches = max(len(src_loader), len(tgt_labeled_loader), len(tgt_unlabeled_loader))
    epoch_losses = {'total': [], 'sup_src': [], 'sup_tgt': [], 'pseudo': [], 'freq_perturb': [], 'mask_ratio': []}
    pbar = tqdm(range(num_batches), desc=f'Epoch {epoch}')
    for batch_idx in pbar:
        try:
            src_batch = next(src_iter)
        except StopIteration:
            src_iter = iter(src_loader)
            src_batch = next(src_iter)
        try:
            tgt_l_batch = next(tgt_l_iter)
        except StopIteration:
            tgt_l_iter = iter(tgt_labeled_loader)
            tgt_l_batch = next(tgt_l_iter)
        try:
            tgt_u_batch = next(tgt_u_iter)
        except StopIteration:
            tgt_u_iter = iter(tgt_unlabeled_loader)
            tgt_u_batch = next(tgt_u_iter)
        src_weak = src_batch['image_weak'].to(device)
        src_strong = src_batch['image_strong'].to(device)
        src_mask = src_batch['mask'].to(device)
        tgt_l_weak = tgt_l_batch['image_weak'].to(device)
        tgt_l_strong = tgt_l_batch['image_strong'].to(device)
        tgt_l_mask = tgt_l_batch['mask'].to(device)
        tgt_u_weak = tgt_u_batch['image_weak'].to(device)
        tgt_u_strong = tgt_u_batch['image_strong'].to(device)
        pred_src_student = model.student_forward(src_strong)
        pred_tgt_l_student = model.student_forward(tgt_l_strong)
        pred_tgt_u_student = model.student_forward(tgt_u_strong)
        if USE_FREQ_PERTURB:
            pred_tgt_u_perturbed = model.forward_with_freq_perturb(tgt_u_strong, apply_perturb=True)
        else:
            pred_tgt_u_perturbed = None
        with torch.no_grad():
            pred_tgt_u_teacher = model.teacher_forward(tgt_u_weak)
        original_loss, loss_info = loss_fn(pred_src_student, src_mask, pred_tgt_l_student, tgt_l_mask, pred_tgt_u_student, pred_tgt_u_teacher, epoch=epoch)
        if USE_FREQ_PERTURB and pred_tgt_u_perturbed is not None:
            freq_perturb_loss = loss_fn.sup_loss_fn(pred_tgt_u_perturbed, pred_tgt_u_teacher.detach() > 0.5)
            total_loss = original_loss + FREQ_PERTURB_WEIGHT * freq_perturb_loss
        else:
            freq_perturb_loss = torch.tensor(0.0).to(device)
            total_loss = original_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model.update_teacher()
        epoch_losses['total'].append(loss_info['total_loss'])
        epoch_losses['sup_src'].append(loss_info.get('loss_sup_src', 0))
        epoch_losses['sup_tgt'].append(loss_info.get('loss_sup_tgt', 0))
        epoch_losses['pseudo'].append(loss_info.get('loss_pseudo', 0))
        epoch_losses['freq_perturb'].append(freq_perturb_loss.item() if USE_FREQ_PERTURB else 0)
        epoch_losses['mask_ratio'].append(loss_info.get('pseudo_mask_ratio', 0))
        pbar.set_postfix({'loss': f'{total_loss.item():.3f}', 'pseudo': f"{loss_info.get('loss_pseudo', 0):.3f}", 'freq': f'{(freq_perturb_loss.item() if USE_FREQ_PERTURB else 0):.3f}', 'mask%': f"{loss_info.get('pseudo_mask_ratio', 0) * 100:.0f}"})
        global_step = epoch * num_batches + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar('Loss/total', total_loss.item(), global_step)
            writer.add_scalar('Loss/pseudo', loss_info.get('loss_pseudo', 0), global_step)
            if USE_FREQ_PERTURB:
                writer.add_scalar('Loss/freq_perturb', freq_perturb_loss.item(), global_step)
            writer.add_scalar('Mask/ratio', loss_info.get('pseudo_mask_ratio', 0), global_step)
    return {k: np.mean(v) for k, v in epoch_losses.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Train or test mode')
    parser.add_argument('--checkpoint', type=str, default='best_model_false_50.pth', help='Checkpoint filename for test mode')
    args = parser.parse_args()
    if args.mode == 'test':
        test_with_args(args)
        return
    print('=' * 60)
    print('Mean Teacher + FixMatch (No MixStyle, Single GPU)')
    print('=' * 60)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Frequency perturbation: eps={FREQ_PERTURB_EPS}, weight={FREQ_PERTURB_WEIGHT}')
    print(f'Save dir: {SAVE_DIR}')
    print('=' * 60)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)
    print('\n加载数据...')
    src_df = pd.read_csv(SRC_TRAIN_CSV)
    src_df['image'] = SRC_ROOT + '/images/' + src_df['image']
    src_df['mask'] = SRC_ROOT + '/masks/' + src_df['mask']
    tgt_l_df = pd.read_csv(TGT_LABELED_CSV)
    tgt_l_df['image'] = TGT_ROOT + '/images/' + tgt_l_df['image']
    tgt_l_df['mask'] = TGT_ROOT + '/masks/' + tgt_l_df['mask']
    tgt_u_df = pd.read_csv(TGT_UNLABELED_CSV)
    tgt_u_df['image'] = TGT_ROOT + '/images/' + tgt_u_df['image']
    test_df = pd.read_csv(TGT_TEST_CSV)
    test_df['image'] = TGT_TEST_ROOT + '/images/' + test_df['image']
    test_df['mask'] = TGT_TEST_ROOT + '/masks/' + test_df['mask']
    print(f'源域: {len(src_df)} | 目标域有标注: {len(tgt_l_df)} | 目标域无标注: {len(tgt_u_df)} | 测试: {len(test_df)}')
    dual_aug = DualAugmentation(img_size=IMG_SIZE)
    val_aug = get_validation_augmentation(img_size=IMG_SIZE)
    src_dataset = MixedBatchDataset(src_df, dual_aug, IMG_SIZE, has_mask=True)
    tgt_l_dataset = MixedBatchDataset(tgt_l_df, dual_aug, IMG_SIZE, has_mask=True)
    tgt_u_dataset = MixedBatchDataset(tgt_u_df, dual_aug, IMG_SIZE, has_mask=False)
    test_dataset = ValidationDataset(test_df, val_aug, IMG_SIZE)
    src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    tgt_l_loader = torch.utils.data.DataLoader(tgt_l_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    tgt_u_loader = torch.utils.data.DataLoader(tgt_u_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print('\n初始化模型（ResNet18 + 频率扰动，无MixStyle）...')
    print(f'- Backbone: ResNet18')
    print(f'- MixStyle: Disabled (消融实验)')
    print(f'- Frequency Perturbation: eps={FREQ_PERTURB_EPS}, weight={FREQ_PERTURB_WEIGHT}')
    model = MeanTeacherWithFSM(backbone='resnet18', num_classes=1, ema_decay=EMA_DECAY, pretrained=False, fsm_layers=[], fsm_p=0.0, fsm_alpha=0.0, adaptive_fsm=False, use_freq_perturb=USE_FREQ_PERTURB, freq_perturb_eps=FREQ_PERTURB_EPS).to(device)
    model.ema_teacher.teacher.to(device)
    loss_fn = TotalLoss(sup_weight=SUP_WEIGHT, unsup_weight=UNSUP_WEIGHT, confidence_threshold=CONFIDENCE_THRESHOLD, rampup_epochs=RAMPUP_EPOCHS, adaptive_threshold=False)
    optimizer = optim.Adam(model.student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print('\n' + '=' * 60)
    print('开始训练...')
    print('=' * 60 + '\n')
    best_dice = 0.0
    for epoch in range(NUM_EPOCHS):
        progress = epoch / NUM_EPOCHS
        loss_fn.set_progress(progress)
        train_losses = train_one_epoch(model, src_loader, tgt_l_loader, tgt_u_loader, optimizer, loss_fn, epoch, device, writer)
        val_metrics = validate(model, test_loader, device)
        writer.add_scalar('Metrics/Dice', val_metrics['dice'], epoch)
        writer.add_scalar('Metrics/IoU', val_metrics['iou'], epoch)
        print(f"\nEpoch {epoch}: Dice={val_metrics['dice']:.4f}, IoU={val_metrics['iou']:.4f}, mask%={train_losses['mask_ratio'] * 100:.1f}")
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_dice': best_dice, 'config': {'lr': LR, 'batch_size': BATCH_SIZE, 'confidence_threshold': CONFIDENCE_THRESHOLD}}, os.path.join(SAVE_DIR, 'best_model_false_50.pth'))
            print(f'  ✓ 保存最佳模型 (Dice: {best_dice:.4f})')
    writer.close()
    print(f'\n训练完成！最佳 Dice: {best_dice:.4f}')

def test_with_args(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    TEST_ROOT = os.path.join(PROJECT_ROOT, 'test_images')
    TEST_CSV = os.path.join(TEST_ROOT, 'test_independent.csv')
    print('\n加载测试数据...')
    test_df = pd.read_csv(TEST_CSV)
    test_df['image'] = TEST_ROOT + '/images/' + test_df['image']
    test_df['mask'] = TEST_ROOT + '/masks/' + test_df['mask']
    val_aug = get_validation_augmentation(img_size=IMG_SIZE)
    test_dataset = ValidationDataset(test_df, val_aug, IMG_SIZE)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model = MeanTeacherWithFSM(backbone='resnet18', num_classes=1, ema_decay=EMA_DECAY, pretrained=False, fsm_layers=[], fsm_p=0.0, fsm_alpha=0.0, adaptive_fsm=False, use_freq_perturb=USE_FREQ_PERTURB, freq_perturb_eps=FREQ_PERTURB_EPS).to(device)
    if os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    elif os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(SAVE_DIR, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f'错误：找不到checkpoint文件 {checkpoint_path}')
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    merged_state_dict = dict(state_dict)
    for key, value in state_dict.items():
        if key.startswith('student.encoder.'):
            merged_state_dict[key.replace('student.encoder.', 'student_encoder.')] = value
        elif key.startswith('student.decoder.'):
            merged_state_dict[key.replace('student.decoder.', 'student_decoder.')] = value
    model.load_state_dict(merged_state_dict, strict=False)
    print('\n开始测试...')
    model.eval()
    total_inter = 0
    total_union = 0
    total_pred = 0
    total_target = 0
    dice_per_image = []
    iou_per_image = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            preds = model.student_forward(images)
            preds = torch.sigmoid(preds)
            preds_bin = (preds > 0.5).float()
            inter = (preds_bin.squeeze(1) * masks).sum()
            union = preds_bin.squeeze(1).sum() + masks.sum() - inter
            total_inter += inter.item()
            total_union += union.item()
            total_pred += preds_bin.sum().item()
            total_target += masks.sum().item()
            for i in range(images.shape[0]):
                pred_i = preds_bin[i].squeeze()
                mask_i = masks[i]
                inter_i = (pred_i * mask_i).sum().item()
                union_i = pred_i.sum().item() + mask_i.sum().item() - inter_i
                dice_i = 2 * inter_i / (pred_i.sum().item() + mask_i.sum().item() + 1e-08)
                iou_i = inter_i / (union_i + 1e-08)
                dice_per_image.append(dice_i)
                iou_per_image.append(iou_i)
    dice_avg = 2 * total_inter / (total_pred + total_target + 1e-08)
    iou_avg = total_inter / (total_union + 1e-08)
    dice_mean = np.mean(dice_per_image)
    dice_std = np.std(dice_per_image)
    iou_mean = np.mean(iou_per_image)
    iou_std = np.std(iou_per_image)
    print('\n' + '=' * 60)
    print('测试结果')
    print('=' * 60)
    print(f'整体 Dice: {dice_avg:.4f}')
    print(f'整体 IoU:  {iou_avg:.4f}')
    print('=' * 60)
if __name__ == '__main__':
    main()
