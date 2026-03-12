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
from models_with_fsm_fixed import MeanTeacherWithMixStyle
from losses import TotalLoss
from augmentations import DualAugmentation, get_validation_augmentation, apply_bright_warm_stylization, apply_dark_cool_stylization
LABELED_RATIO = '10p'
SRC_TRAIN_CSV = '/root/autodl-tmp/kits23_ready_v7/train.csv'
TGT_TEST_CSV = '/root/autodl-tmp/kits19_ready_v7/test.csv'
TGT_VAL_CSV = '/root/autodl-tmp/kits19_ready_v7/val.csv'
SRC_ROOT = '/root/autodl-tmp/kits23_ready_v7'
TGT_VAL_ROOT = '/root/autodl-tmp/kits19_ready_v7'
TGT_TEST_ROOT = '/root/autodl-tmp/kits19_ready_v7'

def get_labeled_csv(ratio):
    return f'/root/autodl-tmp/kits19_ready_v7/train_labeled_{ratio}.csv'

def get_unlabeled_csv(ratio):
    return f'/root/autodl-tmp/kits19_ready_v7/train_unlabeled_{ratio}.csv'
NUM_CLASSES = 3
IMG_SIZE = 256
BATCH_SIZE = 16
NUM_EPOCHS = 70
LR = 0.0001
WEIGHT_DECAY = 0.0001
EMA_DECAY = 0.999
SUP_WEIGHT = 1.0
UNSUP_WEIGHT = 0.5
RAMPUP_EPOCHS = 50
EXP_NAME = 'resnet18_baseline_kits_v7'
SAVE_DIR = os.path.join(SCRIPT_DIR, 'checkpoints', EXP_NAME)
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs', EXP_NAME)

class NullWriter:

    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass

class MixedBatchDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform, img_size=256, normalize=True, has_mask=True, stylize=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.normalize = normalize
        self.has_mask = has_mask
        self.stylize = stylize

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
            mask = mask.astype(np.uint8)
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
            result['mask'] = torch.from_numpy(augmented['mask']).long()
        return result

class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, df, transform, img_size=256, normalize=True, stylize=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size
        self.normalize = normalize
        self.stylize = stylize

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
        mask = mask.astype(np.uint8)
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image'].astype(np.float32) / 255.0
        if self.normalize:
            image = (image - 0.5) / 0.22
        return {'image': torch.from_numpy(image.transpose(2, 0, 1)), 'mask': torch.from_numpy(augmented['mask']).long()}

@torch.no_grad()
def validate(model, val_loader, device, num_classes=3):
    model.eval()
    class_inter = [0.0] * num_classes
    class_union = [0.0] * num_classes
    class_pred = [0.0] * num_classes
    class_target = [0.0] * num_classes
    for batch in tqdm(val_loader, desc='Validating', leave=False):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        preds = model.student_forward(images)
        preds_class = preds.argmax(dim=1)
        for c in range(num_classes):
            pred_c = (preds_class == c).float()
            target_c = (masks == c).float()
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - inter
            class_inter[c] += inter.item()
            class_union[c] += union.item()
            class_pred[c] += pred_c.sum().item()
            class_target[c] += target_c.sum().item()
    class_names = ['背景', '肾', '肿瘤']
    dice_per_class = []
    iou_per_class = []
    for c in range(num_classes):
        dice = 2 * class_inter[c] / (class_pred[c] + class_target[c] + 1e-08)
        iou = class_inter[c] / (class_union[c] + 1e-08)
        dice_per_class.append(dice)
        iou_per_class.append(iou)
    mean_dice = np.mean(dice_per_class[1:])
    mean_iou = np.mean(iou_per_class[1:])
    return {'dice': mean_dice, 'iou': mean_iou, 'dice_per_class': dice_per_class, 'iou_per_class': iou_per_class, 'class_names': class_names}

def train_one_epoch(model, src_loader, tgt_labeled_loader, tgt_unlabeled_loader, optimizer, loss_fn, epoch, device, writer):
    model.train()
    src_iter = iter(src_loader)
    tgt_l_iter = iter(tgt_labeled_loader)
    tgt_u_iter = iter(tgt_unlabeled_loader)
    num_batches = max(len(src_loader), len(tgt_labeled_loader), len(tgt_unlabeled_loader))
    epoch_losses = {'total': [], 'sup_src': [], 'sup_tgt': [], 'pseudo': [], 'mask_ratio': []}
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
        with torch.no_grad():
            pred_tgt_u_teacher = model.teacher_forward(tgt_u_weak)
        total_loss, loss_info = loss_fn(pred_src_student, src_mask, pred_tgt_l_student, tgt_l_mask, pred_tgt_u_student, pred_tgt_u_teacher, epoch=epoch)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        model.update_teacher()
        epoch_losses['total'].append(loss_info['total_loss'])
        epoch_losses['sup_src'].append(loss_info.get('loss_sup_src', 0))
        epoch_losses['sup_tgt'].append(loss_info.get('loss_sup_tgt', 0))
        epoch_losses['pseudo'].append(loss_info.get('loss_pseudo', 0))
        epoch_losses['mask_ratio'].append(loss_info.get('pseudo_mask_ratio', 0))
        pbar.set_postfix({'loss': f'{total_loss.item():.3f}', 'consist': f"{loss_info.get('loss_pseudo', 0):.3f}"})
        global_step = epoch * num_batches + batch_idx
        if batch_idx % 50 == 0:
            writer.add_scalar('Loss/total', total_loss.item(), global_step)
            writer.add_scalar('Loss/pseudo', loss_info.get('loss_pseudo', 0), global_step)
            writer.add_scalar('Mask/ratio', loss_info.get('pseudo_mask_ratio', 0), global_step)
    return {k: np.mean(v) for k, v in epoch_losses.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--labeled_ratio', type=str, default='10p', choices=['10p', '20p', '30p'], help='Labeled ratio: 10p, 20p or 30p')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for checkpoint directory (e.g., v2)')
    parser.add_argument('--domain_shift', type=str, default='v7', choices=['none', 'v7'], help='Domain shift mode: none or v7 (bidirectional)')
    args = parser.parse_args()
    labeled_ratio = args.labeled_ratio
    suffix = args.suffix
    TGT_LABELED_CSV = get_labeled_csv(labeled_ratio)
    TGT_UNLABELED_CSV = get_unlabeled_csv(labeled_ratio)
    exp_suffix = f'_{suffix}' if suffix else ''
    save_dir = os.path.join(SCRIPT_DIR, 'checkpoints', f'{EXP_NAME}{exp_suffix}')
    log_dir = os.path.join(SCRIPT_DIR, 'logs', f'{EXP_NAME}{exp_suffix}')
    print('=' * 60)
    print(f'Mean Teacher + Consistency Reg (Single GPU, {labeled_ratio} labeled, no MixStyle)')
    print('=' * 60)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Batch size: {BATCH_SIZE}, LR: {LR}')
    print(f'Loss: Mean Teacher (MSE consistency, unsup_weight={UNSUP_WEIGHT}, rampup={RAMPUP_EPOCHS})')
    print(f'ImageNet Pretrained: False')
    print(f'Save dir: {save_dir}')
    writer = NullWriter()
    print('\n加载数据...')
    src_df = pd.read_csv(SRC_TRAIN_CSV)
    src_df['image'] = SRC_ROOT + '/images/' + src_df['image']
    src_df['mask'] = SRC_ROOT + '/masks/' + src_df['mask']
    tgt_l_df = pd.read_csv(TGT_LABELED_CSV)
    print(f'Labeled CSV: {TGT_LABELED_CSV}')
    tgt_l_df['image'] = TGT_VAL_ROOT + '/images/' + tgt_l_df['image']
    tgt_l_df['mask'] = TGT_VAL_ROOT + '/masks/' + tgt_l_df['mask']
    tgt_u_df = pd.read_csv(TGT_UNLABELED_CSV)
    tgt_u_df['image'] = TGT_VAL_ROOT + '/images/' + tgt_u_df['image']
    print(f'Unlabeled CSV: {TGT_UNLABELED_CSV}')
    val_df = pd.read_csv(TGT_VAL_CSV)
    val_df['image'] = TGT_VAL_ROOT + '/images/' + val_df['image']
    val_df['mask'] = TGT_VAL_ROOT + '/masks/' + val_df['mask']
    test_df = pd.read_csv(TGT_TEST_CSV)
    test_df['image'] = TGT_TEST_ROOT + '/images/' + test_df['image']
    test_df['mask'] = TGT_TEST_ROOT + '/masks/' + test_df['mask']
    print(f'源域: {len(src_df)} | 目标域有标注: {len(tgt_l_df)} | 目标域无标注: {len(tgt_u_df)} | 验证: {len(val_df)} | 测试: {len(test_df)}')
    dual_aug = DualAugmentation(img_size=IMG_SIZE)
    val_aug = get_validation_augmentation(img_size=IMG_SIZE)
    print('=' * 60)
    print(f'ResNet18 Baseline v7 (Single GPU, {labeled_ratio} labeled, 无MixStyle)')
    print(f'使用预处理 v7 数据: Source=BRIGHT_WARM, Target=DARK_COOL')
    print('=' * 60)
    src_dataset = MixedBatchDataset(src_df, dual_aug, img_size=IMG_SIZE, has_mask=True, stylize=None)
    tgt_l_dataset = MixedBatchDataset(tgt_l_df, dual_aug, IMG_SIZE, has_mask=True, stylize=None)
    tgt_u_dataset = MixedBatchDataset(tgt_u_df, dual_aug, IMG_SIZE, has_mask=False, stylize=None)
    val_dataset = ValidationDataset(val_df, val_aug, IMG_SIZE, stylize=None)
    test_dataset = ValidationDataset(test_df, val_aug, IMG_SIZE, stylize=None)
    src_loader = torch.utils.data.DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    tgt_l_loader = torch.utils.data.DataLoader(tgt_l_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    tgt_u_loader = torch.utils.data.DataLoader(tgt_u_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print('\n初始化模型（ResNet18 Baseline，无MixStyle，无预训练）...')
    model = MeanTeacherWithMixStyle(backbone='resnet18', num_classes=NUM_CLASSES, ema_decay=EMA_DECAY, pretrained=False, mixstyle_layers=[], mixstyle_p=0.0, mixstyle_alpha=0.0, adaptive_mixstyle=False).to(device)
    model.ema_teacher.teacher.to(device)
    loss_fn = TotalLoss(sup_weight=SUP_WEIGHT, unsup_weight=UNSUP_WEIGHT, rampup_epochs=RAMPUP_EPOCHS, consistency_mode='consistency').to(device)
    optimizer = optim.Adam(model.student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print('\n' + '=' * 60)
    print('开始训练...')
    print('=' * 60 + '\n')
    best_dice = 0.0
    for epoch in range(NUM_EPOCHS):
        progress = epoch / NUM_EPOCHS
        loss_fn.set_progress(progress)
        train_losses = train_one_epoch(model, src_loader, tgt_l_loader, tgt_u_loader, optimizer, loss_fn, epoch, device, writer)
        val_metrics = validate(model, val_loader, device, NUM_CLASSES)
        writer.add_scalar('Metrics/mDice_val', val_metrics['dice'], epoch)
        writer.add_scalar('Metrics/mIoU_val', val_metrics['iou'], epoch)
        print(f"Epoch [{epoch}/{NUM_EPOCHS - 1}] | Val mDice: {val_metrics['dice']:.4f} | Val mIoU: {val_metrics['iou']:.4f}")
        if epoch > 0 and (epoch % 5 == 0 or epoch == NUM_EPOCHS - 1):
            best_ckpt_path = os.path.join(save_dir, f'best_model_{labeled_ratio}.pth')
            if os.path.exists(best_ckpt_path):
                current_state = {k: v.cpu().clone() for k, v in model.student.state_dict().items()}
                best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
                model.student.load_state_dict(best_ckpt['model'])
                test_metrics = validate(model, test_loader, device, NUM_CLASSES)
                model.student.load_state_dict(current_state)
                model.student.to(device)
                writer.add_scalar('Metrics/mDice_test', test_metrics['dice'], epoch)
                writer.add_scalar('Metrics/mIoU_test', test_metrics['iou'], epoch)
                print(f"      [Test@best_ep{best_ckpt['epoch']}] mDice: {test_metrics['dice']:.4f} | Test mIoU: {test_metrics['iou']:.4f}")
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            checkpoint = {'epoch': epoch, 'model': model.student.state_dict(), 'teacher': model.ema_teacher.teacher.state_dict(), 'optimizer': optimizer.state_dict(), 'best_dice': best_dice}
            print(f'  ✓ 最佳模型已更新 (Val mDice: {best_dice:.4f})')
    writer.close()
    print(f'\n训练完成！最佳 Dice: {best_dice:.4f}')

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--labeled_ratio', type=str, default='10p', choices=['10p', '20p', '30p'], help='Labeled ratio: 10p, 20p or 30p')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for checkpoint directory (e.g., v2)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint filename (auto if not set)')
    args = parser.parse_args()
    suffix = args.suffix
    exp_suffix = f'_{suffix}' if suffix else ''
    save_dir = os.path.join(SCRIPT_DIR, 'checkpoints', f'{EXP_NAME}{exp_suffix}')
    if args.checkpoint is None:
        args.checkpoint = f'best_model_{args.labeled_ratio}.pth'
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('\n加载测试数据...')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    test_csv = os.path.join(data_dir, 'test.csv')
    test_df = pd.read_csv(test_csv)
    test_df['image'] = data_dir + '/images/' + test_df['image']
    test_df['mask'] = data_dir + '/masks/' + test_df['mask']
    val_aug = get_validation_augmentation(img_size=IMG_SIZE)
    test_dataset = ValidationDataset(test_df, val_aug, IMG_SIZE, stylize=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    model = MeanTeacherWithMixStyle(backbone='resnet18', num_classes=NUM_CLASSES, ema_decay=EMA_DECAY, pretrained=False, mixstyle_layers=[], mixstyle_p=0.0, mixstyle_alpha=0.0, adaptive_mixstyle=False).to(device)
    if os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f'错误：找不到checkpoint文件 {checkpoint_path}')
        return
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.student.load_state_dict(checkpoint['model'])
    print('\n开始测试...')
    test_metrics = validate(model, test_loader, device, NUM_CLASSES)
    print('\n' + '=' * 60)
    print('测试结果')
    print('=' * 60)
    for c in range(1, NUM_CLASSES):
        print(f"{test_metrics['class_names'][c]}: Dice={test_metrics['dice_per_class'][c]:.4f}, IoU={test_metrics['iou_per_class'][c]:.4f}")
    print(f"\n前景平均 mDice: {test_metrics['dice']:.4f}")
    print(f"前景平均 mIoU:  {test_metrics['iou']:.4f}")
    print('=' * 60)
if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        sys.argv.remove('--test')
        test()
    else:
        main()
