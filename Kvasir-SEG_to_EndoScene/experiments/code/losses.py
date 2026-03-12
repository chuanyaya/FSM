import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1).float()
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class PseudoLabelLoss(nn.Module):

    def __init__(self, confidence_threshold=0.95, use_soft_label=False, adaptive_threshold=True):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.use_soft_label = use_soft_label
        self.adaptive_threshold = adaptive_threshold
        self.training_progress = 0.0

    def set_progress(self, progress):
        self.training_progress = progress

    def get_dynamic_threshold(self, confidence_values=None):
        if not self.adaptive_threshold:
            return self.confidence_threshold
        threshold = 0.3 - 0.15 * self.training_progress
        threshold = max(0.1, min(0.3, threshold))
        return threshold

    def forward(self, pred_student, pred_teacher):
        prob_teacher = torch.sigmoid(pred_teacher.detach())
        pseudo_label = (prob_teacher > 0.5).float()
        confidence = torch.abs(prob_teacher - 0.5) * 2
        dynamic_threshold = self.get_dynamic_threshold(confidence)
        confidence_mask = (confidence > dynamic_threshold).float()
        num_confident = confidence_mask.sum()
        if num_confident < 1:
            zero_loss = (pred_student * 0).sum()
            return (zero_loss, 0.0)
        if self.use_soft_label:
            loss = F.binary_cross_entropy_with_logits(pred_student, prob_teacher.detach(), reduction='none')
        else:
            loss = F.binary_cross_entropy_with_logits(pred_student, pseudo_label, reduction='none')
        loss = (loss * confidence_mask).sum() / (num_confident + 1e-06)
        mask_ratio = num_confident / confidence_mask.numel()
        return (loss, mask_ratio.item())

class ConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred_student, pred_teacher):
        prob_student = torch.sigmoid(pred_student)
        prob_teacher = torch.sigmoid(pred_teacher.detach())
        return F.mse_loss(prob_student, prob_teacher)

class TotalLoss(nn.Module):

    def __init__(self, sup_weight=1.0, unsup_weight=1.0, confidence_threshold=0.95, rampup_epochs=20, adaptive_threshold=True):
        super().__init__()
        self.sup_weight = sup_weight
        self.unsup_weight = unsup_weight
        self.adaptive_threshold = adaptive_threshold
        self.rampup_epochs = rampup_epochs
        self.sup_loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.pseudo_loss_fn = PseudoLabelLoss(confidence_threshold=confidence_threshold, adaptive_threshold=adaptive_threshold)

    def set_progress(self, progress):
        self.pseudo_loss_fn.set_progress(progress)

    def get_unsup_weight(self, epoch):
        if epoch < self.rampup_epochs:
            return self.unsup_weight * (epoch / self.rampup_epochs)
        return self.unsup_weight

    def forward(self, pred_student_src, mask_src, pred_student_tgt_l=None, mask_tgt_l=None, pred_student_tgt_u=None, pred_teacher_tgt_u=None, epoch=0):
        loss_dict = {}
        total_loss = 0.0
        loss_sup_src = self.sup_loss_fn(pred_student_src, mask_src)
        loss_dict['loss_sup_src'] = loss_sup_src.item()
        total_loss += self.sup_weight * loss_sup_src
        if pred_student_tgt_l is not None and mask_tgt_l is not None:
            loss_sup_tgt = self.sup_loss_fn(pred_student_tgt_l, mask_tgt_l)
            loss_dict['loss_sup_tgt'] = loss_sup_tgt.item()
            total_loss += self.sup_weight * loss_sup_tgt
        if pred_student_tgt_u is not None and pred_teacher_tgt_u is not None:
            unsup_weight = self.get_unsup_weight(epoch)
            loss_pseudo, mask_ratio = self.pseudo_loss_fn(pred_student_tgt_u, pred_teacher_tgt_u)
            loss_dict['loss_pseudo'] = loss_pseudo.item()
            loss_dict['pseudo_mask_ratio'] = mask_ratio
            total_loss += unsup_weight * loss_pseudo
        loss_dict['total_loss'] = total_loss.item()
        return (total_loss, loss_dict)
if __name__ == '__main__':
    pred_student = torch.randn(2, 1, 256, 256)
    pred_teacher = torch.randn(2, 1, 256, 256)
    mask = torch.randint(0, 2, (2, 1, 256, 256)).float()
    dice_loss = DiceLoss()
    print(f'Dice Loss: {dice_loss(pred_student, mask):.4f}')
    bce_dice = BCEDiceLoss()
    print(f'BCE+Dice Loss: {bce_dice(pred_student, mask):.4f}')
    pseudo_loss = PseudoLabelLoss(confidence_threshold=0.9)
    loss, ratio = pseudo_loss(pred_student, pred_teacher)
    print(f'Pseudo Loss: {loss:.4f}, Mask Ratio: {ratio:.4f}')
    total_loss_fn = TotalLoss()
    total, loss_dict = total_loss_fn(pred_student, mask, pred_student, mask, pred_student, pred_teacher, epoch=10)
    print(f'Total Loss: {total:.4f}')
    print(f'Loss Dict: {loss_dict}')
