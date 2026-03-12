import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, num_classes=3, ignore_background=True):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes
        self.ignore_background = ignore_background

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target.long(), self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        start_c = 1 if self.ignore_background else 0
        num_fg = self.num_classes - start_c
        dice_loss = 0.0
        for c in range(start_c, self.num_classes):
            pred_c = pred_soft[:, c].contiguous().view(-1)
            target_c = target_onehot[:, c].contiguous().view(-1)
            intersection = (pred_c * target_c).sum()
            dice = (2.0 * intersection + self.smooth) / (pred_c.sum() + target_c.sum() + self.smooth)
            dice_loss += 1 - dice
        return dice_loss / num_fg

class CEDiceLoss(nn.Module):

    def __init__(self, ce_weight=0.5, dice_weight=0.5, num_classes=3, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('_class_weights', w)
        else:
            self._class_weights = None
        self.dice = DiceLoss(num_classes=num_classes, ignore_background=True)

    def forward(self, pred, target):
        target = target.long()
        w = self._class_weights.to(pred.device) if self._class_weights is not None else None
        ce_loss = F.cross_entropy(pred, target, weight=w)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss

class ConsistencyLoss(nn.Module):

    def __init__(self, loss_type='mse', temperature=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, pred_student, pred_teacher):
        pred_teacher = pred_teacher.detach()
        if self.loss_type == 'mse':
            prob_student = F.softmax(pred_student / self.temperature, dim=1)
            prob_teacher = F.softmax(pred_teacher / self.temperature, dim=1)
            loss = F.mse_loss(prob_student, prob_teacher)
        elif self.loss_type == 'kl':
            log_prob_student = F.log_softmax(pred_student / self.temperature, dim=1)
            prob_teacher = F.softmax(pred_teacher / self.temperature, dim=1)
            loss = F.kl_div(log_prob_student, prob_teacher, reduction='batchmean')
        else:
            raise ValueError(f'Unknown loss_type: {self.loss_type}')
        return (loss, 1.0)

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
        prob_teacher = F.softmax(pred_teacher.detach(), dim=1)
        max_prob, pseudo_label = prob_teacher.max(dim=1)
        confidence = max_prob
        dynamic_threshold = self.get_dynamic_threshold(confidence)
        confidence_mask = (confidence > dynamic_threshold).float()
        num_confident = confidence_mask.sum()
        if num_confident < 1:
            zero_loss = (pred_student * 0).sum()
            return (zero_loss, 0.0)
        loss = F.cross_entropy(pred_student, pseudo_label, reduction='none')
        loss = (loss * confidence_mask).sum() / (num_confident + 1e-06)
        mask_ratio = num_confident / confidence_mask.numel()
        return (loss, mask_ratio.item())

class TotalLoss(nn.Module):

    def __init__(self, sup_weight=1.0, unsup_weight=1.0, confidence_threshold=0.95, rampup_epochs=50, adaptive_threshold=True, consistency_mode='fixmatch', consistency_loss_type='mse', consistency_temperature=1.0):
        super().__init__()
        self.sup_weight = sup_weight
        self.unsup_weight = unsup_weight
        self.adaptive_threshold = adaptive_threshold
        self.rampup_epochs = rampup_epochs
        self.consistency_mode = consistency_mode
        self.sup_loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5, num_classes=3, class_weights=[0.5, 1.0, 2.0])
        if consistency_mode == 'consistency':
            self.unsup_loss_fn = ConsistencyLoss(loss_type=consistency_loss_type, temperature=consistency_temperature)
        else:
            self.unsup_loss_fn = PseudoLabelLoss(confidence_threshold=confidence_threshold, adaptive_threshold=adaptive_threshold)
        self.pseudo_loss_fn = self.unsup_loss_fn

    def set_progress(self, progress):
        if self.consistency_mode == 'fixmatch':
            self.unsup_loss_fn.set_progress(progress)

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
            loss_pseudo, mask_ratio = self.unsup_loss_fn(pred_student_tgt_u, pred_teacher_tgt_u)
            loss_dict['loss_pseudo'] = loss_pseudo.item()
            loss_dict['pseudo_mask_ratio'] = mask_ratio
            total_loss += unsup_weight * loss_pseudo
        loss_dict['total_loss'] = total_loss.item()
        return (total_loss, loss_dict)
if __name__ == '__main__':
    num_classes = 3
    pred_student = torch.randn(2, num_classes, 256, 256)
    pred_teacher = torch.randn(2, num_classes, 256, 256)
    mask = torch.randint(0, num_classes, (2, 256, 256))
    dice_loss = DiceLoss(num_classes=num_classes)
    print(f'Dice Loss: {dice_loss(pred_student, mask):.4f}')
    ce_dice = CEDiceLoss(num_classes=num_classes)
    print(f'CE+Dice Loss: {ce_dice(pred_student, mask):.4f}')
    pseudo_loss = PseudoLabelLoss(confidence_threshold=0.9)
    loss, ratio = pseudo_loss(pred_student, pred_teacher)
    print(f'Pseudo Loss: {loss:.4f}, Mask Ratio: {ratio:.4f}')
    total_loss_fn = TotalLoss()
    total, loss_dict = total_loss_fn(pred_student, mask, pred_student, mask, pred_student, pred_teacher, epoch=10)
    print(f'Total Loss: {total:.4f}')
    print(f'Loss Dict: {loss_dict}')
