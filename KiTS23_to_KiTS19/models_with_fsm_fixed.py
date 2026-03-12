import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from style_transfer import MixStyle, CrossNorm, AdaptiveMixStyle, StyleAugmentor
from freq_perturbation import FrequencyAwarePerturbation

class EMATeacher:

    def __init__(self, student_model, decay=0.999):
        self.decay = decay
        self.teacher = self._create_clean_teacher(student_model)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _create_clean_teacher(self, student_model):
        teacher = copy.deepcopy(student_model)
        for module in teacher.modules():
            if isinstance(module, (MixStyle, CrossNorm, AdaptiveMixStyle, StyleAugmentor)):
                module.set_activation(False)
        return teacher

    @torch.no_grad()
    def update(self, student_model):
        student_dict = student_model.state_dict()
        teacher_dict = self.teacher.state_dict()
        for key in teacher_dict:
            if key in student_dict:
                teacher_dict[key] = self.decay * teacher_dict[key] + (1 - self.decay) * student_dict[key]
        self.teacher.load_state_dict(teacher_dict)

    def __call__(self, x):
        self.teacher.eval()
        return self.teacher(x)

class ResNetEncoderWithMixStyle(nn.Module):

    def __init__(self, backbone='resnet50', pretrained=True, mixstyle_layers=['layer1', 'layer2'], mixstyle_p=0.2, mixstyle_alpha=0.1, adaptive_mixstyle=True, use_freq_perturb=False, freq_perturb_eps=0.2):
        super().__init__()
        self.use_freq_perturb = use_freq_perturb
        self.freq_perturb_eps = freq_perturb_eps
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f'Unknown backbone: {backbone}')
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.mixstyle_layers = mixstyle_layers
        self.mixstyle1 = None
        self.mixstyle2 = None
        if 'layer1' in mixstyle_layers:
            if adaptive_mixstyle:
                self.mixstyle1 = AdaptiveMixStyle(p_start=0.3, p_end=0.1, alpha=mixstyle_alpha, mix='crossdomain')
            else:
                self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='crossdomain')
        if 'layer2' in mixstyle_layers:
            if adaptive_mixstyle:
                self.mixstyle2 = AdaptiveMixStyle(p_start=0.3, p_end=0.1, alpha=mixstyle_alpha, mix='crossdomain')
            else:
                self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix='crossdomain')
        self.freq_perturb = None
        if use_freq_perturb:
            self.freq_perturb = FrequencyAwarePerturbation(perturb_high=True, eps_high=freq_perturb_eps, low_freq_ratio=0.25)

    def set_progress(self, progress):
        if self.mixstyle1 is not None and hasattr(self.mixstyle1, 'set_progress'):
            self.mixstyle1.set_progress(progress)
        if self.mixstyle2 is not None and hasattr(self.mixstyle2, 'set_progress'):
            self.mixstyle2.set_progress(progress)

    def forward(self, x, domain_labels=None, return_both=False):
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.relu(x)
        x = self.maxpool(f1)
        f2 = self.layer1(x)
        if return_both and self.mixstyle1 is not None:
            f2_original = f2
            f2_mixed = self.mixstyle1(f2.clone(), domain_labels)
        elif self.mixstyle1 is not None:
            f2_original = self.mixstyle1(f2, domain_labels)
            f2_mixed = None
        else:
            f2_original = f2
            f2_mixed = None
        f3_original = self.layer2(f2_original)
        f3_mixed = self.layer2(f2_mixed) if return_both and f2_mixed is not None else None
        if return_both and self.mixstyle2 is not None and (f3_mixed is not None):
            f3_mixed = self.mixstyle2(f3_mixed.clone(), domain_labels)
        elif self.mixstyle2 is not None and (not return_both):
            f3_original = self.mixstyle2(f3_original, domain_labels)
        f4_original = self.layer3(f3_original)
        f5_original = self.layer4(f4_original)
        if return_both:
            f4_mixed = self.layer3(f3_mixed)
            f5_mixed = self.layer4(f4_mixed)
            return ([f1, f2_original, f3_original, f4_original, f5_original], [f1, f2_mixed, f3_mixed, f4_mixed, f5_mixed])
        return [f1, f2_original, f3_original, f4_original, f5_original]

    def forward_with_freq_perturb(self, x, apply_perturb=True):
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.relu(x)
        x = self.maxpool(f1)
        f2 = self.layer1(x)
        if apply_perturb and self.freq_perturb is not None:
            f2 = self.freq_perturb(f2)
        f3 = self.layer2(f2)
        if apply_perturb and self.freq_perturb is not None:
            f3 = self.freq_perturb(f3)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return [f1, f2, f3, f4, f5]

class UNetDecoder(nn.Module):

    def __init__(self, encoder_channels, num_classes=1):
        super().__init__()
        self.up4 = self._up_block(encoder_channels[4], encoder_channels[3])
        self.up3 = self._up_block(encoder_channels[3], encoder_channels[2])
        self.up2 = self._up_block(encoder_channels[2], encoder_channels[1])
        self.up1 = self._up_block(encoder_channels[1], encoder_channels[0])
        self.final = nn.Sequential(nn.Conv2d(encoder_channels[0], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, num_classes, 1))

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        x = self.up4(f5)
        x = x + f4
        x = self.up3(x)
        x = x + f3
        x = self.up2(x)
        x = x + f2
        x = self.up1(x)
        x = x + f1
        return self.final(x)

class StudentModelWithMixStyle(nn.Module):

    def __init__(self, backbone='resnet50', num_classes=1, pretrained=True, mixstyle_layers=['layer1', 'layer2'], mixstyle_p=0.2, mixstyle_alpha=0.1, adaptive_mixstyle=True, use_freq_perturb=False, freq_perturb_eps=0.2):
        super().__init__()
        self.encoder = ResNetEncoderWithMixStyle(backbone=backbone, pretrained=pretrained, mixstyle_layers=mixstyle_layers, mixstyle_p=mixstyle_p, mixstyle_alpha=mixstyle_alpha, adaptive_mixstyle=adaptive_mixstyle, use_freq_perturb=use_freq_perturb, freq_perturb_eps=freq_perturb_eps)
        self.decoder = UNetDecoder(self.encoder.channels, num_classes)

    def forward(self, x, domain_labels=None, return_both=False):
        if return_both:
            features_original, features_mixed = self.encoder(x, domain_labels, return_both=True)
            out_original = self.decoder(features_original)
            out_mixed = self.decoder(features_mixed)
            out_original = F.interpolate(out_original, size=x.shape[2:], mode='bilinear', align_corners=False)
            out_mixed = F.interpolate(out_mixed, size=x.shape[2:], mode='bilinear', align_corners=False)
            return (out_original, out_mixed)
        else:
            features = self.encoder(x, domain_labels)
            out = self.decoder(features)
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            return out

    def set_progress(self, progress):
        self.encoder.set_progress(progress)

class MeanTeacherWithMixStyle(nn.Module):

    def __init__(self, backbone='resnet50', num_classes=1, ema_decay=0.999, pretrained=True, mixstyle_layers=['layer1', 'layer2'], mixstyle_p=0.2, mixstyle_alpha=0.1, adaptive_mixstyle=True, use_freq_perturb=False, freq_perturb_eps=0.2):
        super().__init__()
        self.student = StudentModelWithMixStyle(backbone=backbone, num_classes=num_classes, pretrained=pretrained, mixstyle_layers=mixstyle_layers, mixstyle_p=mixstyle_p, mixstyle_alpha=mixstyle_alpha, adaptive_mixstyle=adaptive_mixstyle, use_freq_perturb=use_freq_perturb, freq_perturb_eps=freq_perturb_eps)
        self.ema_teacher = EMATeacher(self.student, decay=ema_decay)
        self.ema_decay = ema_decay
        self.use_freq_perturb = use_freq_perturb
        self.freq_perturb_eps = freq_perturb_eps

    def forward(self, x_weak, x_strong=None, domain_labels=None):
        results = {}
        with torch.no_grad():
            pred_teacher = self.ema_teacher(x_weak)
            results['pred_teacher'] = pred_teacher
        self.student.train()
        if x_strong is not None:
            pred_student = self.student(x_strong, domain_labels)
        else:
            pred_student = self.student(x_weak, domain_labels)
        results['pred_student'] = pred_student
        return results

    def update_teacher(self):
        student_model = self.student.module if hasattr(self.student, 'module') else self.student
        self.ema_teacher.update(student_model)

    def student_forward(self, x, domain_labels=None, return_both=False):
        return self.student(x, domain_labels, return_both=return_both)

    def teacher_forward(self, x):
        with torch.no_grad():
            return self.ema_teacher(x)

    def set_mixstyle_active(self, active):
        enc = self.student.encoder
        if enc.mixstyle1 is not None:
            enc.mixstyle1.set_activation(active)
        if enc.mixstyle2 is not None:
            enc.mixstyle2.set_activation(active)

    def set_progress(self, progress):
        student_model = self.student.module if hasattr(self.student, 'module') else self.student
        student_model.set_progress(progress)

    def student_encode(self, x, domain_labels=None):
        student_model = self.student.module if hasattr(self.student, 'module') else self.student
        features = student_model.encoder(x, domain_labels)
        return features

    def student_decode(self, features, target_size):
        student_model = self.student.module if hasattr(self.student, 'module') else self.student
        out = student_model.decoder(features)
        out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out

    def forward_with_freq_perturb(self, x, apply_perturb=True):
        if not self.use_freq_perturb or not apply_perturb:
            return self.student(x, domain_labels=None, return_both=False)
        student_model = self.student.module if hasattr(self.student, 'module') else self.student
        student_model.encoder.train()
        if student_model.encoder.freq_perturb is not None:
            student_model.encoder.freq_perturb.train()
        features_perturbed = student_model.encoder.forward_with_freq_perturb(x, apply_perturb=True)
        pred = student_model.decoder(features_perturbed)
        pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return pred
if __name__ == '__main__':
    print('测试MeanTeacher + MixStyle模型...')
    model = MeanTeacherWithMixStyle(backbone='resnet50', num_classes=1, mixstyle_layers=['layer1', 'layer2'], adaptive_mixstyle=True)
    model = model.cuda()
    model.train()
    B = 8
    x_weak = torch.randn(B, 3, 256, 256).cuda()
    x_strong = torch.randn(B, 3, 256, 256).cuda()
    domain_labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2]).cuda()
    outputs = model(x_weak, x_strong, domain_labels)
    print(f"Teacher pred: {outputs['pred_teacher'].shape}")
    print(f"Student pred: {outputs['pred_student'].shape}")
    for progress in [0.0, 0.3, 0.5, 1.0]:
        model.set_progress(progress)
        print(f'Progress={progress}')
    model.update_teacher()
    print('EMA update done!')
    print(f'\nStudent参数量: {sum((p.numel() for p in model.student.parameters())):,}')
    print('测试通过!')
