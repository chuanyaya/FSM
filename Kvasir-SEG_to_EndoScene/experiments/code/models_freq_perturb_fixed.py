import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from style_transfer import FSM, CrossNorm, AdaptiveFSM, StyleAugmentor
from freq_perturbation import FrequencyAwarePerturbation, AdaptiveFrequencyPerturbation

class EMATeacher:

    def __init__(self, student_model, decay=0.999):
        self.decay = decay
        self.teacher = self._create_clean_teacher(student_model)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _create_clean_teacher(self, student_model):
        teacher = copy.deepcopy(student_model)
        for module in teacher.modules():
            if isinstance(module, (FSM, CrossNorm, AdaptiveFSM, StyleAugmentor)):
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

class ResNetEncoderWithFSM(nn.Module):

    def __init__(self, backbone='resnet50', pretrained=True, fsm_layers=['layer1', 'layer2'], fsm_p=0.2, fsm_alpha=0.1, adaptive_fsm=True, use_freq_perturb=False, freq_perturb_eps=0.2):
        super().__init__()
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
        self.fsm_layers = fsm_layers
        self.fsm1 = None
        self.fsm2 = None
        if 'layer1' in fsm_layers:
            if adaptive_fsm:
                self.fsm1 = AdaptiveFSM(p_start=0.3, p_end=0.1, alpha=fsm_alpha, mix='crossdomain')
            else:
                self.fsm1 = FSM(p=fsm_p, alpha=fsm_alpha, mix='crossdomain')
        if 'layer2' in fsm_layers:
            if adaptive_fsm:
                self.fsm2 = AdaptiveFSM(p_start=0.3, p_end=0.1, alpha=fsm_alpha, mix='crossdomain')
            else:
                self.fsm2 = FSM(p=fsm_p, alpha=fsm_alpha, mix='crossdomain')
        self.use_freq_perturb = use_freq_perturb
        self.freq_perturb = None
        if use_freq_perturb:
            self.freq_perturb = FrequencyAwarePerturbation(perturb_high=True, eps_high=freq_perturb_eps, low_freq_ratio=0.25)

    def set_progress(self, progress):
        if self.fsm1 is not None and hasattr(self.fsm1, 'set_progress'):
            self.fsm1.set_progress(progress)
        if self.fsm2 is not None and hasattr(self.fsm2, 'set_progress'):
            self.fsm2.set_progress(progress)

    def forward(self, x, domain_labels=None, return_both=False):
        x = self.conv1(x)
        x = self.bn1(x)
        f1 = self.relu(x)
        x = self.maxpool(f1)
        f2 = self.layer1(x)
        if return_both and self.fsm1 is not None:
            f2_original = f2
            f2_mixed = self.fsm1(f2.clone(), domain_labels)
        else:
            f2_original = f2
            f2_mixed = self.fsm1(f2, domain_labels) if self.fsm1 is not None else f2
        f3_original = self.layer2(f2_original)
        f3_mixed = self.layer2(f2_mixed) if return_both else None
        if return_both and self.fsm2 is not None:
            f3_mixed = self.fsm2(f3_mixed.clone(), domain_labels)
        elif self.fsm2 is not None and (not return_both):
            f3_original = self.fsm2(f3_original, domain_labels)
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

class StudentModelWithFSM(nn.Module):

    def __init__(self, backbone='resnet50', num_classes=1, pretrained=True, fsm_layers=['layer1', 'layer2'], fsm_p=0.2, fsm_alpha=0.1, adaptive_fsm=True, use_freq_perturb=False, freq_perturb_eps=0.2):
        super().__init__()
        self.encoder = ResNetEncoderWithFSM(backbone=backbone, pretrained=pretrained, fsm_layers=fsm_layers, fsm_p=fsm_p, fsm_alpha=fsm_alpha, adaptive_fsm=adaptive_fsm, use_freq_perturb=use_freq_perturb, freq_perturb_eps=freq_perturb_eps)
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

class MeanTeacherWithFSM(nn.Module):

    def __init__(self, backbone='resnet50', num_classes=1, ema_decay=0.999, pretrained=True, fsm_layers=['layer1', 'layer2'], fsm_p=0.2, fsm_alpha=0.1, adaptive_fsm=True, use_freq_perturb=False, freq_perturb_eps=0.2):
        super().__init__()
        self.student_encoder = ResNetEncoderWithFSM(backbone=backbone, pretrained=pretrained, fsm_layers=fsm_layers, fsm_p=fsm_p, fsm_alpha=fsm_alpha, adaptive_fsm=adaptive_fsm, use_freq_perturb=use_freq_perturb, freq_perturb_eps=freq_perturb_eps)
        self.student_decoder = UNetDecoder(self.student_encoder.channels, num_classes)
        self.student = StudentModelWithFSM(backbone=backbone, num_classes=num_classes, pretrained=pretrained, fsm_layers=fsm_layers, fsm_p=fsm_p, fsm_alpha=fsm_alpha, adaptive_fsm=adaptive_fsm, use_freq_perturb=use_freq_perturb, freq_perturb_eps=freq_perturb_eps)
        self.ema_teacher = EMATeacher(self.student, decay=ema_decay)
        self.ema_decay = ema_decay

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

    def set_progress(self, progress):
        student_model = self.student.module if hasattr(self.student, 'module') else self.student
        student_model.set_progress(progress)
        self.student_encoder.set_progress(progress)

    def forward_with_freq_perturb(self, x, apply_perturb=True):
        features = self.student_encoder.forward_with_freq_perturb(x, apply_perturb=apply_perturb)
        out = self.student_decoder(features)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
if __name__ == '__main__':
    print('测试MeanTeacher + FSM模型...')
    model = MeanTeacherWithFSM(backbone='resnet50', num_classes=1, fsm_layers=['layer1', 'layer2'], adaptive_fsm=True)
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
