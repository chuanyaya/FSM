# FSM

# Requirements


1.  Create conda environment:
    ```
    conda create -n FSM python=3.11
    ```

2.  Activate the environment:
    ```
    conda activate FSM
    ```
3.  Install the requirements:
    ```
    pip install -r requirements.txt
    ```

# Usage



### KiTS23_to_KiTS19

One click to run:
```
cd /KiTS23_to_KiTS19
python train_fsm.py --test --gpu 0 --labeled_ratio 10p --checkpoint checkpoints/fsm/best_model_10p.pth
python train_fsm.py --test --gpu 0 --labeled_ratio 20p --checkpoint checkpoints/fsm/best_model_20p.pth
python train_baseline.py --test --gpu 0 --labeled_ratio 10p --checkpoint checkpoints/baseline/best_model_10p.pth
python train_baseline.py --test --gpu 0 --labeled_ratio 20p --checkpoint checkpoints/baseline/best_model_20p.pth
```
### Kvasir-SEG_to_EndoScene

One click to run:
```
cd Kvasir-SEG_to_EndoScene
bash scripts/train.sh gpu_num port
python experiments/code/train_FSM.py --mode test --checkpoint checkpoints/10p.pth --gpu 0
python experiments/code/train_FSM.py --mode test --checkpoint checkpoints/30p.pth --gpu 0
python experiments/code/train_baseline.py --mode test --checkpoint checkpoints/10p_base.pth --gpu 0
python experiments/code/train_baseline.py --mode test --checkpoint checkpoints/30p_base.pth --gpu 0
```


# Results
## Plug-and-play results on cross-domain datasets

We evaluate the plug-and-play performance of integrating **FSM** into different SSDA frameworks on two cross-domain medical image segmentation benchmarks.

### Kvasir-SEG → EndoScene (Polyp)
- The target domain uses **10% labeled / 30% labeled** settings.
- Evaluation metrics include **Dice** and **IoU**.

| Method          | Reference | 10% Dice(%)↑ | 10% IoU(%)↑ | 30% Dice(%)↑ | 30% IoU(%)↑ |
|:----------------|:---------:|-------------:|------------:|-------------:|------------:|
| Baseline        |           | 78.76        | 64.97       | 84.10        | 72.56       |
| DuCiSC          | [24]      | 68.86        | 52.50       | 81.89        | 69.34       |
| DuCiSC + FSM    | [24]      | 80.61        | 67.52       | 87.20        | 77.30       |
| CorrMatch       | [25]      | 81.10        | 68.21       | 87.14        | 77.21       |
| CorrMatch + FSM | [25]      | 85.28        | 74.34       | 90.38        | 82.45       |
| ABD             | [26]      | 81.39        | 68.63       | 85.06        | 74.01       |
| ABD + FSM       | [26]      | 84.15        | 72.64       | 86.33        | 75.95       |
| **Ours**        |           | **87.39**    | **77.60**   | **91.40**    | **84.16**   |

### KiTS23 → KiTS19 (Kidney Tumor)
- The target domain uses **10% labeled / 20% labeled** settings.
- Evaluation metrics include **mDice** and **mIoU**.

| Method          | Reference | 10% mDice(%)↑ | 10% mIoU(%)↑ | 20% mDice(%)↑ | 20% mIoU(%)↑ |
|:----------------|:---------:|--------------:|-------------:|--------------:|-------------:|
| Baseline        |           | 82.61         | 70.74        | 87.92         | 78.68        |
| DuCiSC          | [24]      | 82.26         | 70.19        | 85.75         | 75.31        |
| DuCiSC + FSM    | [24]      | 86.35         | 76.21        | 88.63         | 79.80        |
| CorrMatch       | [25]      | 78.16         | 64.60        | 84.46         | 73.39        |
| CorrMatch + FSM | [25]      | 81.07         | 68.57        | 86.00         | 75.78        |
| ABD             | [26]      | 80.29         | 67.61        | 87.00         | 77.24        |
| ABD + FSM       | [26]      | 83.94         | 72.74        | 88.41         | 79.51        |
| **Ours**        |           | **86.53**     | **76.61**    | **89.46**     | **81.16**    |

### ACDC dataset results
-   The training set consists of 3 labeled scans and 67 unlabeled scans and the testing set includes 20 scans.

| Method      | Reference      | Dice(%)↑   | Jaccard(%)↑ | 95HD(voxel)↓ | ASD(voxel)↓ |
|:------------|:-------------:|-----------:|-----------:|-------------:|-----------:|
| UA-MT       | (MICCAI'19)   | 46.04      | 35.97      | 20.08        | 7.75       |
| SASSNet     | (MICCAI'20)   | 57.77      | 46.14      | 20.05        | 6.06       |
| DTC         | (AAAI'21)     | 56.90      | 45.67      | 23.36        | 7.39       |
| MC-Net      | (MICCAI'21)   | 62.85      | 52.29      | 7.62         | 2.33       |
| URPC        | (MedIA'22)    | 55.87      | 44.64      | 13.60        | 3.74       |
| SS-Net      | (MICCAI'22)   | 65.82      | 55.38      | 6.67         | 2.28       |
| DMD         | (MICCAI'23)   | 80.60      | 69.08      | 5.96         | 1.90       |
| ABD    | (CVPR'24)     | 88.96      | 80.70      | 1.57       | 0.52       |
| **Ours** | |**89.68** |**82.31** |**1.33**| **0.37** |
-   The training set consists of 7 labeled scans and 63 unlabeled scans and the testing set includes 20 scans.

| Method    | Reference    | Dice(%)↑ | Jaccard(%)↑ | 95HD(voxel)↓ | ASD(voxel)↓ |
|:----------|:-----------:|---------:|-----------:|-------------:|-----------:|
| UA-MT     | (MICCAI'19) | 81.65    | 70.64      | 6.88         | 2.02       |
| SASSNet   | (MICCAI'20) | 84.50    | 74.34      | 5.42         | 1.86       |
| DTC       | (AAAI'21)   | 84.29    | 73.92      | 12.81        | 4.01       |
| MC-Net    | (MICCAI'21) | 86.44    | 77.04      | 5.50         | 1.84       |
| URPC      | (MedIA'22)  | 83.10    | 72.41      | 4.84         | 1.53       |
| SS-Net    | (MICCAI'22) | 86.78    | 77.67      | 6.07         | 1.40       |
| DMD       | (MICCAI'23) | 87.52    | 78.62      | 4.81         | 1.60       |
| ABD    | (CVPR'24)     | 89.81      | 81.95      | 1.46       | 0.49       |
| **Ours**  |             | **90.19**| **83.00**  | **1.35**     | **0.37**       |



# Acknowledgement
-   This code is adapted from  [UA-MT](https://github.com/yulequan/UA-MT),  [DTC](https://github.com/HiLab-git/DTC.git)  and  [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical)  .
-   We thank Lequan Yu, Xiangde Luo and Lihe Yang for their elegant and efficient code base.
