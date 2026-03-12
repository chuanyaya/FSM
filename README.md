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

## Comparison with state-of-the-art SSDA methods

We compare our method with representative SSDA approaches on two cross-domain medical image segmentation tasks.

### Kvasir-SEG → EndoScene (Polyp)
- The target domain uses **10% labeled / 30% labeled** settings.
- Evaluation metrics include **Dice** and **IoU**.

| Method   | Reference | 10% Dice(%)↑ | 10% IoU(%)↑ | 30% Dice(%)↑ | 30% IoU(%)↑ |
|:---------|:---------:|-------------:|------------:|-------------:|------------:|
| ACT      | [4]       | 75.52        | 67.20       | 79.88        | 71.89       |
| SLA      | [27]      | 75.23        | 66.47       | 82.90        | 75.33       |
| GFDA     | [14]      | 85.03        | 73.95       | 87.95        | 78.49       |
| **Ours** |           | **87.39**    | **77.60**   | **91.40**    | **84.16**   |

### KiTS23 → KiTS19 (Kidney Tumor)
- The target domain uses **10% labeled / 20% labeled** settings.
- Evaluation metrics include **mDice** and **mIoU**.

| Method   | Reference | 10% mDice(%)↑ | 10% mIoU(%)↑ | 20% mDice(%)↑ | 20% mIoU(%)↑ |
|:---------|:---------:|--------------:|-------------:|--------------:|-------------:|
| ACT      | [4]       | 82.09         | 75.38        | 86.41         | 81.09        |
| GFDA     | [14]      | 85.88         | 75.53        | 89.08         | 80.50        |
| SLA      | [27]      | 83.03         | 71.48        | 85.44         | 74.99        |
| **Ours** |           | **86.53**     | **76.61**    | **89.46**     | **81.16**    |


## Ablation study of FSM and FP components

We conduct an ablation study on **FSM** and **FP (Frequency Perturbation)** over two cross-domain medical image segmentation tasks.

### Kvasir-SEG → EndoScene (Polyp)
- The target domain uses **10% labeled / 30% labeled** settings.
- Evaluation metrics include **Dice** and **IoU**.

| Configuration    | 10% Dice(%)↑     | 10% IoU(%)↑      | 30% Dice(%)↑     | 30% IoU(%)↑      |
|:-----------------|-----------------:|-----------------:|-----------------:|------------------:|
| Baseline         | 78.76            | 64.97            | 84.10            | 72.56             |
| Baseline + FP    | 83.99 (+5.23)    | 72.40 (+7.43)    | 88.81 (+4.71)    | 79.87 (+7.31)     |
| Baseline + FSM   | 83.58 (+4.82)    | 71.80 (+6.83)    | 88.68 (+4.58)    | 79.66 (+7.10)     |
| **Ours (FSM + FP)** | **87.39 (+8.63)** | **77.60 (+12.63)** | **91.40 (+7.30)** | **84.16 (+11.60)** |

### KiTS23 → KiTS19 (Kidney Tumor)
- The target domain uses **10% labeled / 20% labeled** settings.
- Evaluation metrics include **mDice** and **mIoU**.

| Configuration    | 10% mDice(%)↑    | 10% mIoU(%)↑     | 20% mDice(%)↑    | 20% mIoU(%)↑     |
|:-----------------|-----------------:|-----------------:|-----------------:|-----------------:|
| Baseline         | 82.61            | 70.74            | 86.85            | 77.03            |
| Baseline + FP    | 85.35 (+2.74)    | 74.68 (+3.94)    | 87.47 (+0.62)    | 78.00 (+0.97)    |
| Baseline + FSM   | 86.47 (+3.86)    | 76.50 (+5.76)    | 88.54 (+1.69)    | 79.69 (+2.66)    |
| **Ours (FSM + FP)** | **86.53 (+3.92)** | **76.61 (+5.87)** | **89.46 (+2.61)** | **81.16 (+4.13)** |



# Acknowledgement
-   This code is adapted from  [UA-MT](https://github.com/yulequan/UA-MT),  [DTC](https://github.com/HiLab-git/DTC.git)  and  [UniMatch](https://github.com/LiheYoung/UniMatch/tree/main/more-scenarios/medical)  .
-   We thank Lequan Yu, Xiangde Luo and Lihe Yang for their elegant and efficient code base.
