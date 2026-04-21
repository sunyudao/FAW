# FAW: A Framework-Agnostic and Scalable Model Watermarking Scheme for Distributed Learning

This repository contains the official implementation of **FAW**, the first framework-agnostic and scalable model watermarking framework for distributed learning (DL). FAW generalizes across diverse DL architectures — including Federated Learning (FL), Split Learning (SL), Parallel Split Learning (PSL), and Split Federated Learning (SFL) — while preserving main-task fidelity.


### Key Features

- **Framework-Agnostic**: Works across FL, SL, PSL, SFL, and variants (e.g., Asynchronous FL, Hierarchical SL).
- **Non-Intrusive**: Watermark generation is decoupled from training — zero degradation (Δ=0) on main-task accuracy.
- **Scalable**: Achieves >98% watermark detection accuracy with up to 80 clients.
- **Robust**: Resilient against fine-tuning, pruning, quantization, backdoor, and data poisoning attacks.
- **Client-Specific Soft Masking**: Ensures distinguishability and robustness via randomized amplitude modulation.

## Repository Structure

```
├── utils/
│   ├── dataset_utils.py        # Dataset loading utilities
│   ├── model_utils.py          # Model architecture definitions
│   ├── soft_mask_watermark.py  # Watermark Generation Methods
│   └── data_distribution.py    # Dataset Partitioning
├── step1_training.py           # Step 1: Distributed learning training (FL/SL/PSL/SFL)
├── step2_gen_watermarks.py     # Step 2: Watermark sample generation
├── step3_verification.py       # Step 3: Watermark verification
├── run.sh                      # End-to-end execution script
├── requirements.txt            # Python dependencies
└── README.md
```

## Pipeline

FAW follows a three-step pipeline:

### Step 1: Model Training

Train models under four distributed learning paradigms:

```bash
python step1_training.py \
    --num_clients 10 \
    --model lenet \
    --dataset mnist \
    --num_rounds 20 \
    --learning_rate 0.01 \
    --batch_size 64 \
    --local_epochs 1 \
    --iid true \
    --experiments_dir ./experiments \
    --data_dir ./data
```

This trains FL, SL, PSL, and SFL models sequentially and saves checkpoints to `./experiments/checkpoints/{FL,SL,PSL,SFL}/`.

### Step 2: Watermark Generation

Generate client-specific watermark samples using soft-mask–guided PGD optimization:

```bash
python step2_gen_watermarks.py \
    --num_clients 10 \
    --model lenet \
    --dataset mnist \
    --pgd_eps 0.8 \
    --pgd_alpha 0.015 \
    --pgd_steps 500 \
    --mask_epsilon 0.05 \
    --cleanset_max 200 \
    --seed 2026 \
    --experiments_dir ./experiments \
    --data_dir ./data \
    --attack_type pdg
```

Watermark artifacts (clean set, masks, target sequences, watermarked examples) are saved to `./experiments/watermarks/{margin, pgd, mi_fgsm, ni_fgsm, si_ni_fgsm, vmi_fgsm, emi_fgsm}/{FL,SL,PSL,SFL}/`.

### Step 3: Watermark Verification

Verify watermark effectiveness by measuring watermark success rate and clean accuracy across all clients:

```bash
python step3_verification.py \
    --num_clients 10 \
    --model lenet \
    --dataset mnist \
    --experiments_dir ./experiments \
    --data_dir ./data
    --attack_type pdg
```

## Quick Start

Run the full pipeline end-to-end:

```bash
bash run.sh
```

## Supported Configurations

| Dataset       | Model             | Resolution |
|---------------|-------------------|------------|
| MNIST         | LeNet             | 28×28      |
| CIFAR-10      | ResNet18          | 32×32      |
| Tiny-ImageNet | VGG16             | 64×64      |
| ImageNet      | Transformer (ViT) | 224×224    |

| DL Paradigm | Description                                    |
|-------------|------------------------------------------------|
| FL          | Federated Learning (FedAvg)                    |
| SL          | Split Learning (Sequential)                    |
| PSL         | Parallel Split Learning                        |
| SFL         | Split Federated Learning (SFLV2)               |

## Key Parameters

| Parameter        | Description                                  | Default |
|------------------|----------------------------------------------|---------|
| `--num_clients`  | Number of clients                            | 10      |
| `--pgd_eps`      | Maximum perturbation budget (ℓ∞)             | 0.8     |
| `--pgd_alpha`    | PGD step size                                | 0.015   |
| `--pgd_steps`    | Number of PGD iterations                     | 500     |
| `--mask_epsilon`  | Soft mask randomization parameter (η)       | 0.05    |
| `--cleanset_max` | Number of clean samples for watermarking     | 200     |

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: PyTorch, torchvision, NumPy.

## Citation

```bibtex
@inproceedings{faw2025,
  title={FAW: A Framework-Agnostic and Scalable Model Watermarking Scheme for Distributed Learning},
  author={Anonymous},
  booktitle={Proceedings of Conference},
  year={2025}
}
```

## License

This project is released for academic research purposes.
