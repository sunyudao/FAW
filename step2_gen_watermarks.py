"""
Step 2: Generate Watermarks for Distributed Learning Models

Generates targeted adversarial examples (watermarks) for each client across
FL, SL, PSL, and SFL frameworks. Uses masked PGD attacks to create
client-specific watermarks that are saved for later verification.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import copy

from utils.model_utils import get_models
from utils.dataset_utils import get_dataset


# ============== Target Sequence Generator ==============
def generate_wrong_labels(labels, num_classes):
    """Generate wrong labels for each sample (different from true label)."""
    labels = labels.cpu().numpy()
    out = []
    for y in labels:
        cand = [c for c in range(num_classes) if c != y]
        out.append(np.random.choice(cand))
    return torch.tensor(out, dtype=torch.long)


def generate_target_sequences(
    true_labels: torch.Tensor, num_clients: int, num_classes: int, seed: int
) -> list:
    """
    Generate unique target sequences for each client.

    Each client receives a distinct set of target labels that differ from
    the true labels, enabling client-specific watermark verification.
    """
    assert true_labels.dim() == 1, "true_labels must be a 1D tensor"

    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    target_seqs = []
    for client_id in range(num_clients):
        # Use different seed per client to ensure uniqueness
        np.random.seed(seed + client_id)
        target_seq = generate_wrong_labels(true_labels, num_classes)
        target_seqs.append(target_seq)

    # Reset seed
    np.random.seed(seed)

    return target_seqs


# ============== Mask Generator ==============
def generate_soft_masks(num_clients, shape, mask_epsilon):
    """
    Generate soft masks with randomized values for watermark embedding.

    Each client receives a non-overlapping region of the input space.
    Mask values are randomized to [1-mask_epsilon, 1] for assigned regions
    and [0, mask_epsilon] for non-assigned regions.
    """
    flat_dim = int(np.prod(shape))
    indices = np.random.permutation(flat_dim)

    masks = torch.zeros((num_clients, flat_dim), dtype=torch.float32)
    size_per_client = flat_dim // num_clients

    for i in range(num_clients):
        start = i * size_per_client
        end = (i + 1) * size_per_client if i != num_clients - 1 else flat_dim
        masks[i, indices[start:end]] = 1.0

    # Randomize mask values
    high = torch.rand_like(masks) * mask_epsilon + (1 - mask_epsilon)  # [1-epsilon, 1]
    low = torch.rand_like(masks) * mask_epsilon  # [0, epsilon]

    masks = masks * high + (1 - masks) * low
    masks = masks.view(num_clients, *shape)

    return masks


# ============== Margin Attack for Watermark Generation ==============
def margin_attack_targeted(
    model, images, target_labels, eps, alpha, steps, mask, device, method_type="FL"
):
    """
    Targeted margin attack maximizing softmax[target] - max(other softmax).

    Preserves masked regions during the attack to ensure watermarks are
    embedded only in the assigned pixel regions.
    """
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)

    images.requires_grad = True

    for step in range(steps):
        # Forward pass (method-dependent)
        if method_type == "FL":
            logits = model(images * (1 - mask))
        elif method_type in ["SL", "PSL", "SFL"]:
            smashed = model["client"](images * (1 - mask))
            logits = model["server"](smashed)
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        softmax = F.softmax(logits, dim=1)

        # Get softmax score for target label
        target_scores = softmax.gather(1, target_labels.view(-1, 1)).squeeze(1)

        # Get max softmax score for other classes
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(1, target_labels.view(-1, 1), 1.0)
        other_scores = softmax * (1 - one_hot)
        max_other_scores, _ = other_scores.max(dim=1)

        # Loss: maximize (target_score - max_other_score)
        loss = -(target_scores - max_other_scores).mean()

        # Backward pass
        loss.backward()

        # PGD update
        with torch.no_grad():
            adv = images - alpha * images.grad.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)

            # Apply mask: preserve original values in masked region
            images = images * (1 - mask) + ori_images * mask

        images = images.detach()
        images.requires_grad = True

    return images


def pgd_attack_targeted(
    model, images, target_labels, eps, alpha, steps, mask, device, method_type="FL"
):
    """
    Targeted PGD attack minimizing cross-entropy loss for target labels.

    Preserves masked regions during the attack to ensure watermarks are
    embedded only in the assigned pixel regions.
    """
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)

    images.requires_grad = True
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        # Forward pass (method-dependent)
        if method_type == "FL":
            logits = model(images * (1 - mask))
        elif method_type in ["SL", "PSL", "SFL"]:
            smashed = model["client"](images * (1 - mask))
            logits = model["server"](smashed)
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        # Loss: minimize cross-entropy for target label (targeted attack)
        loss = criterion(logits, target_labels)

        # Backward pass
        loss.backward()

        # PGD update (targeted: subtract gradient to minimize loss)
        with torch.no_grad():
            adv = images - alpha * images.grad.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)

            # Apply mask: preserve original values in masked region
            images = images * (1 - mask) + ori_images * mask

        images = images.detach()
        images.requires_grad = True

    return images


# ============== Evaluation Functions ==============
@torch.no_grad()
def evaluate_watermark_success(
    model, watermarked_images, target_labels, mask, device, method_type="FL", batch_size=64
):
    """Evaluate targeted attack success rate on watermarked images."""
    total, success = 0, 0

    for i in range(0, watermarked_images.size(0), batch_size):
        x = watermarked_images[i : i + batch_size].to(device)
        y = target_labels[i : i + batch_size].to(device)
        m = mask.to(device)

        # Apply mask and forward pass
        if method_type == "FL":
            preds = model(x * (1 - m)).argmax(dim=1)
        elif method_type in ["SL", "PSL", "SFL"]:
            smashed = model["client"](x * (1 - m))
            preds = model["server"](smashed).argmax(dim=1)
        else:
            raise ValueError(f"Unknown method_type: {method_type}")

        success += (preds == y).sum().item()
        total += y.numel()

    return success / total if total > 0 else 0.0


def create_clean_set(dataset, max_samples=384, seed=2025):
    """Create a clean set of samples for watermark generation."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:max_samples]

    images, labels = [], []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    clean_images = torch.stack(images)
    true_labels = torch.tensor(labels, dtype=torch.long)

    return clean_images, true_labels


# ============== Model Loading Functions ==============
def load_fl_model(checkpoint_path, num_classes, device):
    """Load Federated Learning global model."""
    model = FullModel(num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_split_models(client_checkpoint_path, server_checkpoint_path, num_classes, device):
    """Load Split Learning models (client and server components)."""
    client_model = ClientModel().to(device)
    server_model = ServerModel(num_classes).to(device)

    # Load client model
    client_ckpt = torch.load(client_checkpoint_path, map_location=device)
    if "model_state_dict" in client_ckpt:
        client_model.load_state_dict(client_ckpt["model_state_dict"])
    else:
        client_model.load_state_dict(client_ckpt)

    # Load server model
    server_ckpt = torch.load(server_checkpoint_path, map_location=device)
    if "model_state_dict" in server_ckpt:
        server_model.load_state_dict(server_ckpt["model_state_dict"])
    else:
        server_model.load_state_dict(server_ckpt)

    client_model.eval()
    server_model.eval()

    return {"client": client_model, "server": server_model}


# ============== Main Function ==============
def main():
    parser = argparse.ArgumentParser(description="Generate watermarks for distributed learning models" )
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--experiments_dir", type=str, default="./experiments")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cleanset_max", type=int, default=200, help="Maximum number of clean samples for watermarking")
    parser.add_argument("--pgd_eps", type=float, default=0.8, help="Maximum perturbation for PGD attack" )
    parser.add_argument("--pgd_alpha", type=float, default=0.015, help="Step size for PGD attack")
    parser.add_argument("--pgd_steps", type=int, default=500, help="Number of PGD iterations")
    parser.add_argument("--mask_epsilon", type=float, default=0.05, help="Epsilon for continuous mask randomization")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Configuration for {os.path.basename(__file__)}")
    print(f"{'='*50}")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print(f"{'='*50}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model architecture classes
    global ClientModel, ServerModel, FullModel
    ClientModel, ServerModel, FullModel = get_models(args.model)

    # Load dataset
    _, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)

    # Create clean set for watermark generation
    print(f"\nCreating clean set with {args.cleanset_max} samples...")
    clean_images, true_labels = create_clean_set(test_dataset, args.cleanset_max, args.seed)
    print(f"Clean set shape: {clean_images.shape}")

    # Generate target sequences for each client
    print("\nGenerating target sequences for each client...")
    target_seqs = generate_target_sequences(
        true_labels, args.num_clients, num_classes, seed=args.seed
    )
    print(f"Generated {len(target_seqs)} unique target sequences")

    # Generate masks
    print("\nGenerating non-overlapping masks...")
    mask_shape = tuple(clean_images[0].shape)  # (C, H, W)
    all_masks = generate_soft_masks(args.num_clients, mask_shape, mask_epsilon=args.mask_epsilon)
    print(f"Masks shape: {all_masks.shape}")

    # Process each distributed learning method
    methods_list = ["FL", "SL", "PSL", "SFL"]

    for method in methods_list:
        print(f"\n{'#'*60}")
        print(f"Processing method: {method}")
        print(f"{'#'*60}")

        # Initialize paths and result containers
        watermark_dir = os.path.join(args.experiments_dir, "watermarks", method)
        checkpoint_dir = os.path.join(args.experiments_dir, "checkpoints", method)
        os.makedirs(watermark_dir, exist_ok=True)

        watermarked_examples = []
        success_rates = []

        # Iterate through clients
        for client_id in range(args.num_clients):
            print(f"\n[{method}] Processing Client {client_id + 1}/{args.num_clients}")

            # Load model based on method type
            if method == "FL":
                checkpoint_path = os.path.join(checkpoint_dir, "global_model.pt")
                if not os.path.exists(checkpoint_path):
                    print(f"  Warning: FL model not found at {checkpoint_path}, skipping...")
                    continue
                model = load_fl_model(checkpoint_path, num_classes, device)

            elif method == "SL":
                client_path = os.path.join(checkpoint_dir, "client_model.pt")
                server_path = os.path.join(checkpoint_dir, "server_model.pt")
                if not os.path.exists(client_path) or not os.path.exists(server_path):
                    print(f"  Warning: SL models not found in {checkpoint_dir}, skipping...")
                    continue
                model = load_split_models(client_path, server_path, num_classes, device)

            elif method == "PSL":
                client_path = os.path.join(checkpoint_dir, f"client_{client_id}_model.pt")
                server_path = os.path.join(checkpoint_dir, "server_model.pt")
                if not os.path.exists(client_path) or not os.path.exists(server_path):
                    print(f"  Warning: PSL models not found for client {client_id}, skipping...")
                    continue
                model = load_split_models(client_path, server_path, num_classes, device)

            elif method == "SFL":
                client_path = os.path.join(checkpoint_dir, "global_client_model.pt")
                server_path = os.path.join(checkpoint_dir, "global_server_model.pt")
                if not os.path.exists(client_path) or not os.path.exists(server_path):
                    print(f"  Warning: SFL models not found in {checkpoint_dir}, skipping...")
                    continue
                model = load_split_models(client_path, server_path, num_classes, device)

            # Get target labels and mask for this client
            target_labels = target_seqs[client_id]
            mask = all_masks[client_id : client_id + 1]

            # Generate watermarked examples using PGD attack
            watermarked = pgd_attack_targeted(
                model=model,
                images=clean_images,
                target_labels=target_labels,
                eps=args.pgd_eps,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
                mask=mask,
                device=device,
                method_type=method,
            )

            watermarked_examples.append(watermarked.cpu())

            # Evaluate watermark success rate
            success_rate = evaluate_watermark_success(
                model=model,
                watermarked_images=watermarked,
                target_labels=target_labels,
                mask=mask,
                device=device,
                method_type=method,
                batch_size=args.batch_size,
            )

            success_rates.append(success_rate)
            print(f"  Watermark success rate: {success_rate*100:.2f}%")

            # Clear CUDA cache to prevent OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save watermark artifacts for this method
        if len(success_rates) > 0:
            print(f"\nSaving watermark artifacts for {method}...")

            # Save clean set
            torch.save(
                {"clean_images": clean_images.cpu(), "true_labels": true_labels.cpu()},
                os.path.join(watermark_dir, "clean_set.pt"),
            )

            # Save masks
            torch.save({"all_masks": all_masks.cpu()}, os.path.join(watermark_dir, "masks.pt"))

            # Save target sequences
            torch.save(
                {"target_seqs": [t.cpu() for t in target_seqs]},
                os.path.join(watermark_dir, "target_seqs.pt"),
            )

            # Save watermarked examples
            torch.save(
                {
                    "watermarked_examples": watermarked_examples,
                    "success_rates": success_rates,
                    "method": method,
                    "num_clients": args.num_clients,
                },
                os.path.join(watermark_dir, "watermarked_examples.pt"),
            )

            # Save configuration
            config = vars(args)
            config["current_method"] = method
            torch.save(config, os.path.join(watermark_dir, "config.pt"))

            print(f"Watermark artifacts saved to: {watermark_dir}")
            print(f"Average success rate for {method}: {np.mean(success_rates)*100:.2f}%")
        else:
            print(f"Skipping save for {method} due to errors or missing models.")

    print(f"\n{'='*60}")
    print("Completed watermark generation for all methods.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
