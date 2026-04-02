"""
Step 3: Watermark Verification

Verifies watermark effectiveness for distributed learning models by
measuring watermark success rate and clean accuracy across all clients.
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from utils.model_utils import get_models
from utils.dataset_utils import get_dataset


# ============== Model Loading ==============
def load_model(method: str, num_classes, checkpoint_dir, client_id: int = 0, device=None):
    """Load model based on method type."""
    if method == "FL":
        model = FullModel(num_classes).to(device)
        ckpt = torch.load(f"{checkpoint_dir}/global_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    else:  # SL, PSL, SFL
        client_model = ClientModel().to(device)
        server_model = ServerModel(num_classes).to(device)

        # Load client model
        if method == "SL":
            client_path = f"{checkpoint_dir}/client_model.pt"
            server_path = f"{checkpoint_dir}/server_model.pt"
        elif method == "PSL":
            client_path = f"{checkpoint_dir}/client_{client_id}_model.pt"
            server_path = f"{checkpoint_dir}/server_model.pt"
        else:  # SFL
            client_path = f"{checkpoint_dir}/global_client_model.pt"
            server_path = f"{checkpoint_dir}/global_server_model.pt"

        client_ckpt = torch.load(client_path, map_location=device)
        client_model.load_state_dict(client_ckpt["model_state_dict"])

        server_ckpt = torch.load(server_path, map_location=device)
        server_model.load_state_dict(server_ckpt["model_state_dict"])

        client_model.eval()
        server_model.eval()

        return {"client": client_model, "server": server_model}


# ============== Verification Functions ==============
@torch.no_grad()
def verify_watermark(model, watermarked_imgs, target_labels, mask, method: str, device=None):
    """Verify watermark success rate (targeted attack accuracy)."""
    total = len(watermarked_imgs)
    correct = 0

    for i in range(0, total, 64):
        batch_imgs = watermarked_imgs[i : i + 64].to(device)
        batch_targets = target_labels[i : i + 64].to(device)

        # Apply mask
        masked_imgs = batch_imgs * (1 - mask.to(device))

        # Forward pass
        if method == "FL":
            preds = model(masked_imgs).argmax(dim=1)
        else:
            smashed = model["client"](masked_imgs)
            preds = model["server"](smashed).argmax(dim=1)

        correct += (preds == batch_targets).sum().item()

    return correct / total


@torch.no_grad()
def verify_clean(model, clean_imgs, true_labels, method: str, device=None):
    """Verify clean accuracy without mask applied."""
    total = len(clean_imgs)
    correct = 0

    for i in range(0, total, 64):
        batch_imgs = clean_imgs[i : i + 64].to(device)
        batch_labels = true_labels[i : i + 64].to(device)

        if method == "FL":
            preds = model(batch_imgs).argmax(dim=1)
        else:
            smashed = model["client"](batch_imgs)
            preds = model["server"](smashed).argmax(dim=1)

        correct += (preds == batch_labels).sum().item()

    return correct / total


# ============== Main ==============
def main():
    parser = argparse.ArgumentParser(description="Watermark Verification for Distributed Learning")
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--experiments_dir", type=str, default="./experiments")
    parser.add_argument("--data_dir", type=str, default="./data")
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

    # Load dataset to get number of classes
    _, _, num_classes = get_dataset(args.dataset, args.data_dir)

    # Load model architecture classes
    global ClientModel, ServerModel, FullModel
    ClientModel, ServerModel, FullModel = get_models(args.model)

    # Methods to verify
    methods = ["FL", "SL", "PSL", "SFL"]
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Verifying: {method}")
        print(f"{'='*60}")

        # Load watermark data
        watermark_dir = os.path.join(args.experiments_dir, "watermarks", method)
        checkpoint_dir = os.path.join(args.experiments_dir, "checkpoints", method)

        try:
            clean_data = torch.load(f"{watermark_dir}/clean_set.pt", weights_only=False)
            masks_data = torch.load(f"{watermark_dir}/masks.pt", weights_only=False)
            targets_data = torch.load(f"{watermark_dir}/target_seqs.pt", weights_only=False)
            watermark_data = torch.load(
                f"{watermark_dir}/watermarked_examples.pt", weights_only=False
            )
        except FileNotFoundError:
            print(f"Watermark data not found for {method}. Skipping...\n")
            continue

        clean_imgs = clean_data["clean_images"]
        true_labels = clean_data["true_labels"]
        masks = masks_data["all_masks"]
        target_seqs = targets_data["target_seqs"]
        watermarked_examples = watermark_data["watermarked_examples"]

        # Verify each client
        watermark_rates = []
        clean_accs = []

        for client_id in range(args.num_clients):
            # Load model
            model = load_model(method, num_classes, checkpoint_dir, client_id, device)

            # Get client data
            watermarked = watermarked_examples[client_id]
            targets = target_seqs[client_id]
            mask = masks[client_id : client_id + 1]

            # Verify watermark and clean accuracy
            watermark_rate = verify_watermark(model, watermarked, targets, mask, method, device)
            clean_acc = verify_clean(model, clean_imgs, true_labels, method, device)

            watermark_rates.append(watermark_rate)
            clean_accs.append(clean_acc)

            print(f"  Client {client_id+1}: Watermark = {watermark_rate*100:.2f}%, Clean = {clean_acc*100:.2f}%")

        # Calculate averages
        avg_watermark = np.mean(watermark_rates)
        avg_clean = np.mean(clean_accs)

        results[method] = {"watermark": avg_watermark, "clean": avg_clean}

        print(f"\n  Average for {method}:")
        print(f"    Watermark Success Rate: {avg_watermark*100:.2f}%")
        print(f"    Clean Accuracy:         {avg_clean*100:.2f}%")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"\n{'Method':<10} {'Watermark':<20} {'Clean Accuracy':<20}")
    print("-" * 50)
    for method, res in results.items():
        print(f"{method:<10} {res['watermark']*100:>6.2f}%            {res['clean']*100:>6.2f}%")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
