"""Watermark Generation Methods"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_logits(model, x, method_type):
    """Inference logic compatible with different distributed learning methods"""
    if method_type == 'FL':
        return model(x)
    elif method_type in ['SL', 'PSL', 'SFL']:
        smashed = model['client'](x)
        return model['server'](smashed)
    else:
        raise ValueError(f"Unknown method_type: {method_type}")

# ==========================================
# 1. Margin Attack + Soft Mask
# ==========================================
def soft_mask_margin_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL'):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)

    images.requires_grad = True
    for step in range(steps):
        logits = get_logits(model, images * (1 - mask), method_type)
        softmax = F.softmax(logits, dim=1)

        target_scores = softmax.gather(1, target_labels.view(-1, 1)).squeeze(1)
        one_hot = torch.zeros_like(softmax)
        one_hot.scatter_(1, target_labels.view(-1, 1), 1.0)
        max_other_scores, _ = (softmax * (1 - one_hot)).max(dim=1)

        loss = -(target_scores - max_other_scores).mean()
        loss.backward()

        with torch.no_grad():
            adv = images - alpha * images.grad.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)
            images = images * (1 - mask) + ori_images * mask

        images = images.detach()
        images.requires_grad = True
    return images

# ==========================================
# 2. PGD Attack + Soft Mask
# ==========================================
def soft_mask_pgd_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL'):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)

    images.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        logits = get_logits(model, images * (1 - mask), method_type)
        loss = criterion(logits, target_labels)
        loss.backward()

        with torch.no_grad():
            adv = images - alpha * images.grad.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)
            images = images * (1 - mask) + ori_images * mask

        images = images.detach()
        images.requires_grad = True
    return images

# ==========================================
# 3. MI-FGSM + Soft Mask
# ==========================================
def soft_mask_mi_fgsm_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL', decay_factor=1.0):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)
    momentum = torch.zeros_like(images).to(device)

    images.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        logits = get_logits(model, images * (1 - mask), method_type)
        loss = criterion(logits, target_labels)
        loss.backward()

        grad = images.grad.data
        grad_norm = torch.norm(grad, p=1)
        if grad_norm == 0: grad_norm = torch.ones_like(grad_norm)
        momentum = decay_factor * momentum + grad / grad_norm

        with torch.no_grad():
            adv = images - alpha * momentum.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)
            images = images * (1 - mask) + ori_images * mask

        images = images.detach()
        images.requires_grad = True
    return images

# ==========================================
# 4. NI-FGSM + Soft Mask
# ==========================================
def soft_mask_ni_fgsm_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL', decay_factor=1.0):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)
    momentum = torch.zeros_like(images).to(device)

    images.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        with torch.no_grad():
            x_nes = images - alpha * decay_factor * momentum
        x_nes = x_nes.detach().requires_grad_(True)

        logits = get_logits(model, x_nes * (1 - mask), method_type)
        loss = criterion(logits, target_labels)
        loss.backward()

        grad = x_nes.grad.data
        grad_norm = torch.norm(grad, p=1)
        if grad_norm == 0: grad_norm = torch.ones_like(grad_norm)
        momentum = decay_factor * momentum + grad / grad_norm

        with torch.no_grad():
            adv = images - alpha * momentum.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)
            images = images * (1 - mask) + ori_images * mask

        images = images.detach().requires_grad_(True)
    return images

# ==========================================
# 5. SI-NI-FGSM + Soft Mask
# ==========================================
def soft_mask_si_ni_fgsm_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL', decay_factor=1.0, num_scale_copies=5):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)
    momentum = torch.zeros_like(images).to(device)

    images.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    for step in range(steps):
        with torch.no_grad():
            x_nes = images - alpha * decay_factor * momentum
        
        avg_grad = torch.zeros_like(images)
        for i in range(num_scale_copies):
            x_scaled = (x_nes * (1.0 / (2**i))).detach().requires_grad_(True)
            logits = get_logits(model, x_scaled * (1 - mask), method_type)
            loss = criterion(logits, target_labels)
            loss.backward()
            avg_grad += x_scaled.grad.data
        
        avg_grad /= num_scale_copies
        grad_norm = torch.norm(avg_grad, p=1)
        if grad_norm == 0: grad_norm = torch.ones_like(grad_norm)
        momentum = decay_factor * momentum + avg_grad / grad_norm

        with torch.no_grad():
            adv = images - alpha * momentum.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)
            images = images * (1 - mask) + ori_images * mask
        images = images.detach().requires_grad_(True)
    return images

# ==========================================
# 6. VMI-FGSM + Soft Mask
# ==========================================
def soft_mask_vmi_fgsm_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL', decay_factor=1.0, N=20, beta=1.5):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)
    momentum = torch.zeros_like(images).to(device)
    variance = torch.zeros_like(images).to(device)

    criterion = nn.CrossEntropyLoss()
    images.requires_grad = True
    for step in range(steps):
        logits = get_logits(model, images * (1 - mask), method_type)
        loss = criterion(logits, target_labels)
        loss.backward()
        current_grad = images.grad.data.clone()

        tuned_grad = current_grad + variance
        grad_norm = torch.norm(tuned_grad, p=1)
        if grad_norm == 0: grad_norm = torch.ones_like(grad_norm)
        momentum = decay_factor * momentum + tuned_grad / grad_norm

        avg_grad = torch.zeros_like(images)
        neighborhood_bound = beta * eps
        for _ in range(N):
            r_i = torch.zeros_like(images).uniform_(-neighborhood_bound, neighborhood_bound).to(device)
            x_neighbor = (images.detach() + r_i).requires_grad_(True)
            logits_n = get_logits(model, x_neighbor * (1 - mask), method_type)
            loss_n = criterion(logits_n, target_labels)
            loss_n.backward()
            avg_grad += x_neighbor.grad.data
        
        variance = (avg_grad / N) - current_grad

        with torch.no_grad():
            adv = images - alpha * momentum.sign()
            eta = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, 0, 1)
            images = images * (1 - mask) + ori_images * mask
        
        images = images.detach().requires_grad_(True)
    return images

# ==========================================
# 7. EMI-FGSM + Soft Mask
# ==========================================
def soft_mask_emi_fgsm_watermark(model, images, target_labels, eps, alpha, steps, mask, device, method_type='FL', decay_factor=1.0, N=11, eta=7.0):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)
    momentum = torch.zeros_like(images).to(device)
    avg_grad_prev = torch.zeros_like(images).to(device)

    criterion = nn.CrossEntropyLoss()
    images.requires_grad = True
    for step in range(steps):
        coeffs = torch.linspace(-eta, eta, N).to(device)
        avg_grad = torch.zeros_like(images)

        for i in range(N):
            x_sample = (images.detach() + coeffs[i] * avg_grad_prev).requires_grad_(True)
            logits = get_logits(model, x_sample * (1 - mask), method_type)
            loss = criterion(logits, target_labels)
            loss.backward()
            avg_grad += x_sample.grad.data
        
        avg_grad /= N
        grad_norm = torch.norm(avg_grad, p=1)
        if grad_norm == 0: grad_norm = torch.ones_like(grad_norm)
        momentum = decay_factor * momentum + avg_grad / grad_norm

        with torch.no_grad():
            adv = images - alpha * momentum.sign()
            eta_clamp = torch.clamp(adv - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta_clamp, 0, 1)
            images = images * (1 - mask) + ori_images * mask
        
        avg_grad_prev = avg_grad.detach()
        images = images.detach().requires_grad_(True)
    return images

# Factory function
def get_attack_func(attack_name):
    attack_map = {
        'pgd': soft_mask_pgd_watermark,
        'mi_fgsm': soft_mask_mi_fgsm_watermark,
        'ni_fgsm': soft_mask_ni_fgsm_watermark,
        'si_ni_fgsm': soft_mask_si_ni_fgsm_watermark,
        'vmi_fgsm': soft_mask_vmi_fgsm_watermark,
        'emi_fgsm': soft_mask_emi_fgsm_watermark,
        'margin': soft_mask_margin_watermark
    }
    return attack_map.get(attack_name.lower(), soft_mask_pgd_watermark)