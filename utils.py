def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    weight_params = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n and p.requires_grad)

    return total_params, total_trainable_params, weight_params
