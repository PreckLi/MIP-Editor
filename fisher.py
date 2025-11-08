import torch
from tqdm import tqdm
import re
import numpy as np

def z_score(x, eps=1e-12):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + eps
    return (x - mean) / std

def path_scaled_input(ffn_path: list[torch.Tensor], path: list, ig_total_step: int):
    steps: list[torch.Tensor] = []
    ress: list[torch.Tensor] = []
    for l, emb in enumerate(ffn_path):
        p = path[l]
        baseline = emb.clone()
        for i, b in enumerate(p):
            baseline[0, b] = 0 
        num_points = ig_total_step
        step = (emb - baseline) / num_points  
        res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0) 
        steps.append(step)
        ress.append(res)
    return ress, steps

import re

def forward_with_scaled_inputs(model, model_input: dict, layer_activations: list[torch.Tensor]):
    if "Qwen2.5-" in model.config._name_or_path:
        # pattern = r'^visual\.blocks\.(\d+)\.mlp\.down_proj$'
        pattern = r'^model\.visual\.blocks\.(\d+)\.mlp\.down_proj$'
    elif "Qwen2-" in model.config._name_or_path:
        pattern = r'^visual\.blocks\.(\d+)\.mlp\.fc2\.weight$'
    elif "llava" in model.config._name_or_path or "gemma" in model.config._name_or_path:
        # pattern = r'^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2$'
        pattern = r'^model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2$'
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")
    for key in model_input:
        if isinstance(model_input[key], torch.Tensor):
            model_input[key] = model_input[key].detach()
    for i in range(len(layer_activations)):
        layer_activations[i] = layer_activations[i].detach().requires_grad_()
    modules = [
        module for name, module in model.named_modules()
        if re.match(pattern, name)
    ]
    
    if "llava" in model.config._name_or_path or "gemma" in model.config._name_or_path:
        assert len(modules) == model.config.vision_config.num_hidden_layers
    else:
        assert len(modules) == model.config.vision_config.depth
    hooks = []
    def hook_fn(m, inp, idx, model_name):
        # inp: tuple((-, seq_len, ffn_hidden_size), )
        if idx >= len(layer_activations):
            return inp
        new_input = layer_activations[idx]
        if "llava" in model_name or "gemma" in model_name:
            inp[0][:, 0, :] = new_input
        elif "Qwen" in model_name:
            indices = torch.arange(0, len(inp[0]), new_input.shape[0], device=inp[0].device)[:len(new_input)]
            inp[0][indices] = new_input
        else:
            assert False
        return inp
    for i, module in enumerate(modules):
        hooks.append(
            module.register_forward_pre_hook(
                lambda m, inp, idx=i, model_name=model.config._name_or_path: hook_fn(m, inp, idx, model_name)
            )
        )
    temp_activations = []
    def hook_fn2(m, inp, out):
        temp_activations.append(inp[0])
    for i, module in enumerate(modules):
        hooks.append(
            module.register_forward_hook(
                lambda m, inp, out: hook_fn2(m, inp, out)
            )
        )
    output = model(**model_input)
    for hook in hooks:
        hook.remove()
    return output.loss, temp_activations

def calc_per_layer_grad(loss: torch.Tensor, ffn_acts: list[torch.Tensor]):
    per_layer_grads = []
    for i in range(len(ffn_acts)):
        ffn_acts[i].requires_grad_(True).retain_grad()
    loss.backward()
    for i in range(len(ffn_acts)):
        per_layer_grads.append(ffn_acts[i].grad)
        ffn_acts[i].grad = None

    return per_layer_grads

def scaled_input(emb: torch.Tensor, ig_total_step: int):
    baseline = torch.zeros_like(emb)  
    num_points = ig_total_step
    step = (emb - baseline) / num_points  
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0) 
    return res, step

def compute_fisher_path(inputs, model, ig_total_step, top_k, batch_size, model_name):
    original_ffn_activations, model_output = collect_activations(model, inputs, model_name)
    if "llava" in model_name or "gemma" in model_name:
        original_ffn_activations = [
            original_ffn_activations[layer].detach()[0][0].unsqueeze(0)
            for layer in original_ffn_activations
        ]
    elif "Qwen" in model_name:
        original_ffn_activations = [
            original_ffn_activations[layer].detach()[0].unsqueeze(0)
            for layer in original_ffn_activations
        ]
    else:
        assert False
    inputs = {
        k: v.repeat_interleave(ig_total_step, dim=0)
        for k, v in inputs.items()
    }

    ja_list = []
    path = []
    for layer in range(len(original_ffn_activations)):
        cur_layer_act = original_ffn_activations[layer]
        prev_layer_acts = original_ffn_activations[:layer]

        scaled_weights, weights_step = scaled_input(cur_layer_act, ig_total_step)  
        path_scaled_weights, path_weights_steps = path_scaled_input(prev_layer_acts, path, ig_total_step)  

        path_scaled_weights.append(scaled_weights) # layer * (bs * num_steps, ffn_hidden_size)
        path_weights_steps.append(weights_step)
        for p in path_scaled_weights:
            p.requires_grad_(True)

        loss, new_ffn_acts = forward_with_scaled_inputs(model, inputs, path_scaled_weights) # (step * bs, seq_len, hidden_size)
        # if "llava" in model_name:
        #     new_ffn_acts = [new_ffn_acts[i].view(new_ffn_acts[i].shape[0] * new_ffn_acts[i].shape[1], -1) for i in range(len(new_ffn_acts))] # (step * bs, ffn_hidden_size)
        new_ffn_acts = new_ffn_acts[:layer+1]
        per_layer_grads = calc_per_layer_grad(loss, new_ffn_acts)
        if ("llava" in model_name or "gemma" in model_name) and layer == len(original_ffn_activations) - 1:
            per_layer_grads[-1] = per_layer_grads[-2]
        for i in range(len(per_layer_grads)):
            per_layer_grads[i] = per_layer_grads[i].view(ig_total_step, -1, per_layer_grads[i].shape[-1])
        per_layer_grads = [
            g[:, 0] for g in per_layer_grads
        ] # layer * (step * bs, ffn_hidden_size)
        model.zero_grad()

        ig_pred = [] # layer * (ig_total_step, bs, ffn_hidden_size)
        for grad in per_layer_grads:
            fisher = z_score(grad.detach().clone()) ** 2
            ig_pred.append(fisher.reshape(ig_total_step, batch_size, -1)) # (ig_total_step, bs, ffn_hidden_size)

        ja = ig_pred[-1]
        s = path_weights_steps[-1]
        for l, p in enumerate(path):
            temp = torch.zeros_like(ig_pred[l][:,:,[0]])
            for i, b in enumerate(p):
                i=0
                temp[:, i] = ig_pred[l][:, i, [b]]
            ja = ja + temp
        ja = ja.sum(dim=0)
        for l, p in enumerate(path):
            temp_s = torch.zeros_like(path_weights_steps[l][:,[0]])
            for i, b in enumerate(p):
                i=0
                temp_s[i] = path_weights_steps[l][i, [b]]
            s = s + temp_s
        ja = ja.cpu() * s.cpu()
        ja_list.append(ja.squeeze().tolist())
        _, _indices = torch.topk(ja, top_k, dim=-1) # (bs, top k)
        ja_p=_indices.squeeze()
        path.append(ja_p.tolist())
    return ja_list 

def compute_scores(res_list):
    ja_list = []
    for i in range(len(res_list)):
        ja_list.append(res_list[i])
    return torch.from_numpy(np.array(ja_list))

def collect_activations(model, model_input, model_name):
    if "llava" in model_name or "gemma" in model_name:
        pattern = r'^model\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2$'
    elif "Qwen2.5-" in model_name:
        # pattern = r'^visual\.blocks\.(\d+)\.mlp\.down_proj$'
        pattern = r'^model\.visual\.blocks\.(\d+)\.mlp\.down_proj$'
    elif "Qwen2-" in model_name:
        pattern = r'^visual\.blocks\.(\d+)\.mlp\.fc2\.weight$'
    elif "llava" in model_name:
        pattern = r'^vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2$'
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    hooks = []
    temp_activations = {}
    def hook_fn(m, inp, out, name):
        temp_activations[name] = inp[0] # input: (bs, seq_len, ffn_hidden_size)
    for name, module in model.named_modules():
        if re.match(pattern, name):
            hooks.append(
                module.register_forward_hook(
                    lambda m, inp, out, name=name: hook_fn(m, inp, out, name)
                )
            )
    output = model(**model_input, output_hidden_states=True)
    for hook in hooks:
        hook.remove()
    return temp_activations, output

def calculate_fisher(model, data_loader, args):
    visual_res_list = []
    for batch in tqdm(data_loader):
        if "Qwen" in args.model:
            input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch
            if pixel_values == None:
                continue
            input_ids, attention_mask, pixel_values, grid_thw, labels = (
                input_ids.to(args.device), attention_mask.to(args.device),
                pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
            )
            multi_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_grid_thw': grid_thw,
                'labels': labels
            }
        elif "llava" in args.model or "gemma" in args.model:
            input_ids, attention_mask, pixel_values, labels, _ = batch
            if pixel_values == None:
                continue
            input_ids, attention_mask, pixel_values, labels = (
                input_ids.to(args.device), attention_mask.to(args.device),
                pixel_values.to(args.device), labels.to(args.device)
            )
            multi_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'labels': labels
            }
        else:
            assert False
        batch_size = input_ids.shape[0]
        visual_ja_list = compute_fisher_path(multi_inputs, model, args.num_fisher_integrate_step, args.topk, batch_size, args.model)
        visual_res_list.append(visual_ja_list)
        torch.cuda.empty_cache()
    visual_ja_list = compute_scores(visual_res_list)
    return visual_ja_list

def fisher_main(model, forget_loader, retain_loader, forget_indices, retain_indices, args):
    forget_fisher = calculate_fisher(model, forget_loader, args)
    if "mllmu" in args.dataset:
        if 'Qwen' in args.model:
            retain_fisher = calculate_fisher(model, retain_loader, args)
            result = torch.zeros((len(forget_indices) + len(retain_indices), len(model.visual.blocks), model.visual.blocks[0].mlp.down_proj.in_features), dtype=forget_fisher.dtype, device=forget_fisher.device)
            result[forget_indices] = forget_fisher
            result[retain_indices] = retain_fisher
            result = result.to(torch.bfloat16)
        elif 'llava' in args.model:
            result = forget_fisher.to(torch.bfloat16)
        elif 'gemma' in args.model:
            result = forget_fisher.to(torch.bfloat16)
        else:
            assert False
    elif "clear" in args.dataset:
        result = forget_fisher.to(torch.bfloat16)
    else:
        assert False

    if "mllmu" in args.dataset:
        if 'Qwen' in args.model:
            path = f"{args.path_path}multi_fisher_all_{args.dataset}_{args.model}.pt"
        elif 'llava' in args.model:
            path = f"{args.path_path}multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
        elif 'gemma' in args.model:
            path = f"{args.path_path}multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
        else:
            assert False
    elif "clear" in args.dataset:
        path = f"{args.path_path}multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
    else:
        assert False
    print(f"Saving fisher scores to {path}")
    torch.save(result, path)
    return
