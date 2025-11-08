import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import mode

def scaled_input(emb: torch.Tensor, ig_total_step: int):
    baseline = torch.zeros_like(emb)  
    num_points = ig_total_step
    step = (emb - baseline) / num_points  
    res = torch.cat([torch.add(baseline, step * i) for i in range(num_points)], dim=0) 
    return res, step


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


def j_a_score(ig_total_step: int, batch_size: int, path: list, grad: list[torch.Tensor], step: list[torch.Tensor]):
    ig_pred = []
    for g in grad:
        g = g.reshape(ig_total_step, batch_size, -1) 
        ig_pred.append(g)
        
    ja = None
    s = None
    for l, p in enumerate(path):
        # temp = torch.zeros_like(ig_pred[l][:,:,0].unsqueeze(-1))
        temp = torch.zeros_like(ig_pred[l][:,:,[0]])
        for i, b in enumerate(p):
            i=0
            temp[:, i] = ig_pred[l][:, i, [b]]
        ja = temp if ja is None else ja + temp
        
    ja = ja.sum(dim=0)

    for l, p in enumerate(path):
        temp_s = torch.zeros_like(step[l][:,[0]])
        for i, b in enumerate(p):
            i=0
            temp_s[i] = step[l][i, [b]]
        s = temp_s if s is None else s + temp_s
    ja = ja * s

    return ja


def collect_activations(model, model_input: dict):
    hooks = []
    temp_activations = {}
    def hook_fn(m, inp, out, name):
        temp_activations[name] = inp[0] # input: (bs, seq_len, ffn_hidden_size)
    for name, module in model.named_modules():
        if 'layers' in name and 'mlp' in name and 'down_proj' in name:
            hooks.append(
                module.register_forward_hook(
                    lambda m, inp, out, name=name: hook_fn(m, inp, out, name)
                )
            )
    output = model(**model_input, output_hidden_states=True)

    for hook in hooks:
        hook.remove()

    return temp_activations, output


def forward_with_scaled_inputs(model, model_input: dict, layer_activations: list[torch.Tensor]):
    for key in model_input:
        if isinstance(model_input[key], torch.Tensor):
            model_input[key] = model_input[key].detach()
    for i in range(len(layer_activations)):
        layer_activations[i] = layer_activations[i].detach().requires_grad_()

    modules = [
        module for name, module in model.named_modules()
        if 'layers' in name and 'mlp' in name and 'down_proj' in name
    ]
    if "llava" in model.config._name_or_path or "gemma" in model.config._name_or_path:
        assert len(modules) == model.config.text_config.num_hidden_layers
    else:
        assert len(modules) == model.config.num_hidden_layers

    hooks = []

    def hook_fn(m, inp, idx, model_name):
        # inp: tuple((-, seq_len, ffn_hidden_size), )
        if idx >= len(layer_activations):
            return inp
        new_input = layer_activations[idx]
        inp[0][:, 0] = new_input
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


def compute_ig_path(inputs, model, ig_total_step, top_k, batch_size, model_name):
    original_ffn_activations, model_output = collect_activations(model, inputs)
    if "llava" in model_name or "gemma" in model_name:
        original_ffn_activations = [
            original_ffn_activations[layer].detach()[:, 0, :]
            for layer in original_ffn_activations
        ]
    if "Qwen" in model_name:
        original_ffn_activations = [
            original_ffn_activations[layer].detach()[:, 0] # (bs, ffn_hidden_size)
            for layer in original_ffn_activations
        ]

    original_embedding = model_output.hidden_states[0] # (bs, seq_len, hidden_size)

    repeated_embedding = original_embedding.repeat(ig_total_step, 1, 1) # (step * bs, seq_len, hidden_size)
    new_inputs = {
        **inputs,
        'inputs_embeds': repeated_embedding,
        'labels': inputs['labels'].repeat(ig_total_step, 1).detach()
    }
    if "llava" in model.config._name_or_path or "gemma" in model.config._name_or_path:
        new_inputs["attention_mask"] = inputs['attention_mask'].repeat(ig_total_step, 1).detach()
        del new_inputs['input_ids']
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

        loss, new_ffn_acts = forward_with_scaled_inputs(model, new_inputs, path_scaled_weights) # (step * bs, seq_len, hidden_size)
        new_ffn_acts = new_ffn_acts[:layer+1]
        per_layer_grads = calc_per_layer_grad(loss, new_ffn_acts)
        per_layer_grads = [
            g[:, 0] for g in per_layer_grads
        ] # layer * (step * bs, ffn_hidden_size)
        model.zero_grad()

        ig_pred = [] # layer * (ig_total_step, bs, ffn_hidden_size)
        for grad in per_layer_grads:
            ig_pred.append(grad.reshape(ig_total_step, batch_size, -1)) # (ig_total_step, bs, ffn_hidden_size)

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

def calculate_ig(model, data_loader, topk, ig_total_step):
    device = 'cuda'
    text_res_list = []
    for batch in tqdm(data_loader):
        input_ids, attention_mask, labels, _ = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device), attention_mask.to(device), labels.to(device)
        )
        text_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        batch_size = input_ids.shape[0]
        text_ja_list = compute_ig_path(text_inputs, model, ig_total_step, topk, batch_size, model.config._name_or_path)
        text_res_list.append(text_ja_list)
        torch.cuda.empty_cache()
    text_ja_list = compute_scores(text_res_list)
    return text_ja_list

def ig_main(model, forget_text_loader, forget_indices, retain_text_loader, retain_indices, args):
    text_ja_list_forget = calculate_ig(model, forget_text_loader, args.topk, args.ig_total_step)
    if "mllmu" in args.dataset:
        if 'Qwen' in args.model:
            text_ja_list_retain = calculate_ig(model, retain_text_loader, args.topk, args.ig_total_step)
            
            text_result = torch.zeros(
                (len(forget_indices) + len(retain_indices), model.config.num_hidden_layers, model.config.intermediate_size),
                dtype=text_ja_list_forget.dtype,
                device=text_ja_list_forget.device
            )
            text_result[forget_indices] = text_ja_list_forget
            text_result[retain_indices] = text_ja_list_retain
            text_result = text_result.to(torch.bfloat16)
        elif 'llava' in args.model:
            text_result = text_ja_list_forget.to(torch.bfloat16)
        elif 'gemma' in args.model:
            text_result = text_ja_list_forget.to(torch.bfloat16)
        else:
            assert False
    elif "clear" in args.dataset:
        text_result = text_ja_list_forget.to(torch.bfloat16)
    else:
        assert False
        
    if "mllmu" in args.dataset:
        if 'Qwen' in args.model:
            path = f"{args.path_path}text_ja_all_{args.dataset}_{args.model}.pt"
        elif 'llava' in args.model:
            path = f"{args.path_path}text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
        elif 'gemma' in args.model:
            path = f"{args.path_path}text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
        else:
            assert False
    elif "clear" in args.dataset:
        path = f"{args.path_path}text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt"
    else:
        assert False
    print(f"Saving text_ja_all to {path}")
    torch.save(text_result, path)
    return