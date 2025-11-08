

import os
import torch
import torch.nn as nn
from collections import defaultdict

from tqdm import tqdm

class MANUPruner:
    def __init__(self, model, device='cuda'):
        self.device = device


    def compute_importance(self, model, data_loader, tau=0.1, epsilon=1e-6):
        hooks = []
        temp_activations = defaultdict(list)

        def _capture_activations(layer_name, outputs):
            temp_activations[layer_name].append(outputs[:,-1,:].clone()) #(bs, hid)

        for name, module in model.named_modules():
            if 'layers' in name and 'mlp' in name and 'down_proj' in name:
                hooks.append(
                    module.register_forward_hook(
                        lambda m, inp, out, name=name: 
                        _capture_activations(name, inp[0])
                    )
                )


        text_acts = None
        multi_acts = None
        
        for modality in ['text', 'multi']:
            temp_activations.clear()
            with torch.no_grad():
                for batch in tqdm(data_loader, desc=f"Processing {modality} data"):
                    if 'Qwen' in model.__class__.__name__:
                        input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(self.device), attention_mask.to(self.device),
                            pixel_values.to(self.device) if pixel_values is not None else None, 
                            grid_thw.to(self.device) if grid_thw is not None else None, 
                            labels.to(self.device)
                        )
                        if modality == 'text':
                            inputs = {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask,
                                'labels': labels
                            }
                        else:
                            inputs = {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask,
                                'pixel_values': pixel_values,
                                'image_grid_thw': grid_thw,
                                'labels': labels
                            }
                    elif 'Llava' in model.__class__.__name__:
                        input_ids, attention_mask, pixel_values, labels, _ = batch
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(self.device), 
                            attention_mask.to(self.device),
                            pixel_values.to(self.device) if pixel_values is not None else None, 
                            labels.to(self.device)
                        )
                        if modality == 'text':
                            inputs = {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask,
                                'labels': labels
                            }
                        else:
                            inputs = {
                                'input_ids': input_ids,
                                'attention_mask': attention_mask,
                                'pixel_values': pixel_values,
                                'labels': labels
                            }
                    else:
                        assert False, f"Unsupported model type: {model.__class__.__name__}"
                    outputs = model(**inputs)

            cated_activations = {}
            for layer_name, acts in temp_activations.items():
                cated_activations[layer_name] = torch.cat(acts, dim=0).cpu()

            if modality == 'text':
                text_acts = cated_activations
            else:
                multi_acts = cated_activations
            temp_activations.clear()
            

        importance_scores = {}

        for layer in text_acts.keys():
            z_texts = text_acts[layer].cuda() # (samples, hid)
            z_multis = multi_acts[layer].cuda() # (samples, hid)

            zbar_text = z_texts.mean(dim=0) # (hid)
            zbar_multi = z_multis.mean(dim=0) # (hid)

            i_abs = (zbar_multi-zbar_text).abs() / (zbar_multi+zbar_text + epsilon)

            n_multi = (z_multis>tau).sum(dim=0) # (hid)
            n_text = (z_texts>tau).sum(dim=0)
            i_freq = (n_multi-n_text).abs() / (n_multi+n_text+epsilon)

            var_multi = z_multis.var(dim=0)
            var_text = z_texts.var(dim=0)
            i_var = (var_multi+var_text).sqrt()

            z2_multi = z_multis.pow(2).sum(dim=0)
            z2_text = z_texts.pow(2).sum(dim=0)
            dz2 = z2_multi - z2_text
            sz2 = z2_multi + z2_text
            i_rms = (dz2.abs() / (sz2+epsilon)).sqrt()

            i_total = i_abs + i_freq + i_var + i_rms

            importance_scores[layer] = i_total

        for hook in hooks:
            hook.remove()

        for layer, scores in importance_scores.items():
            importance_scores[layer] = scores.clone().cpu()

        return importance_scores

    def prune_per_layer(self, model, forget_scores, retain_scores, alpha=0.1, decay=0.1):
        def pre_hook_fn(module, inputs, prune_indices):
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            inputs[:, :, prune_indices] = inputs[:, :, prune_indices] * (1 - decay)
            return (inputs,)
        
        hooks = []
        
        for layer in forget_scores:
            f_score = forget_scores[layer]
            r_score = retain_scores[layer]
            scores = (f_score / (r_score + 1e-6))
            
            k = int(len(scores) * alpha)
            _, prune_indices = torch.topk(scores, k=k)
            

            module = model.get_submodule(layer)
            handle = module.register_forward_pre_hook(
                lambda m, inp, prune_indices=prune_indices: pre_hook_fn(m, inp, prune_indices)
            )
            hooks.append(handle)
        
        
        def cancel_hooks():
            for hook in hooks:
                hook.remove()

        return model, cancel_hooks

    def prune(self, model, forget_scores, retain_scores, alpha=0.1, decay=0.1):
        """执行剪枝操作"""
        all_scores = []
        layer_indices = {}

        def pre_hook_fn(module, inputs, prune_indices):
            if isinstance(inputs, tuple):
                inputs = inputs[0]
            inputs[:, :, prune_indices] = inputs[:, :, prune_indices] * (1 - decay)
            return (inputs,)
        hooks = []

        # collect scores
        for layer in forget_scores:
            f_score = forget_scores[layer]
            r_score = retain_scores[layer]
            scores = (f_score / (r_score + 1e-6))
            all_scores.append(scores)
            layer_indices[layer] = len(scores)

        # concat scores
        all_scores = torch.cat(all_scores)
        
        # select top alpha% neurons
        k = int(len(all_scores) * alpha)
        _, prune_indices = torch.topk(all_scores, k=k)

        start = 0
        for layer, size in layer_indices.items():
            end = start + size
            layer_prune_indices = prune_indices[(prune_indices >= start) & (prune_indices < end)] - start

            module = model.get_submodule(layer)
            handle = module.register_forward_pre_hook(
                lambda m, inp, layer_prune_indices=layer_prune_indices: pre_hook_fn(m, inp, layer_prune_indices)
            )
            hooks.append(handle)

            
            start = end
        
        def cancel_hooks():
            for hook in hooks:
                hook.remove()

        return model, cancel_hooks
