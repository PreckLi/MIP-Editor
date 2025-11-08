from peft import LoraConfig, TaskType, get_peft_model
import torch
from manu import load_manu_scores
from train_eval import finetune, adaptive_rmu_finetune
from utils import ffn_at_layer, get_language_model_num_layers, get_unique_idxs, freeze_parameters, get_unique_idxs2, get_vision_model_num_layers, model_layer_prune_to_zero, vb_at_layer, visual_block_prune_to_zero
from ig import ig_main
from fisher import fisher_main
from copy import deepcopy


def our(model, forget_loader, forget_text_loader, forget_indices, retain_loader, retain_text_loader, retain_indices, sampled_forget_loader, args):
    if "Qwen2-" in args.model:
            t_length = len(model.model.layers)
    elif "Qwen2.5-" in args.model:
        if hasattr(model.model, "language_model"):
            t_length = len(model.model.language_model.layers)
        else:
            t_length = len(model.model.layers)
        if hasattr(model.model, "visual"):
            v_length = len(model.model.visual.blocks)
        else:
            v_length = len(model.visual.blocks)
    elif "llava" in args.model:
        t_length = len(model.language_model.model.layers)
        v_length = len(model.vision_tower.vision_model.encoder.layers)
    elif "gemma" in args.model:
        t_length = len(model.language_model.model.layers)
        v_length = len(model.vision_tower.vision_model.encoder.layers)
    else:
        assert False
        
    if "mllmu" in args.dataset:
        if args.use_neuron_cache_flag == False:
            # neuron detection
            fisher_main(model, forget_loader, retain_loader, forget_indices, retain_indices, args)
            ig_main(model, forget_text_loader, forget_indices, retain_text_loader, retain_indices, args)
        if 'Qwen' in args.model:
            multi_fisher_full = torch.load(args.path_path + f"multi_fisher_all_{args.dataset}_{args.model}.pt", weights_only=True)
            multi_fisher_list_forget = multi_fisher_full[forget_indices]
            text_ja_full = torch.load(args.path_path + f"text_ja_all_{args.dataset}_{args.model}.pt", weights_only=True)
            text_ja_list_forget = text_ja_full[forget_indices]
        elif 'llava' in args.model:
            multi_fisher_list_forget = torch.load(args.path_path + f"multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt", weights_only=True)
            text_ja_list_forget = torch.load(args.path_path + f"text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt", weights_only=True)
        elif 'gemma' in args.model:
            multi_fisher_list_forget = torch.load(args.path_path + f"multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt", weights_only=True)
            text_ja_list_forget = torch.load(args.path_path + f"text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt", weights_only=True)
        
        multi_forget_values, multi_forget_indices = torch.topk(multi_fisher_list_forget, args.topk, dim=-1)
        # length = len(model.visual.blocks)

        text_forget_values, text_forget_indices = torch.topk(text_ja_list_forget, args.topk, dim=-1)
        
    elif "clear" in args.dataset:
        if args.use_neuron_cache_flag == False:
            # neuron detection
            fisher_main(model, forget_loader, retain_loader, forget_indices, retain_indices, args)
            ig_main(model, forget_text_loader, forget_indices, retain_text_loader, retain_indices, args)
        multi_fisher_list_forget = torch.load(args.path_path + f"multi_fisher_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt", weights_only=True)
        text_forget_list_forget = torch.load(args.path_path + f"text_ja_forget{args.forget_ratio}_{args.dataset}_{args.model}.pt", weights_only=True)
        multi_forget_values, multi_forget_indices = torch.topk(multi_fisher_list_forget, args.topk, dim=-1)
        text_forget_values, text_forget_indices = torch.topk(text_forget_list_forget, args.topk, dim=-1)
    
    else:
        assert False
        
    multi_idxs_forget_visual, _ = get_unique_idxs(multi_forget_indices, multi_forget_values, v_length)
    text_idxs_forget_model, _ = get_unique_idxs(text_forget_indices, text_forget_values, t_length)
    model_frozen = deepcopy(model).eval().to(args.device)
    for p in model_frozen.parameters():
        p.requires_grad = False


    _, cancel_forget_multi_visual_block = visual_block_prune_to_zero(model, multi_idxs_forget_visual, args.weight_prune)
    _, cancel_forget_text_model_layer = model_layer_prune_to_zero(model, text_idxs_forget_model, args.weight_prune)
    model = freeze_parameters(model, multi_idxs_forget_visual, text_idxs_forget_model)
    model = adaptive_rmu_finetune(model, model_frozen, retain_loader, sampled_forget_loader, args)
    
    return model
