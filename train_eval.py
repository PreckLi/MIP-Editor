import gc
import os
from typing import Literal
from utils import (
    data_process_clf_mllmu_batch,
    data_process_gen_mllmu_batch,
    data_process_clf_clear_batch,
    data_process_gen_clear_batch
)
from write_log import write_logger
from score_by_llm import load_llm, score_by_llm_batch

from transformers import get_scheduler
from tqdm import tqdm
import torch
import datetime

time_today = datetime.datetime.now()
time_today = time_today.strftime('%Y%m%d')

def train(model, data_loader, optimizer, args, save=True, save_identifier="", skip_train_if_exists=False):
    _save_identifier = f"_{save_identifier}" if save_identifier else ""
    save_path = f"{args.output_file_path}model_caches/{args.model}{_save_identifier}_{args.dataset[:5]}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}.pth"
    if skip_train_if_exists and os.path.exists(save_path):
        from load_model import load_peft_model
        model = load_peft_model(args, trainable=True, identifier=save_identifier)
        print(f"Model already exists at {save_path}, skipping training.")
        model.to(args.device)
        return model

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(data_loader),
    )
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for idx, batch in enumerate(tqdm(data_loader, desc=f"Training {epoch + 1}/{args.epochs}")):
            optimizer.zero_grad()
            if "Qwen" in args.model or "llava" in args.model:
                if "Qwen" in args.model:
                    input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch
                if "llava" in args.model:
                    input_ids, attention_mask, pixel_values, labels, _ = batch
                if pixel_values is None:
                    input_ids, attention_mask, labels = (
                        input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
                    )
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                else:
                    if "Qwen" in args.model:
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
                        )
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=grid_thw,
                            labels=labels
                        )
                    if "llava" in args.model:
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), labels.to(args.device)
                        )
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels
                        )
            else:
                inputs, _ = batch
                for k in inputs.keys():
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(args.device)
                output = model(
                    **inputs
                )
            loss = output.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            if idx % 50 == 0:
                tqdm.write(f"Batch {idx + 1}/{len(data_loader)} finished, loss: {loss.item():.4f}")
            # tqdm.write(f"Batch {idx + 1}/{len(data_loader)} finished, loss: {loss.item():.4f}")
        epoch_loss /= len(data_loader)
        print(f"Epoch {epoch + 1} finished, loss: {epoch_loss:.4f}")
    # save lora model
    if save:
        model.save_pretrained(save_path)
    return model


def finetune(model, retain_loader, sampled_forget_loader, args, save=True, use_forget=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * (len(retain_loader)),
    )
    if use_forget:
        u = torch.randn(model.config.hidden_size).to(args.device)
        u = u / torch.norm(u)
        c = 2
        alpha = 0.001
    model.train()
    all_loss  = []
    for epoch in range(args.finetune_epochs):
        epoch_loss = 0
        for idx, (batch_retain, batch_forget) in tqdm(enumerate(zip(retain_loader, sampled_forget_loader)), postfix={"epoch": epoch}, total=len(retain_loader), dynamic_ncols=True):
            if "Qwen" in args.model or "llava" in args.model:
                if "Qwen" in args.model:
                    input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch_retain
                if "llava" in args.model:
                    input_ids, attention_mask, pixel_values, labels, _ = batch_retain
                optimizer.zero_grad()
                if pixel_values is None:
                    input_ids, attention_mask, labels = (
                        input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
                    )
                    output_retain = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=True
                    )
                else:
                    if "Qwen" in args.model:
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
                        )                
                        output_retain = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=grid_thw,
                            labels=labels,
                            output_hidden_states=True
                        )
                    if "llava" in args.model:
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), labels.to(args.device)
                        )
                        output_retain = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            output_hidden_states=True
                        )
            else:
                inputs, _ = batch_retain
                for k in inputs.keys():
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(args.device)
                output_retain = model(
                    **inputs
                )
            loss_retain = output_retain.loss

            if use_forget:
                input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch_forget
                optimizer.zero_grad()
                if pixel_values is None:
                    input_ids, attention_mask, labels = (
                        input_ids.to(args.device), attention_mask.to(args.device), labels.to(args.device)
                    )
                    output_forget = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        output_hidden_states=True
                    )
                else:
                    if "Qwen" in args.model:
                        input_ids, attention_mask, pixel_values, grid_thw, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), grid_thw.to(args.device), labels.to(args.device)
                        )
                        output_forget = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            image_grid_thw=grid_thw,
                            labels=labels,
                            output_hidden_states=True
                        )
                    if "llava" in args.model:
                        input_ids, attention_mask, pixel_values, labels = (
                            input_ids.to(args.device), attention_mask.to(args.device),
                            pixel_values.to(args.device), labels.to(args.device)
                        )
                        output_forget = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels,
                            output_hidden_states=True
                        )
                h_forget = output_forget.hidden_states[7]
                loss_unlearn = torch.mean(torch.norm(h_forget - c * u, dim=1)**2)

                loss = loss_retain + alpha * loss_unlearn
            else:
                loss = loss_retain
                loss_unlearn = torch.tensor(0.0, device=args.device)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            all_loss.append(loss.item())
            if idx % 50 == 0:
                tqdm.write(f"Retain Set Batch {idx + 1}/{len(retain_loader)} finished, loss: {loss.item():.4f}, loss_retain: {loss_retain.item():.4f}, loss_unlearn: {loss_unlearn.item():.4f}")
    # save lora model
    if save:
        model.save_pretrained(f"{args.output_file_path}model_caches/{args.model}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}_finetune.pth")

    import json
    with open(f'{args.output_file_path}/loss_{args.this_run_id}.json', 'w') as f:
        json.dump(all_loss, f)
    print('saved')


    return model


def adaptive_rmu_finetune(updated_model, frozen_model, retain_loader, forget_loader, args, save=True):
    def get_params(model, layer_ids, param_ids):
        params = []
        for layer_id in layer_ids:
            for i, p in enumerate(model.model.layers[layer_id].parameters()):
                if i in param_ids:
                    params.append(p)
        return params    
    
    def forward_with_cache(model, inputs, module, no_grad=True):
        # define a tensor with the size of our cached activations
        cache = []
        def hook(module, input, output):
            if isinstance(output, tuple):
                cache.append(output[0])
            else:
                cache.append(output)
            return None 
        
        hook_handle = module.register_forward_hook(hook)
        
        if no_grad:
            with torch.no_grad():
                if "Qwen" in model.config.architectures[0]:
                    input_ids, attention_mask, pixel_values, image_grid_thw, labels = inputs[0].to(model.device), inputs[1].to(model.device), inputs[2].to(model.device), inputs[3].to(model.device), inputs[4].to(model.device)
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)
                elif "Llava" in model.config.architectures[0]:
                    input_ids, attention_mask, pixel_values, labels = inputs[0].to(model.device), inputs[1].to(model.device), inputs[2].to(model.device), inputs[3].to(model.device)
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
                elif "Gemma" in model.config.architectures[0]:
                    inputs = inputs[0]
                    input_ids, attention_mask, pixel_values, labels = inputs["input_ids"].to(model.device), inputs["attention_mask"].to(model.device), inputs["pixel_values"].to(model.device), inputs["labels"].to(model.device)
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
        else:
            if "Qwen" in model.config.architectures[0]:
                input_ids, attention_mask, pixel_values, image_grid_thw, labels = inputs[0].to(model.device), inputs[1].to(model.device), inputs[2].to(model.device), inputs[3].to(model.device), inputs[4].to(model.device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)
            elif "Llava" in model.config.architectures[0]:
                input_ids, attention_mask, pixel_values, labels = inputs[0].to(model.device), inputs[1].to(model.device), inputs[2].to(model.device), inputs[3].to(model.device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
            elif "Gemma" in model.config.architectures[0]:
                inputs = inputs[0]
                input_ids, attention_mask, pixel_values, labels = inputs["input_ids"].to(model.device), inputs["attention_mask"].to(model.device), inputs["pixel_values"].to(model.device), inputs["labels"].to(model.device)
                _ = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
        hook_handle.remove()

        return cache[0]

    def forward_model(batch, model_target, model_name):
        if "Qwen" not in model_name and "llava" not in model_name:
            common_kwargs = batch[0].to(args.device)
        
        else:
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            pixel_values = batch[2]
            if "llava" in model_name:
                labels = batch[3].to(args.device)
            else:
                labels = batch[4].to(args.device)

            common_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            if pixel_values is not None:
                pixel_values = pixel_values.to(args.device)
                common_kwargs["pixel_values"] = pixel_values

                if "Qwen" in model_name:
                    common_kwargs["image_grid_thw"] = batch[3].to(args.device)

        return model_target(
            **common_kwargs,
            output_hidden_states=True
        )
    layer_id = args.rmu_layer_id
    steering_coeff = args.rmu_steering_coeff
    alpha = args.rmu_alpha
    beta = args.rmu_beta
    optimizer = torch.optim.AdamW(updated_model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0,
                                 num_training_steps=args.epochs * len(retain_loader))
    if "Qwen" in updated_model.config.architectures[0]:
        frozen_module = eval(
            "{model_name}.model.layers[{layer_id}]".format(model_name="frozen_model", layer_id=layer_id)
        )
        updated_module = eval(
            "{model_name}.model.layers[{layer_id}]".format(model_name="updated_model.base_model.model", layer_id=layer_id)
        )
    if "Llava" in updated_model.config.architectures[0] or "Gemma" in updated_model.config.architectures[0]:
        frozen_module = eval(
            "{model_name}.language_model.model.layers[{layer_id}]".format(model_name="frozen_model", layer_id=layer_id)
        )
        updated_module = eval(  
            "{model_name}.language_model.model.layers[{layer_id}]".format(model_name="updated_model", layer_id=layer_id)
        )
    
    if hasattr(updated_model.config, 'hidden_size'):
        hidden_size = updated_model.config.hidden_size
    elif hasattr(updated_model.config, 'text_config') and hasattr(updated_model.config.text_config, 'hidden_size'):
        hidden_size = updated_model.config.text_config.hidden_size
    else:
        raise ValueError("Model configuration does not contain 'hidden_size' or 'text_config.hidden_size'.")
    
    random_vector = torch.rand(1, 1, hidden_size, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = random_vector / torch.norm(random_vector) * steering_coeff

    all_loss = []

    updated_model.train()
    for epoch in range(args.finetune_epochs):
        coeffs = args.rmu_coeffs
        for idx, (batch_r, batch_f) in tqdm(enumerate(zip(retain_loader, forget_loader)),
                                            total=len(retain_loader), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            updated_forget_activations = forward_with_cache(
                    updated_model, batch_f, module=updated_module, no_grad=False
            ).to(updated_model.device)
            # if idx == 0:
            #     coeffs = torch.mean(updated_forget_activations.norm(dim=-1).mean(dim=1), dim=0).item() * 5.0 
            # else:
            #     pass
            unlearn_loss = torch.nn.functional.mse_loss(
                updated_forget_activations, control_vec * coeffs
            )
            updated_retain_activations = forward_with_cache(
                updated_model, batch_r, module=updated_module, no_grad=False
            ).to(updated_model.device)
            frozen_retain_activations = forward_with_cache(
                frozen_model, batch_r, module=frozen_module, no_grad=True
            ).to(updated_model.device)

            retain_loss = torch.nn.functional.mse_loss(
                updated_retain_activations, frozen_retain_activations
            )

            # Update model
            loss = alpha * retain_loss + beta * unlearn_loss
            loss.backward()
            optimizer.step()
            
            lr_scheduler.step()
            all_loss.append(loss.item())
            if idx % 10 == 0:
                tqdm.write(f"[{epoch}/{args.finetune_epochs}] Step {idx} | loss: {loss.item():.4f} "
                           f"| retain: {retain_loss.item():.4f} | unlearn: {unlearn_loss.item():.4f}")

    if save:
        updated_model.save_pretrained(f"{args.output_file_path}model_caches/"
                              f"{args.model}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}_finetune.pth")
    
    import json
    with open(f'{args.output_file_path}/loss_{args.this_run_id}.json', 'w') as f:
        json.dump(all_loss, f)
    print('saved')

    return updated_model

def evaluate_clf(dataset, 
                 processor, 
                 dataset_split: Literal['forget', 'retain'], 
                 data_modality: Literal['multi', 'text'],
                 args, 
                 model=None):
    # Evaluation loop
    if data_modality == 'multi':
        if "mllmu" in args.dataset:
            preds = data_process_clf_mllmu_batch(dataset, processor, model, args, modality='multi', data_type="train")
        elif "clear" in args.dataset:
            preds = data_process_clf_clear_batch(dataset, processor, model, args)
    elif data_modality == 'text':
        if "mllmu" in args.dataset:
            preds = data_process_clf_mllmu_batch(dataset, processor, model, args, modality='text', data_type="train")
        elif "clear" in args.dataset:
            assert False
    else:
        assert False

    model.cpu()
    preds, acc_by_llm = score_batch(preds, 'clf', args)
    model.to(args.device)

    msg = f"{dataset_split}set Finished.\tAccuracy by LLM: {acc_by_llm:.2%}"
    print(msg)
    # save the results
    write_logger(args, msg, args.output_file_path + "logs/")

    os.makedirs(args.output_file_path + f"logs/{args.this_run_id}", exist_ok=True)
    now = datetime.datetime.now().strftime('%d%H%M%S')
    pred_save_path = args.output_file_path + f"logs/{args.this_run_id}/clf_{dataset_split}set_{data_modality}_preds_{now}.json"
    with open(pred_save_path, "w") as f:
        import json
        data = {
            "remark": "",
            "args": args.__dict__,
            # "acc": accuracy,
            "acc_by_llm": acc_by_llm,
            "dataset_split": dataset_split,
            "data_modality": data_modality,
            "preds": preds
        }
        json.dump(data, f, indent=2, ensure_ascii=False) 


def evaluate_gen(forgetset, 
                 processor, 
                 dataset_split: Literal['forget', 'retain'], 
                 data_modality: Literal['multi', 'text'], 
                 args, 
                 model=None):
    # Evaluation loop
    if data_modality == 'multi':
        if "mllmu" in args.dataset:
            preds = data_process_gen_mllmu_batch(forgetset, processor, model, args, modality='multi', data_type="train")
        if "clear" in args.dataset:
            preds = data_process_gen_clear_batch(forgetset, processor, model, args, modality='multi')
    elif data_modality == 'text':
        if "mllmu" in args.dataset:
            preds = data_process_gen_mllmu_batch(forgetset, processor, model, args, modality='text', data_type="train")
        if "clear" in args.dataset:
            preds = data_process_gen_clear_batch(forgetset, processor, model, args, modality='text')
    else:
        assert False
    
    # calculate rouge and bleu
    from metrics.bleu.bleu import Bleu
    from metrics.rouge.rouge import Rouge
    bleu = Bleu()
    rouge = Rouge()
    try:
        bleu_scores = bleu.compute(predictions=[p['pred'] for p in preds], references=[p['gt'] for p in preds])
    except ZeroDivisionError:
        bleu_scores = {'bleu': 0}
    rouge_scores = rouge.compute(predictions=[p['pred'] for p in preds], references=[p['gt'] for p in preds])
    bleumean = bleu_scores['bleu']
    rouge1mean = rouge_scores['rouge1']
    rouge2mean = rouge_scores['rouge2']
    rougeLmean = rouge_scores['rougeL']
    rougeLsummean = rouge_scores['rougeLsum']

    model.cpu()

    preds, acc = score_batch(preds, 'gen', args)
    model.to(args.device)

    msg = f"{dataset_split}set Finished. acc: {acc:.2%}, Rouge1: {rouge1mean:.2%}, Rouge2: {rouge2mean:.2%}, RougeL: {rougeLmean:.2%}, RougeLsum: {rougeLsummean:.2%}, Bleu: {bleumean:.2%}"
    print(msg)
    # save the results
    write_logger(args, msg, args.output_file_path + "logs/")
    now = datetime.datetime.now().strftime('%d%H%M%S')
    
    os.makedirs(args.output_file_path + f"logs/{args.this_run_id}", exist_ok=True)
    pred_save_path = args.output_file_path + f"logs/{args.this_run_id}/gen_{dataset_split}set_{data_modality}_preds_{now}.json"
    with open(pred_save_path, "w") as f:
        import json
        data = {
            "remark": "",
            "args": args.__dict__,
            "acc": acc,
            "bleu": bleumean,
            "rouge1": rouge1mean,
            "rouge2": rouge2mean,
            "rougeL": rougeLmean,
            "rougeLsum": rougeLsummean,
            "dataset_split": dataset_split,
            "data_modality": data_modality,
            "preds": preds
        }
        json.dump(data, f, indent=2, ensure_ascii=False) 


def score_batch(preds, task, args):
    batch_size = 16

    gc.collect()
    torch.cuda.empty_cache()
    score_llm = None
    try:
        score_llm, tokenizer = load_llm(f"{args.llm_directory}{args.score_llm}", args.device)
        print(f'scoring by llm')
        preds = score_by_llm_batch(score_llm, tokenizer, preds, batch_size, task, args.device)
        acc = sum(map(lambda r: int(r['correct']), preds)) / len(preds)
        return preds, acc
    except Exception as e:
        print(f"Error during scoring: {type(e)}  {e}")
        if score_llm is not None:
            del score_llm
        torch.cuda.empty_cache()
        gc.collect()
        return preds, -1