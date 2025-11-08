import torch
from torch.nn import functional as F
from tqdm import tqdm
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from transformers import get_scheduler
import re


def batch2dict(batch, model, device):
    config = model.config if hasattr(model, 'config') else model.text_config
    name = config._name_or_path
    if 'Qwen' in name:
        input_ids, attention_mask, pixel_values, grid_thw, labels, _ = batch
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'pixel_values': pixel_values.to(device) if pixel_values is not None else None,
            'image_grid_thw': grid_thw.to(device) if grid_thw is not None else None,
            'labels': labels.to(device)
        }
    elif 'llava' in name:
        input_ids, attention_mask, pixel_values, labels, _ = batch
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'pixel_values': pixel_values.to(device) if pixel_values is not None else None,
            'labels': labels.to(device)
        }
    else:
        raise ValueError(f"Unsupported model type: {model.__class__.__name__}")
    # return batch
    


def ga_difference_training(forget_loader, retain_loader, model, args):
    forget_training_epochs = getattr(args, 'baseline_forget_training_epochs', 2)
    setattr(args, 'baseline_forget_training_epochs', forget_training_epochs)


    model.print_trainable_parameters()
    print("Forget training starts!")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(forget_loader) + len(retain_loader)) * forget_training_epochs,
    )
    model.train()
    for epoch in range(forget_training_epochs):
        epoch_forget_loss = 0
        epoch_retain_loss = 0
        for batch in tqdm(forget_loader):
            inputs = batch2dict(batch, model, args.device)
            optimizer.zero_grad()
            outputs = model(
                **inputs,
            )
            forget_loss = -outputs.loss
            forget_loss.backward()
            epoch_forget_loss += forget_loss.item()
            optimizer.step()
            lr_scheduler.step()


        for batch in tqdm(retain_loader):
            inputs = batch2dict(batch, model, args.device)
            optimizer.zero_grad()
            outputs = model(
                **inputs,
            )
            retain_loss = outputs.loss
            retain_loss.backward()
            optimizer.step()
            epoch_retain_loss += retain_loss.item()
            lr_scheduler.step()
        
        epoch_forget_loss /= len(forget_loader)
        epoch_retain_loss /= len(retain_loader)
        print(f"Epoch {epoch} forget_loss: {epoch_forget_loss}, retain_loss: {epoch_retain_loss}")
    return model





def kl_min(forget_loader, retain_loader, model, vanilla_model, args):
    forget_training_epochs = getattr(args, 'baseline_forget_training_epochs', 2)
    setattr(args, 'baseline_forget_training_epochs', forget_training_epochs)

    # vanilla_model.cpu()
    vanilla_model.eval()
    model.print_trainable_parameters()
    print("Forget training starts!")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(forget_loader) + len(retain_loader)) * forget_training_epochs,
    )

    model.train()
    for epoch in range(forget_training_epochs):
        epoch_forget_loss = 0
        epoch_retain_loss = 0
        for batch in tqdm(forget_loader, desc=f"Forget Training Epoch {epoch+1}/{forget_training_epochs}"):
            inputs = batch2dict(batch, model, args.device)
            outputs = model(
                **inputs,
            )
            forget_loss = -outputs.loss
            forget_loss.backward()
            epoch_forget_loss += forget_loss.item()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()


        for batch in tqdm(retain_loader, desc=f"Retain Training Epoch {epoch}/{forget_training_epochs}"):
            inputs = batch2dict(batch, model, args.device)

            # model.cpu()
            # vanilla_model.to(args.device)
            with torch.no_grad():
                outputs_original = vanilla_model(
                    **inputs
                )
            # vanilla_model.cpu()
            # model.to(args.device)
                

            outputs = model(
                **inputs,
            )

            kl_div = F.kl_div(
                F.log_softmax(outputs.logits, dim=-1), 
                F.softmax(outputs_original.logits, dim=-1), 
                reduction='batchmean'
            )

            retain_loss = outputs.loss + kl_div
            retain_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            epoch_retain_loss += retain_loss.item()
        
        epoch_forget_loss /= len(forget_loader)
        epoch_retain_loss /= len(retain_loader)
        print(f"Epoch {epoch} forget_loss: {epoch_forget_loss}, retain_loss: {epoch_retain_loss}")
    return model




def npo(forget_loader, retain_loader, model, oracle_model, args):
    """
    """

    forget_training_epochs = getattr(args, 'baseline_forget_training_epochs', 2)
    setattr(args, 'baseline_forget_training_epochs', forget_training_epochs)
    beta = getattr(args, 'baseline_npo_beta', 0.4)
    setattr(args, 'baseline_npo_beta', beta)

    # oracle_model.cpu()
    oracle_model.eval()
    model.print_trainable_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(forget_loader) + len(retain_loader)) * forget_training_epochs,
    )

    model.train()
    for epoch in range(forget_training_epochs):
        epoch_retain_loss = 0

        for batch in tqdm(retain_loader, desc=f"Retain Training Epoch {epoch}/{forget_training_epochs}"):
            inputs = batch2dict(batch, model, args.device)

            with torch.no_grad():
                outputs_original = oracle_model(
                    **inputs
                )

            outputs = model(
                **inputs,
            )

            neg_log_ratios = outputs.loss - outputs_original.loss
            retain_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta
            retain_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            epoch_retain_loss += retain_loss.item()
        
        epoch_retain_loss /= len(retain_loader)
        print(f"Epoch {epoch} retain_loss: {epoch_retain_loss}")
    return model

