import json
from typing import Literal
import torch
import re
from metrics.rouge.rouge import Rouge
from metrics.bleu.bleu import Bleu
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from partial_linear import PartialLinear
import numpy as np
from torch.nn import functional as F
from transformers import AutoTokenizer
import random

image_caption_questions = [
    "What can you see in this picture?",
    "Tell me about the content of this image.",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]


def check_answer(question, options, pred, truth):
    if truth.lower() == "none of the above":
        if "none" in pred.lower():
            return 1
        else:
            return 0
    # 判断答案标号
    if pred[0].upper() == truth:
        return 1
    # 判断正确答案是否包含在pred中
    if options[truth].lower() in pred.lower():
        return 1
    # 判断数字类答案
    if ''.join(re.findall(r'\d+', options[truth])) == ''.join(re.findall(r'\d+', pred)) and ''.join(re.findall(r'\d+', pred)) != '':
        return 1
    # 判断正确答案的某些介词后文本是否出现在pred中
    if any(kw in options[truth].lower() and 
            options[truth].lower().split(kw)[0].strip() in pred.lower() 
            for kw in ['in', 'from']):
        return 1
    else:
        return 0


def data_process_clf_mllmu(dataset, processor, model, args, data_type="train"):
    model.eval()
    ans = []
    labels = []
    correct = 0
    total = 0

    
    pred_list = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating on clf"):
        tc = dataset[idx]["Classification_Task"]['Image_Textual_Questions']
        image_list = []
        if data_type == "test":
            for img in dataset[idx]['images']:
                image_bytes = img.get('bytes')
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((args.image_resize, args.image_resize))
                image_list.append(image)
        else:
            image_bytes = dataset[idx]['image'].get('bytes')
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((args.image_resize, args.image_resize))
            image_list.append(image)

        for i in range(0, len(tc)):
            q = tc[i]["Question"]
            o = tc[i]["Options"]
            if data_type == "test":
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": q + " Options:" + " A:" + o["A"] + " B:" + o["B"] + " C:" + o["C"] + " D:" + o["D"] + "Must give ONE letter representing the answer directly."},
                        ]
                    }
                ]
            elif data_type == "train":
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q + " Options:" + " A:" + o["A"] + " B:" + o["B"] + " C:" + o["C"] + " D:" + o["D"] + "Must give ONE letter representing the answer directly."},
                        ]
                    }
                ]
            
            question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[question], images=image_list, padding=False, return_tensors="pt").to(args.device)
            if "Qwen" in args.model:
                input_ids, attention_mask, pixel_values, grid_thw = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"]
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=grid_thw,
                    max_new_tokens=40,
                )
            if "llava" in args.model:
                input_ids, attention_mask, pixel_values = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_new_tokens=40,
                )
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            pred_idx = output_text[0].lower().find("assistant\n")
            pred = output_text[0][pred_idx + len("assistant\n"):].strip()
            correct += check_answer(q, o, pred, tc[i]['Correct_Answer'])
            total += 1

            pred_list.append({
                "question": q,
                "options": o,
                "gt": tc[i]['Correct_Answer'],
                "pred": pred,
            })
    accuracy = correct / total
    return accuracy, pred_list


def data_process_clf_clear(dataset, processor, model, args, data_type="train"):
    model.eval()
    ans = []
    labels = []
    correct = 0
    total = 0

    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    valid_letter_tokens = list(map(
        lambda letter: tokenizer(letter, add_special_tokens=False, return_tensors="pt")['input_ids'][0],
        ['A', 'B', 'C', 'D', 'E']
    ))
    
    pred_list = []

    for idx in tqdm(range(len(dataset))):
        image = dataset[idx]["image"]
        image = image.resize((args.image_resize, args.image_resize))
        q = dataset[idx].get("question", "Answer with ONE LETTER direclty.")
        t = dataset[idx]["perturbed_captions"]
        o = {}
        o["A"], o["B"], o["C"], o["D"], o["E"], o["F"] = t[0], t[1], t[2], t[3], t[4], dataset[idx]["caption"]
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{q} Options: A. {o['A']} B. {o['B']} C. {o['C']} D. {o['D']} E. {o['E']} F. {o['F']}."},
                ]
            }
        ]
            
        question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[question], images=image, padding=False, return_tensors="pt").to(args.device)
        if "Qwen" in args.model:
            input_ids, attention_mask, pixel_values, grid_thw = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"]
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                max_new_tokens=40,
                min_new_tokens=1,
                # prefix_allowed_tokens_fn=lambda batch_id, sent: valid_letter_tokens,
            )
        if "llava" in args.model:
            input_ids, attention_mask, pixel_values = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=40,
                min_new_tokens=1,
                # prefix_allowed_tokens_fn=lambda batch_id, sent: valid_letter_tokens,
            )
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        pred_idx = output_text[0].lower().find("assistant\n")
        pred = output_text[0][pred_idx + len("assistant\n"):].strip()
        # correct += check_answer(q, o, pred, "E")
        # correct += check_answer(q, o, pred, dataset[idx]["name"])
        total += 1

        pred_list.append({
            "question": q,
            "options": o,
            "gt": dataset[idx]["name"],
            "pred": pred,
        })
    accuracy = correct / total
    return accuracy, pred_list


def data_process_clf_clear_batch(dataset, processor, model, args):
    batch_size = 28

    model.eval()

    from test_data import ClearClfDataset, collator

    pred_list = []
    dataset = ClearClfDataset(dataset, processor, args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collator(x, processor, args),
    )
    for idx, sample in enumerate(tqdm(dataloader, desc=f"Evaluating on clear clf")):
        question_list = sample['question']
        options_list = sample['options']
        gt_list = sample['ground_truth']
        inputs = sample['inputs']

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            min_new_tokens=1,
        )
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i in range(len(question_list)):
            if "Qwen" in args.model:
                pred_idx = output_text[i].lower().find("assistant\n")
                pred = output_text[i][pred_idx + len("assistant\n"):].strip()
            if "llava" in args.model:
                pred_idx = output_text[i].find("ASSISTANT:")
                pred = output_text[i][pred_idx + len("ASSISTANT:"):].split("ASSISTANT:")[0].strip()
                pred = pred.split('\n')[0]
            # correct = check_answer(question_list[i], options_list[i], pred, sample['gt'][i])
            
            pred_list.append({
                "question": question_list[i],
                "options": options_list[i],
                "gt": gt_list[i],
                "pred": pred,
            })

        if idx % 5 == 0:
            tqdm.write(f"Batch {idx + 1}/{len(dataloader)} finished\n {json.dumps(pred_list[-1], indent=2, ensure_ascii=False)}")

    return pred_list



def data_process_clf_text(dataset, processor, model, args):
    model.eval()
    ans = []
    labels = []
    correct = 0
    total = 0

    # tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    # valid_letter_tokens = list(map(
    #     lambda letter: tokenizer(letter, add_special_tokens=False, return_tensors="pt")['input_ids'][0],
    #     ['A', 'B', 'C', 'D']
    # ))
    

    pred_list = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating on clf text"):
        tc = dataset[idx]["Classification_Task"]['Pure_Text_Questions']
        for i in range(0, len(tc)):
            q = tc[i]["Question"]
            o = tc[i]["Options"]
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant that answers questions based on the provided options. Answer with the letter corresponding to the correct option. For example, if the correct answer is option A, simply respond with 'A'."}
                    ]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": q + " Options:" + "\n A:" + o["A"] + "\n B:" + o["B"] + "\n C:" + o["C"] + "\n D:" + o["D"]},
                    ]
                }
            ]
            
            question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[question], padding=False, return_tensors="pt").to(args.device)
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=40,
                # prefix_allowed_tokens_fn=lambda batch_id, sent: valid_letter_tokens,
            )
            
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            pred_idx = output_text[0].lower().find("assistant\n")
            pred = output_text[0][pred_idx + len("assistant\n"):].strip()
            correct += check_answer(q, o, pred, tc[i]['Correct_Answer'])
            total += 1

            pred_list.append({
                "question": q,
                "options": o,
                "gt": tc[i]['Correct_Answer'],
                "pred": pred,
            })

            if idx % 50 == 0:
                if i == 0:
                    tqdm.write(f"Sample {idx + 1}/{len(dataset)} finished, \n\tques:\t{q} \n\tgt:\t{tc[i]['Correct_Answer']} \n\tpred:\t{pred}")
    accuracy = correct / total
    return accuracy, pred_list


def data_process_clf_mllmu_batch(dataset, processor, model, args, modality: Literal['multi', 'text'], data_type="train"):
    batch_size = 28

    model.eval()

    from test_data import MllmuClfDataset, collator

    pred_list = []
    dataset = MllmuClfDataset(dataset, processor, args, modality=modality, data_type=data_type)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collator(x, processor, args),
    )
    for idx, sample in enumerate(tqdm(dataloader, desc=f"Evaluating on mllmu clf {modality}")):
        question_list = sample['question']
        options_list = sample['options']
        gt_list = sample['ground_truth']
        inputs = sample['inputs']

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=40,
        )
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i in range(len(question_list)):
            if "Qwen" in args.model:
                pred_idx = output_text[i].lower().find("assistant\n")
                pred = output_text[i][pred_idx + len("assistant\n"):].strip()
            if "llava" in args.model:
                if 'mistral' in args.model:
                    pred_idx = output_text[i].find("[/INST]")
                    pred = output_text[i][pred_idx + len("[/INST]"):].split("[/INST]")[0].strip()
                else:
                    pred_idx = output_text[i].find("ASSISTANT:")
                    pred = output_text[i][pred_idx + len("ASSISTANT:"):].split("ASSISTANT:")[0].strip()
                pred = pred.split('\n')[0]
            if "gemma" in args.model:
                pred_idx = output_text[i].find("\nmodel\n")
                pred = output_text[i][pred_idx + len("\nmodel\n"):].strip()
            # correct = check_answer(question_list[i], options_list[i], pred, sample['gt'][i])
            
            pred_list.append({
                "question": question_list[i],
                "options": options_list[i],
                "gt": gt_list[i],
                "pred": pred,
            })

        if idx % 5 == 0:
            tqdm.write(f"Batch {idx + 1}/{len(dataset)} finished\n {json.dumps(pred_list[-1], indent=2)}")

    return pred_list


def data_process_gen_mllmu(dataset, processor, model, args, data_type="train"):
    model.eval()

    pred_list = []
    for idx in tqdm(range(len(dataset)), desc="Evaluating on mllmu gen"):
        image_list = []
        if data_type == "test":
            for img in dataset[idx]['images']:
                image_bytes = img.get('bytes')
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((args.image_resize, args.image_resize))
                image_list.append(image)
        else:
            image_bytes = dataset[idx]['image'].get('bytes')
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((args.image_resize, args.image_resize))
            image_list.append(image)
        gt = dataset[idx]["Generation_Task"]
        for i in range(0, len(gt)):
            q = gt[i]["Question"]
            g = gt[i]["Ground_Truth"]
            q_type = gt[i]["Type"]
            if q_type != "Image_Textual":
                continue

            if data_type == "test":
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ]
                    }
                ]
            elif data_type == "train":
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ]
                    }
                ]
            
            question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[question], images=image_list, padding=False, return_tensors="pt").to(args.device)
            input_ids, attention_mask, pixel_values, grid_thw = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"]
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                max_new_tokens=40
            )
            
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            pred_idx = output_text[0].lower().find("assistant\n")
            pred = output_text[0][pred_idx + len("assistant\n"):].strip()
            pred_list.append({
                "question": q,
                "gt": g,
                "pred": pred,
            })

            if idx % 50 == 0:
                if i == 0:
                    tqdm.write(f"Sample {idx + 1}/{len(dataset)} finished, \n\tques:\t{q} \n\tgt:\t{g} \n\tpred:\t{pred}")

    return pred_list



def data_process_gen_mllmu_batch(dataset, processor, model, args, modality: Literal['multi', 'text'], data_type="train"):
    batch_size = 28

    model.eval()

    from test_data import MllmuGenDataset, collator

    pred_list = []
    dataset = MllmuGenDataset(dataset, processor, args, data_type=data_type, modality=modality)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collator(x, processor, args),
    )
    for idx, sample in enumerate(tqdm(dataloader, desc=f"Evaluating on mllmu gen {modality}")):
        questions = sample['question']
        ground_truths = sample['ground_truth']
        inputs = sample['inputs']

        assert len(questions) == len(ground_truths)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=40
        )
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i in range(len(output_text)):
            if "Qwen" in args.model:
                pred_idx = output_text[i].lower().find("assistant\n")
                pred = output_text[i][pred_idx + len("assistant\n"):].strip()
            if "llava" in args.model:
                if 'mistral' in args.model:
                    pred_idx = output_text[i].find("[/INST]")
                    pred = output_text[i][pred_idx + len("[/INST]"):].split("[/INST]")[0].strip()
                else:
                    pred_idx = output_text[i].find("ASSISTANT:")
                    pred = output_text[i][pred_idx + len("ASSISTANT:"):].split("ASSISTANT:")[0].strip()
                pred = pred.split('\n')[0]
            if "gemma" in args.model:
                pred_idx = output_text[i].find("\nmodel\n")
                pred = output_text[i][pred_idx + len("\nmodel\n"):].strip()
            pred_list.append({
                "question": questions[i],
                "gt": ground_truths[i],
                "pred": pred,
            })
        
        if idx % 5 == 0:
            tqdm.write(f"Batch {idx + 1}/{len(dataloader)} finished\n{json.dumps(pred_list[-1], indent=2)}")

    return pred_list



def data_process_gen_text_mllmu(dataset, processor, model, args):
    model.eval()

    pred_list = []
    for idx in tqdm(range(len(dataset)), desc="Evaluating on mllmu gen text"):
        gt = dataset[idx]["Generation_Task"]
        for i in range(0, len(gt)):
            q = gt[i]["Question"]
            g = gt[i]["Ground_Truth"]
            q_type = gt[i]["Type"]
            if q_type != "Pure_Text":
                continue

            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": q},
                    ]
                }
            ]
            
            question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[question], padding=False, return_tensors="pt").to(args.device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=40
            )
            
            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            pred_idx = output_text[0].lower().find("assistant\n")
            pred = output_text[0][pred_idx + len("assistant\n"):].strip()
            pred_list.append({
                "question": q,
                "gt": g,
                "pred": pred,
            })

            if idx % 50 == 0:
                if i == 0:
                    tqdm.write(f"Sample {idx + 1}/{len(dataset)} finished, \n\tques:\t{q} \n\tgt:\t{g} \n\tpred:\t{pred}")

    return pred_list



def data_process_gen_clear(dataset, processor, model, args):
    model.eval()

    pred_list = []
    for idx in tqdm(range(len(dataset)), desc="Evaluating on gen clear"):
        image = dataset[idx]["image"]
        if image is None:
            continue
        image = image.resize((args.image_resize, args.image_resize))
        gt = dataset[idx]["caption"]
        q = random.choice(image_caption_questions)
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ]
            }
        ]
        question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[question], images=image, padding=False, return_tensors="pt").to(args.device)
        if "Qwen" in args.model:
            input_ids, attention_mask, pixel_values, grid_thw = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"], inputs["image_grid_thw"]
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=grid_thw,
                max_new_tokens=100
            )
        if "llava" in args.model:
            input_ids, attention_mask, pixel_values = inputs["input_ids"], inputs["attention_mask"], inputs["pixel_values"]
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=100
            )
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        pred_idx = output_text[0].lower().find("assistant\n")
        pred = output_text[0][pred_idx + len("assistant\n"):].strip()
        pred_list.append({
            "question": q,
            "gt": gt,
            "pred": pred,
        })

        if idx % 1 == 0:
            tqdm.write(f"Sample {idx + 1}/{len(dataset)} finished, \n\tques:\t{q} \n\tgt:\t{gt} \n\tpred:\t{pred}")

    return pred_list


def data_process_gen_text_clear(dataset, processor, model, args):
    model.eval()

    pred_list = []
    for idx in tqdm(range(len(dataset)), desc="Evaluating on gen text clear"):
        image = dataset[idx]["image"]
        if image != None:
            continue
        gt = dataset[idx]["answer"]
        q = dataset[idx]["question"]
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": q},
                ]
            }
        ]
        question = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[question], padding=False, return_tensors="pt").to(args.device)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100
        )
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        pred_idx = output_text[0].lower().find("assistant\n")
        pred = output_text[0][pred_idx + len("assistant\n"):].strip()
        pred_list.append({
            "question": q,
            "gt": gt,
            "pred": pred,
        })

        if idx % 1 == 0:
            tqdm.write(f"Sample {idx + 1}/{len(dataset)} finished, \n\tques:\t{q} \n\tgt:\t{gt} \n\tpred:\t{pred}")

    return pred_list


def data_process_gen_clear_batch(dataset, processor, model, args, modality: Literal['multi', 'text']):
    batch_size = 28

    model.eval()

    from test_data import ClearGenDataset, collator

    pred_list = []
    dataset = ClearGenDataset(dataset, processor, args, modality=modality)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collator(x, processor, args),
    )
    for idx, sample in enumerate(tqdm(dataloader, desc=f"Evaluating on clear gen {modality}")):
        questions = sample['question']
        ground_truths = sample['ground_truth']
        inputs = sample['inputs']

        assert len(questions) == len(ground_truths)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100
        )
        
        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        

        for i in range(len(output_text)):
            if "Qwen" in args.model:
                pred_idx = output_text[i].lower().find("assistant\n")
                pred = output_text[i][pred_idx + len("assistant\n"):].strip()
            if "llava" in args.model:
                pred_idx = output_text[i].find("ASSISTANT:")
                pred = output_text[i][pred_idx + len("ASSISTANT:"):].split("ASSISTANT:")[0].strip()
                pred = pred.split('\n')[0]
            pred_list.append({
                "question": questions[i],
                "gt": ground_truths[i],
                "pred": pred,
            })
        
        if idx % 5 == 0:
            tqdm.write(f"Batch {idx + 1}/{len(dataloader)} finished\n{json.dumps(pred_list[-1], indent=2)}")

    return pred_list

def get_unique_idxs(paths, scores, length):
        new_paths = [[] for _ in range(length)]
        new_scores = [{} for _ in range(length)]
        for base, score in zip(paths, scores):
            for j in range(length):
                for k, path in enumerate(base[j]):
                    if path not in new_scores[j]:
                        new_scores[j][path.item()] = score[j][k].item()
                    else:
                        new_scores[j][path.item()] += score[j][k].item()

        for i in range(length):
            new_paths[i] = list(new_scores[i].keys())
            new_scores[i] = list(new_scores[i].values())

        return new_paths, new_scores

def get_unique_idxs2(paths, scores, length):
    new_paths = []
    new_scores = []
    
    for j in tqdm(range(length)):
        # Extract all paths and scores for the current layer
        layer_paths = torch.cat([p[j] for p in paths])
        layer_scores = torch.cat([s[j] for s in scores])
        
        # Get unique paths and their indices
        unique_paths, inverse_indices = torch.unique(layer_paths, return_inverse=True)
        
        # Use scatter_add to sum scores for each unique path
        unique_scores = torch.zeros_like(unique_paths, dtype=layer_scores.dtype)
        unique_scores.scatter_add_(0, inverse_indices, layer_scores)
        
        # Sort by path values for consistency (optional)
        sorted_indices = torch.argsort(unique_paths)
        sorted_paths = unique_paths[sorted_indices]
        sorted_scores = unique_scores[sorted_indices]
        
        new_paths.append(sorted_paths.tolist())
        new_scores.append(sorted_scores.tolist())
    
    return new_paths, new_scores    

def ffn_at_layer(llm, layer):
    if "Qwen2.5-VL" in llm.config._name_or_path:
        if hasattr(llm.model, "language_model"):
            return llm.model.language_model.layers[layer].mlp
        else:
            return llm.model.layers[layer].mlp
    elif "Qwen2-VL" in llm.config._name_or_path:
        return llm.model.layers[layer].mlp
    elif "llava" in llm.config._name_or_path:
        return llm.language_model.model.layers[layer].mlp
    elif "gemma" in llm.config._name_or_path:
        return llm.language_model.layers[layer].mlp
    else:
        raise ValueError(f"Unsupported model type: {llm.config._name_or_path}")

def vb_at_layer(llm, layer):
    if "Qwen" in llm.config._name_or_path:
        return llm.visual.blocks[layer].mlp
    elif "llava" in llm.config._name_or_path:
        return llm.vision_tower.vision_model.encoder.layers[layer].mlp
    else:
        raise ValueError(f"Unsupported model type: {llm.config._name_or_path}")

def freeze_parameters(model, multi_idxs_forget_visual, text_idxs_forget_model, apply_to_visual=True):
    model.requires_grad_(False)
    for layer, fo_text_idx in zip(range(len(text_idxs_forget_model)), text_idxs_forget_model):
        # fo_idx = fo_mul_idx + fo_text_idx
        fo_idx = fo_text_idx
        ffn = ffn_at_layer(model, layer)
        ffn.up_proj = PartialLinear(
            ffn.up_proj, trainable_cols=fo_idx
        )
        ffn.gate_proj = PartialLinear(
            ffn.gate_proj, trainable_cols=fo_idx
        )
    if apply_to_visual:
        for layer, fo_mul_idx in zip(range(len(multi_idxs_forget_visual)), multi_idxs_forget_visual):
            vb = vb_at_layer(model, layer)
            if "Qwen2.5-VL" in model.config._name_or_path:
                vb.up_proj = PartialLinear(
                    vb.up_proj, trainable_cols=fo_mul_idx
                )
                vb.gate_proj = PartialLinear(
                    vb.gate_proj, trainable_cols=fo_mul_idx
                )
            elif "Qwen2-VL" in model.config._name_or_path:
                vb.fc1 = PartialLinear(
                    vb.fc1, trainable_cols=fo_mul_idx
                )
            elif "llava" in model.config._name_or_path:
                vb.fc1 = PartialLinear(
                    vb.fc1, trainable_cols=fo_mul_idx
                )
            else:
                assert False
    return model

def model_layer_prune_to_zero(model, prune_indices):
    assert get_language_model_num_layers(model) == len(prune_indices)


    def hook_fn(module, inputs, prune_indices):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        inputs[:, 0, prune_indices] = 0
    hooks = []

    modules = [
        ffn_at_layer(model, i).down_proj for i in range(get_language_model_num_layers(model))
    ]
    for i, module in enumerate(modules):
        hook = module.register_forward_pre_hook(
            lambda m, inp, prune_indices=prune_indices[i] :hook_fn(m, inp, prune_indices)
        )
        hooks.append(hook)
    def cancel_hooks():
        for hook in hooks:
            hook.remove()
    return model, cancel_hooks

def visual_block_prune_to_zero(model, prune_indices):
    assert get_vision_model_num_layers(model) == len(prune_indices)

    def hook_fn(module, inputs, prune_indices):
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        if inputs.dim() == 3:
            inputs[:, 0, prune_indices] = 0
        else:
            inputs[0, prune_indices] = 0
    hooks = []

    modules = []
    for i in range(get_vision_model_num_layers(model)):
        vb = vb_at_layer(model, i)
        if "Qwen2.5-" in model.config._name_or_path:
            modules.append(vb.down_proj)
        elif "Qwen2-" in model.config._name_or_path:
            modules.append(vb.fc2)
        elif "llava" in model.config._name_or_path:
            modules.append(vb.fc2)
        elif "gemma" in model.config._name_or_path:
            modules.append(vb.fc2)
        else:
            assert False
    assert len(modules) == len(prune_indices)
    for i, module in enumerate(modules):
        hook = module.register_forward_pre_hook(
            lambda m, inp, prune_indices=prune_indices[i] :hook_fn(m, inp, prune_indices)
        )
        hooks.append(hook)
    def cancel_hooks():
        for hook in hooks:
            hook.remove()
    return model, cancel_hooks


def get_language_model_num_layers(model):
    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        num_layers = model.config.num_hidden_layers
    
    return num_layers

def get_vision_model_num_layers(model):
    if "llava" in model.config._name_or_path:
        return model.config.vision_config.num_hidden_layers
    else:
        return model.config.vision_config.depth

def get_llm_name(model):
    return model.config._name_or_path


def prune_linear_weight_to_zero(
    module: torch.nn.Linear,
    neuron_indices: torch.Tensor | list[int],
):
    module.weight[neuron_indices, : ] = 0
    return module
