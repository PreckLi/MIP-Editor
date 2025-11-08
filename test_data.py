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


class MllmuGenDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, args, data_type="train", modality: Literal["multi", "text"] = "multi"):
        self.raw_dataset = dataset
        self.processor = processor
        self.args = args
        self.data_type = data_type

        self.modality = modality
        assert self.modality in ["multi", "text"], "Modality must be either 'multi' or 'text'"

        self.MODALITY_TO_QUESTION_TYPE_MAP = {
            "multi": "Image_Textual",
            "text": "Pure_Text"
        }

        self.flattened_dataset = self._flatten_dataset()

    def _read_image(self, sample):
        if self.modality == "text":
            return []

        image_list = []
        if self.data_type == "test":
            for img in sample['images']:
                image_bytes = img.get('bytes')
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((self.args.image_resize, self.args.image_resize))
                image_list.append(image)
        else:
            image_bytes = sample['image'].get('bytes')
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((self.args.image_resize, self.args.image_resize))
            image_list.append(image)

        return image_list


    def _flatten_dataset(self):
        flattened = []
        for i in range(len(self.raw_dataset)):
            sample = self.raw_dataset[i]

            image_list = self._read_image(sample)

            gt = sample["Generation_Task"]
            for task in gt:
                q = task["Question"]
                g = task["Ground_Truth"]
                q_type = task["Type"]
                if q_type != self.MODALITY_TO_QUESTION_TYPE_MAP[self.modality]:
                    continue

                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"}
                            for _ in range(len(image_list))
                        ] + [
                            {"type": "text", "text": q},
                        ]
                    }
                ]

                question = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                
                if self.modality == "multi":
                    flattened.append({
                        'question': q,
                        'ground_truth': g,
                        'images': image_list,
                        'chat': question,
                    })
                else:
                    flattened.append({
                        'question': q,
                        'ground_truth': g,
                        'chat': question,
                    })
        
        return flattened

    def __len__(self):
        return len(self.flattened_dataset)

    def __getitem__(self, idx):
        return self.flattened_dataset[idx]



class MllmuClfDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, args, data_type="train", modality: Literal["multi", "text"] = "multi"):
        self.raw_dataset = dataset
        self.processor = processor
        self.args = args
        self.data_type = data_type

        self.modality = modality
        assert self.modality in ["multi", "text"], "Modality must be either 'multi' or 'text'"

        self.MODALITY_TO_QUESTION_TYPE_MAP = {
            "multi": "Image_Textual_Questions",
            "text": "Pure_Text_Questions"
        }

        self.flattened_dataset = self._flatten_dataset()

    def _read_image(self, sample):
        if self.modality == "text":
            return []

        image_list = []
        if self.data_type == "test":
            for img in sample['images']:
                image_bytes = img.get('bytes')
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((self.args.image_resize, self.args.image_resize))
                image_list.append(image)
        else:
            image_bytes = sample['image'].get('bytes')
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            image = image.resize((self.args.image_resize, self.args.image_resize))
            image_list.append(image)

        return image_list


    def _flatten_dataset(self):
        flattened = []
        for i in range(len(self.raw_dataset)):
            sample = self.raw_dataset[i]
            image_list = self._read_image(sample)

            ct = sample["Classification_Task"][self.MODALITY_TO_QUESTION_TYPE_MAP[self.modality]]
            for task in ct:
                q = task["Question"]
                o = task["Options"]

                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant that answers questions based on the provided options. Respond with a single letter corresponding to the correct option."}
                            # {"type": "text", "text": "You are a helpful assistant that answers questions based on the provided options.\n"}
                            # {"type": "text", "text": "You are a helpful assistant that answers questions based on the provided options. Respond with a single letter corresponding to the correct option. Such as A, B, C, or D."}
                        ]
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"}
                            for _ in range(len(image_list))
                        ] + [
                            # {"type": "text", "text": q + " Options:" + "\n A:" + o["A"] + "\n B:" + o["B"] + "\n C:" + o["C"] + "\n D:" + o["D"]},
                            {"type": "text", "text": q + "\nOptions:" + "\n A:" + o["A"] + "\n B:" + o["B"] + "\n C:" + o["C"] + "\n D:" + o["D"] + '\n'},
                        ]
                    }
                ]

                question = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                if self.modality == "multi":
                    flattened.append({
                        'question': q,
                        'options': o,
                        'ground_truth': task["Correct_Answer"],
                        'images': image_list,
                        'chat': question,
                    })
                else:
                    flattened.append({
                        'question': q,
                        'options': o,
                        'ground_truth': task["Correct_Answer"],
                        'chat': question,
                    })

        return flattened

    def __len__(self):
        return len(self.flattened_dataset)

    def __getitem__(self, idx):
        return self.flattened_dataset[idx]
    



class ClearGenDataset(torch.utils.data.Dataset):
    image_caption_questions = [
        "What can you see in this picture?",
        "Tell me about the content of this image",
        "Can you give a description of the image?",
        "What is depicted in the image?",
        "Explain what you observe in the picture.",
        "Describe the image in detail.",
        "What is the main subject of this image?",
        "Can you describe the scene or objects in the image?",
        "What is happening in this image?",
    ]

    def __init__(self, dataset, processor, args, data_type="train", modality: Literal["multi", "text"] = "multi"):
        self.raw_dataset = dataset
        self.processor = processor
        self.args = args
        self.data_type = data_type

        self.modality = modality
        assert self.modality in ["multi", "text"], "Modality must be either 'multi' or 'text'"


        self.processed_dataset = self._process_dataset()

    def _process_dataset(self):
        processed = []
        for i in range(len(self.raw_dataset)):
            sample = self.raw_dataset[i]
            image = sample.get('image')
            if self.modality == "text":
                if image is not None:
                    continue
                image = None
            else:
                if image is None:
                    continue
                image = image.resize((self.args.image_resize, self.args.image_resize))

            if self.modality == "multi":
                gt = sample["caption"]
                q = random.choice(self.image_caption_questions)
            else:
                gt = sample["answer"]
                q = sample["question"]

            if self.modality == "text":
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": q},
                        ]
                    }
                ]
            else:
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q},
                        ]
                    }
                ]
            question = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            if self.modality == "multi":
                processed.append({
                    'question': q,
                    'ground_truth': gt,
                    'image': image,
                    'chat': question,
                })
            else:
                processed.append({
                    'question': q,
                    'ground_truth': gt,
                    'chat': question,
                })
        return processed


    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]
    


class ClearClfDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, args, data_type="train"):
        self.raw_dataset = dataset
        self.processor = processor
        self.args = args
        self.data_type = data_type

        self.processed_dataset = self._process_dataset()

    def _process_dataset(self):
        processed = []
        for i in range(len(self.raw_dataset)):
            sample = self.raw_dataset[i]
            image = sample['image']
            image = image.resize((self.args.image_resize, self.args.image_resize))
            q = sample.get("question", "Answer with ONE LETTER direclty.")
            captions = sample['perturbed_captions']
            gt = sample["caption"]
            o = {}
            o["A"], o["B"], o["C"], o["D"], o["E"], o["F"] = captions[0], captions[1], captions[2], captions[3], captions[4], gt

            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{q} Options: A. {o['A']} B. {o['B']} C. {o['C']} D. {o['D']} E. {o['E']} F. {o['F']}."},
                    ]
                }
            ]

            question = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            processed.append({
                'question': q,
                'ground_truth': gt,
                'options': o,
                'image': image,
                'chat': question,
            })
        return processed

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        return self.processed_dataset[idx]



def collator(batch, processor, args):
    batch_data = {}
    for sample in batch:
        for key, value in sample.items():
            if key not in batch_data:
                batch_data[key] = []
            batch_data[key].append(value)
    
    batch_data['inputs'] = processor(
        text=batch_data['chat'], 
        images=batch_data.get('images') or batch_data.get('image'),
        padding=True, 
        return_tensors="pt"
    ).to(args.device)

    return batch_data
