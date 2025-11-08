import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import json
from datasets import load_dataset
import random

# Special tokens
define_special_tokens = {
    "DEFAULT_IM_START_TOKEN": "<|im_start|>",
    "DEFAULT_IM_END_TOKEN": "<|im_end|>",
    "DEFAULT_IMAGE_TOKEN": "<|image_pad|>",
    "DEFAULT_VIDEO_TOKEN": "<|video_pad|>",
    "LLAVA_IMAGE_TOKEN": "<image>",
    "LLAVA_VIDEO_TOKEN": "<video>",
    "VISION_START_TOKEN": "<|vision_start|>",
    "VISION_END_TOKEN": "<|vision_end|>",
}

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


IGNORE_INDEX = -100

@staticmethod
def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + max_size[1:], padding_value)

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output[i, :length] = seq
        else:
            output[i, -length:] = seq
    return output


class CLEAR_Dataset(Dataset):
    def __init__(self, data_path, processor, image_resize, model_name, train=True):
        self.processor = processor
        self.data_path = data_path
        self.train = train
        self.image_resize = image_resize
        self.fullset = load_dataset(data_path)
        self.model_name = model_name

    def __len__(self):
        return len(self.fullset["train"])

    def __getitem__(self, idx):
        return self.fullset["train"][idx]
    
    def collate(self, batch):
        item_list = []
        chats = []
        images = []

        start_by_multimodal_data = False
        for item in batch:
            image = item.get("image", None)
            if image:
                image = image.resize((self.image_resize, self.image_resize))
                start_by_multimodal_data = True
            else: # 没图片，表示是纯文本模态数据
                if start_by_multimodal_data == True:
                    continue # 丢掉交界处的纯文本模态数据
            question = item.get("question","")
            if question == None:
                question = random.choice(image_caption_questions)
            caption = item.get("caption","")
            answer = item.get("answer", None)
            if answer == None:
                answer = caption
            caption = item.get("caption","")

            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": answer},]
                }
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            chats.append(text)
            if image:
                images.append([image])
            else:
                images.append([Image.new('RGB', (self.image_resize, self.image_resize), 0)])


        if any(images):
            batch = self.processor(text=chats, images=images, padding=True, return_tensors="pt")
        else:
            batch = self.processor(text=chats, padding=True, return_tensors="pt")
            batch["pixel_values"], batch["image_grid_thw"] = None, None
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch["labels"] = labels
        # return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_grid_thw"], batch["labels"], item_list
        if "llava" in self.model_name:
            return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"], item_list
        if "Qwen" in self.model_name:
            return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_grid_thw"], batch["labels"], item_list


    def collate_text(self, batch):
        texts = []
        item_list = []
        for item in batch:
            question = item.get("question","")
            if question == None:
                question = random.choice(image_caption_questions)
            caption = item.get("caption","")
            answer = item.get("answer", None)
            if answer == None:
                answer = caption
            caption = item.get("caption","")
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": question},]
                },
                {
                    "role": "assistant", 
                    "content": [
                        {"type": "text", "text": answer},]
                }
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        batch = self.processor(text=texts, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        batch["labels"] = labels
        return batch["input_ids"], batch["attention_mask"], batch["labels"], item_list

class CLEAR_Clf_Dataset(Dataset):
    def __init__(self, data_path, processor, image_resize, train=True):
        self.processor = processor
        self.data_path = data_path
        self.train = train
        self.image_resize = image_resize
        self.data = self.load_data()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        dataset = load_dataset(self.data_path, split="train")
        return {idx: dataset[idx] for idx in range(dataset.shape[0])}

class CLEAR_Gen_Dataset(Dataset):
    def __init__(self, data_path, processor, image_resize, train=True):
        self.processor = processor
        self.data_path = data_path
        self.train = train
        self.image_resize = image_resize
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        dataset = load_dataset(self.data_path, split="train")
        return {idx: dataset[idx] for idx in range(dataset.shape[0])}