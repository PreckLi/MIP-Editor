import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import json

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

class MMLMU_Clf_Dataset(Dataset):
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
        dataset = pd.read_parquet(self.data_path)
        return {idx: dataset.iloc[idx] for idx in range(dataset.shape[0])}

    def collate(self, batch):
        batch_input_ids, batch_pixel_values, batch_grid_thw, batch_labels = [], [], [], []
        item_list = []
        for item in batch:
            item_list.append(item)
            name = json.loads(item["biography"])["Name"]
            tc = item["Classification_Task"]['Image_Textual_Questions']
            all_input_ids = []
            all_labels = []
            if self.train:
                image_bytes = item['image'].get('bytes')
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((self.image_resize, self.image_resize))        
            for i in range(len(tc)):
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": tc[i]["Question"] + " Options:" + " A:" + tc[i]["Options"]["A"] + " B:" + tc[i]["Options"]["B"] + " C:" + tc[i]["Options"]["C"] + " D:" + tc[i]["Options"]["D"]},
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                label = f"{tc[i]['Correct_Answer']}\n{define_special_tokens['DEFAULT_IM_END_TOKEN']}\n"
                if i == 0:
                    inputs = self.processor(text=[text], images=image, padding=False, return_tensors="pt")
                    input_ids_o, images, grid_thw = inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"]
                else:
                    text = text.replace(define_special_tokens['DEFAULT_IMAGE_TOKEN'], "")
                    input_ids_o = self.processor.tokenizer(text, padding=False, return_tensors="pt")["input_ids"]
                labels = self.processor.tokenizer(label, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]

                if self.train:
                    input_ids = torch.cat([input_ids_o, labels], dim=1).squeeze(0)
                    labels = torch.cat([
                        torch.full_like(input_ids_o[0], IGNORE_INDEX),
                        labels.squeeze(0),
                    ])
                else:
                    input_ids = input_ids_o.squeeze(0)
                    labels = labels.squeeze(0)
                all_input_ids.append(input_ids)
                all_labels.append(labels)
            input_ids = torch.cat(all_input_ids, dim=0)
            labels = torch.cat(all_labels, dim=0)
            batch_input_ids.append(input_ids)
            batch_pixel_values.append(images)
            batch_grid_thw.append(grid_thw)
            batch_labels.append(labels)

        input_ids = pad_sequence(batch_input_ids, padding_side='right', padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = input_ids != self.processor.tokenizer.pad_token_id
        labels = pad_sequence(batch_labels, padding_side='right', padding_value=IGNORE_INDEX)
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        grid_thw = torch.cat(batch_grid_thw, dim=0)

        return input_ids, attention_mask, pixel_values, grid_thw, labels, item_list

class MMLMU_Gen_Dataset(Dataset):
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
        dataset = pd.read_parquet(self.data_path)
        return {idx: dataset.iloc[idx] for idx in range(dataset.shape[0])}

    def collate(self, batch):
        batch_input_ids, batch_pixel_values, batch_grid_thw, batch_labels = [], [], [], []
        item_list = []
        for item in batch:
            item_list.append(item)
            questions = item['question']
            answer = item['answer']
            if self.train:
                image_bytes = item['image'].get('bytes')
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                image = image.resize((self.image_resize, self.image_resize))        
            conversation = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image"},
                        # {"type": "text", "text": f"Biography: {biography} Name: {name} Questions: {questions}"},
                        {"type": "text", "text": f"Questions: {questions}"},
                    ]
                }
            ]
            text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            label = f"{answer}\n{define_special_tokens['DEFAULT_IM_END_TOKEN']}\n"
            inputs = self.processor(text=[text], images=image, padding=False, return_tensors="pt", truncation=True, max_length=1024)
            input_ids_o, images, grid_thw = inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"]
            label = self.processor.tokenizer(label, add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"]
            input_ids = torch.cat([input_ids_o, label], dim=1).squeeze(0)
            label = torch.cat([
                torch.full_like(input_ids_o.squeeze(0), IGNORE_INDEX),
                label.squeeze(0),
            ])
            
            batch_input_ids.append(input_ids.squeeze(0))
            batch_pixel_values.append(images)
            batch_grid_thw.append(grid_thw)
            batch_labels.append(label.squeeze(0))

        input_ids = pad_sequence(batch_input_ids, padding_side='right', padding_value=self.processor.tokenizer.pad_token_id)
        attention_mask = input_ids != self.processor.tokenizer.pad_token_id
        labels = pad_sequence(batch_labels, padding_side='right', padding_value=IGNORE_INDEX)
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        grid_thw = torch.cat(batch_grid_thw, dim=0)
        return input_ids, attention_mask, pixel_values, grid_thw, labels, item_list


class MMLMU_Dataset(Dataset):
    def __init__(self, data_path, fullset_path, processor, image_resize, model_name, train=True):
        self.processor = processor
        self.data_path = data_path
        self.train = train
        self.image_resize = image_resize
        self.model_name = model_name
        self.fullset = pd.read_parquet(fullset_path)  
        self.df = pd.read_parquet(self.data_path)
        self.data = self.flatten_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []
        for (idx, row), bios in zip(self.df.iterrows(), self.fullset.iterrows()):
            # Extract the bytes from the 'image' dictionary
            image_data = row['image'].get('bytes')  # Access the image bytes

            # Convert the image bytes to a PIL Image
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            # Safely load metadata as JSON
            try:
                metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
                name = json.loads(bios[1]['biography'])["Name"]
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue
            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")
                if "person" in question:
                    question = question.replace("person", name)
                if "individual" in question:
                    question = question.replace("individual", name)
                if name in question:
                    pass
                elif "person" not in question and "individual" not in question and name not in question:
                    question = question + " " + name
                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer,
                        "ID": row["ID"],
                        "name": name
                    })
        return flattened_data

    def collate(self, batch):
        if "gemma" in self.model_name:
            messages = []
            images = []
            item_list = []
            for item in batch:
                # if self.train:
                image = item['image']
                image = image.resize((self.image_resize, self.image_resize))        
                message = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an assistant with great geometry skills.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": item["question"],
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]},
                ]
                message = self.processor.apply_chat_template(
                    message,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                messages.append(message)
                images.append([image])

            inputs = self.processor(text=messages, images=images, return_tensors="pt", padding=True)
            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == self.processor.tokenizer.image_token_id] = -100
            labels[labels == self.processor.tokenizer.boi_token_id] = -100
            labels[labels == self.processor.tokenizer.eoi_token_id] = -100
            inputs["labels"] = labels
            return inputs, item_list
        else:
            images, texts = [], []
            item_list = []
            for item in batch:
                if self.train:
                    image = item['image']
                    image = image.resize((self.image_resize, self.image_resize))        
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": item["question"]},]
                    },
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "text", "text": item["answer"]},]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                texts.append(text)
                images.append(image)

            batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch["labels"] = labels
            if "llava" in self.model_name or "gemma" in self.model_name:
                return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"], item_list
            if "Qwen" in self.model_name:
                return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["image_grid_thw"], batch["labels"], item_list
        
    
    def collate_text_with_name(self, batch):
        if "gemma" in self.model_name:
            messages = []
            item_list = []
            for item in batch:       
                message = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an assistant with great geometry skills.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": item["question"].replace("this person", item["name"]),
                            },
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": item["answer"]}]},
                ]
                message = self.processor.apply_chat_template(
                    message,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                messages.append(message)

            inputs = self.processor(text=messages, return_tensors="pt", padding=True)
            labels = inputs["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            labels[labels == self.processor.tokenizer.image_token_id] = -100
            labels[labels == self.processor.tokenizer.boi_token_id] = -100
            labels[labels == self.processor.tokenizer.eoi_token_id] = -100
            inputs["labels"] = labels
            return inputs, item_list
        else:
            texts = []
            item_list = []
            for item in batch:
                conversation = [
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": item["question"].replace("this person", item["name"])},]
                    },
                    {
                        "role": "assistant", 
                        "content": [
                            {"type": "text", "text": item["answer"]},]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                texts.append(text)

            batch = self.processor(text=texts, padding=True, return_tensors="pt")
            labels = batch["input_ids"].clone()
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch["labels"] = labels
            return batch["input_ids"], batch["attention_mask"], batch["labels"], item_list