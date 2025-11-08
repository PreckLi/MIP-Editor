from myqwen2vl import Qwen2VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration
import torch
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


def load_base_model(args):
    model_path = args.llm_directory + f"{args.model}"
    if args.model == "Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device)
    if args.model == "Qwen2-VL-2B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device)
    if args.model == "Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device)
    if args.model == "llava-1.5-7b-hf":
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device)
    return model


def load_model(args, visual_trainable=True):
    model = load_base_model(args)
    print(model)
    if visual_trainable:
        lora_config = LoraConfig(
            # task_type=TaskType.SEQ_CLS,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "qkv", "proj", "fc2", "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"
            ],
        )
    else:
        num_layers = model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else model.config.text_config.num_hidden_layers
        pattern = r'layers'
        lora_config = LoraConfig(
            # task_type=TaskType.SEQ_CLS,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "up_proj", "gate_proj", "down_proj"
            ],
            layers_pattern=pattern,
            layers_to_transform=list(range(num_layers)),
        )
    model = get_peft_model(model, lora_config)
    print("************full peft model************")
    print(model)
    model.print_trainable_parameters()
    model.to(args.device)
    return model

def load_peft_model(args, trainable=False, identifier=""):
    model_path = args.llm_directory + f"{args.model}"
    if identifier:
        identifier = f"_{identifier}"
    peft_path = f"{args.output_file_path}model_caches/{args.model}{identifier}_{args.dataset[:5]}_batch{args.batch_size}_epochs{args.epochs}_img_resize{args.image_resize}.pth"
    print("Loading model: " + peft_path)
    if args.model == "Qwen2-VL-2B-Instruct":
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    elif args.model == "Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    elif args.model == "Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    elif args.model == "llava-1.5-7b-hf":
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    model = PeftModel.from_pretrained(model, peft_path, is_trainable=trainable)
    model.to(args.device)
    return model