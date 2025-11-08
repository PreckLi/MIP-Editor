from tqdm import tqdm
import transformers

import torch
import json




def load_llm(path, device):
    """
    Load a pre-trained LLM model from the specified path.
    
    Args:
        path (str): Path to the pre-trained model.
        device (str): Device to load the model on ('cpu' or 'cuda').
    
    Returns:
        transformers.PreTrainedModel: Loaded LLM model.
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=True, padding_side='left')
    return model, tokenizer


def build_conversation_for_gen_task(item: dict) -> list[dict]:
    system_prompt = """You are a helpful assistant that determines whether the prediction is correct.
Given a question with correct answer and a predicted answer, you will output a JSON object with a boolean field 'answer' indicating
whether the 'pred' answer is semantically the same (true) or different (false) as the 'truth' option.
The sentences will be provided in the fields 'input', 'question', 'truth', and 'pred'.
Please respond only with the JSON object, without any additional text or explanation."""
    data = {
        "question": item['question'],
        "truth": item['gt'],
        "pred": item['pred'][:500],
    }
    prompt = json.dumps(data, ensure_ascii=False, indent=2)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return conversation


def build_conversation_for_clf_task(item: dict) -> list[dict]:
    system_prompt = """You are a helpful assistant that determines whether the prediction is correct.
Given a multiple-choice question with correct answer and a predicted answer, you will output a JSON object with a boolean field 'answer' indicating
whether the 'pred' sentence has correctly answered the question (true) or not (false) according to the 'truth' option.
The 'pred' may be a single option or repeated characters of the same option or a complete sentence. It's okay if the 'pred' is not exactly the same as the 'truth' option, as long as it conveys the same meaning. 
But it should not be a combination of multiple options, as that would be incorrect.
The sentences will be provided in the fields 'input', 'question', 'truth', and 'options'.
Please respond only with the JSON object, without any additional text or explanation."""
    data = {
        "question": item['question'],
        "options": item['options'],
        "truth": item['gt'],
        "pred": item['pred'][:500].split(',')[0],
    }
    prompt = json.dumps(data, ensure_ascii=False, indent=2)
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return conversation


@torch.no_grad()
def score_by_llm_batch(model, tokenizer, pred_list: list[dict], batch_size: int, task: str, device: str = 'cuda'):
    res = []
    for i in tqdm(range(0, len(pred_list), batch_size)):
        batch = pred_list[i:i + batch_size]
        
        conversation_list = []
        for item in batch:
            if task == 'gen':
                conversation = build_conversation_for_gen_task(item)
            elif task == 'clf':
                conversation = build_conversation_for_clf_task(item)
            else:
                assert False, f"Unknown task type: {task}"
            conversation_list.append(conversation)

        inputs_temp = tokenizer.apply_chat_template(conversation_list, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(inputs_temp, return_tensors="pt", truncation=True, max_length=1024, padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
        )
        outputs = outputs[:, inputs['input_ids'].shape[1]:]  # Skip the input part
        responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for response, item in zip(responses, batch):
            response = response.strip()
            response = response.replace("'", '"')  # Ensure valid JSON format
            try:
                scores = json.loads(response)
                if isinstance(scores, dict) and 'answer' in scores:
                    scores = scores['answer']
                elif isinstance(scores, bool):
                    scores = scores
                else:
                    print(f"Unexpected response format: {response}")
                    scores = None
            except Exception as e:
                print(f"Error parsing response: {response}, Error: {e}")
                scores = None
            res.append({
                **item,
                'correct': scores
            })

    return res



def eval_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        preds = json_data['preds']
        args = json_data['args']
    
    batch_size = 16
    

    score_llm = None
    try:
        score_llm, tokenizer = load_llm(f"{args['llm_directory']}{args['score_llm']}", args['device'])
        print(f'scoring by llm')
        preds = score_by_llm_batch(score_llm, tokenizer, preds, batch_size, 'clf', args['device'])
        acc = sum(map(lambda r: int(r['correct']), preds)) / len(preds)
        json_data['acc_by_llm'] = acc
        json_data['preds'] = preds
        new_filepath = filepath.replace('.json', '_scored.json')
        with open(new_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"Scoring completed. acc_by_llm: {acc:.4f}, saved to {new_filepath}")
    except Exception as e:
        print(f"Error during scoring: {type(e)}  {e}")
        if score_llm is not None:
            del score_llm

        return preds, -1
