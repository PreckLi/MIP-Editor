from torch.utils.data import DataLoader, Subset
from transformers import set_seed

from mllmu_bench import MMLMU_Dataset, MMLMU_Clf_Dataset, MMLMU_Gen_Dataset
from clear import CLEAR_Dataset, CLEAR_Clf_Dataset, CLEAR_Gen_Dataset

set_seed(42)

def load_data_train(processor, args):
    if "mllmu" in args.dataset:
        trainset = MMLMU_Dataset(args.train_path, args.fullset_path, processor, args.image_resize, args.model, train=args.train_flag)
    if "clear" in args.dataset:
        trainset = CLEAR_Dataset(args.train_path, processor, args.image_resize, args.model, train=args.train_flag)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, collate_fn=trainset.collate)
    return trainset, train_loader

def load_data_forget(processor, args):
    base_path = args.base_path + "/"
    if args.dataset == "clear":
        forget_path = base_path + f"CLEAR/forget{args.forget_ratio}_perturbed"
        forgetset_clf = CLEAR_Clf_Dataset(forget_path, processor, args.image_resize, train=True)
        forget_path = base_path + f"CLEAR/forget{args.forget_ratio}+tofu"
        forgetset_gen = CLEAR_Gen_Dataset(forget_path, processor, args.image_resize, train=True)
    elif args.dataset == "mllmu":
        forget_path = base_path + f"MLLMU-Bench/forget_{args.forget_ratio}/train-00000-of-00001.parquet"
        forgetset_clf = MMLMU_Clf_Dataset(forget_path, processor, args.image_resize, train=True)
        forgetset_gen = MMLMU_Gen_Dataset(forget_path, processor, args.image_resize, train=True)
    if "mllmu" in args.dataset:
        indices = [int(forgetset_clf[i]["ID"]) - 1 for i in range(len(forgetset_clf))] 
    else:
        indices = []
    return forgetset_clf, forgetset_gen, indices

def load_data_retain(processor, args):
    base_path = args.base_path + "/"
    if args.dataset == "clear":
        retain_path = base_path + f"CLEAR/retain_perturbed"
        retainset_clf = CLEAR_Clf_Dataset(retain_path, processor, args.image_resize, train=True)
        retain_path = base_path + f"CLEAR/retain{100 - args.forget_ratio}+tofu"
        retainset_gen = CLEAR_Gen_Dataset(retain_path, processor, args.image_resize, train=True)
    elif args.dataset == "mllmu":
        retain_path = base_path + f"MLLMU-Bench/retain_{100 - args.forget_ratio}/train-00000-of-00001.parquet"
        retainset_clf = MMLMU_Clf_Dataset(retain_path, processor, args.image_resize, train=True)
        retainset_gen = MMLMU_Gen_Dataset(retain_path, processor, args.image_resize, train=True)
    if "mllmu" in args.dataset:
        indices = [int(retainset_clf[i]["ID"]) - 1 for i in range(len(retainset_clf))]
    else:
        indices = []
    return retainset_clf, retainset_gen, indices

def load_data_finetune(processor, forget_indices, retain_indices, args):
    if "mllmu" in args.dataset:
        fullset = MMLMU_Dataset(args.train_path, args.fullset_path, processor, args.image_resize, args.model, train=True)
        new_forget_indices = []
        new_retain_indices = []
        for idx, i in enumerate(fullset):
            if int(i["ID"]) - 1 in forget_indices:
                new_forget_indices.append(idx)
            elif int(i["ID"]) - 1 in retain_indices:
                new_retain_indices.append(idx)
        forgetset = Subset(fullset, new_forget_indices)
        retainset = Subset(fullset, new_retain_indices)
        batch_size = args.batch_size if args.use_neuron_cache_flag else 1
        forget_loader = DataLoader(forgetset, batch_size=batch_size, collate_fn=fullset.collate)
        retain_loader = DataLoader(retainset, batch_size=batch_size, collate_fn=fullset.collate)
        forget_text_loader = DataLoader(forgetset, batch_size=batch_size, collate_fn=fullset.collate_text_with_name)
        retain_text_loader = DataLoader(retainset, batch_size=batch_size, collate_fn=fullset.collate_text_with_name)

    if "clear" in args.dataset:
        fullset = CLEAR_Dataset(args.train_path, processor, args.image_resize, args.model, train=True)
        forgetset = CLEAR_Dataset(args.base_path + f"CLEAR/forget{args.forget_ratio}+tofu", processor, args.image_resize, args.model, train=True)
        retainset = CLEAR_Dataset(args.base_path + f"CLEAR/retain{100 - args.forget_ratio}+tofu", processor, args.image_resize, args.model, train=True)
        new_forget_indices = list(range(len(forgetset)))
        new_retain_indices = list(range(len(retainset)))
        batch_size = args.batch_size if args.use_neuron_cache_flag else 1
        forget_loader = DataLoader(forgetset, batch_size=batch_size, collate_fn=fullset.collate)
        retain_loader = DataLoader(retainset, batch_size=batch_size, collate_fn=fullset.collate)
        forget_text_loader = DataLoader(forgetset, batch_size=batch_size, collate_fn=fullset.collate_text)
        retain_text_loader = DataLoader(retainset, batch_size=batch_size, collate_fn=fullset.collate_text)
    return forgetset, forget_loader, retainset, retain_loader, forget_text_loader, retain_text_loader, new_forget_indices, new_retain_indices, fullset.collate