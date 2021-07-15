"""
Train script for a single file

Need to set the TPU address first:
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
"""

import torch.multiprocessing as mp
import threading
import time
import random
import sys
import argparse
import gzip
import json
import logging
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch
import torch_xla
import torch_xla.core
import torch_xla.core.functions
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import os
from shutil import copyfile
from pathlib import Path


from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True, pooling='mean'):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer
        self.pooling = pooling

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        if self.pooling == 'mean':
            embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        elif self.pooling == 'cls':
            embeddings = self.cls_pooling(model_output, kwargs['attention_mask'])

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:,0]

    def save_pretrained(self, output_path):
        if xm.is_master_ordinal():
            self.tokenizer.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)

        xm.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
       



def train_function(index, args, queue):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    modelQ = AutoModelForSentenceEmbedding(args.model, tokenizer, not args.no_normalize, args.pooling)
    modelA = AutoModelForSentenceEmbedding(args.model, tokenizer, not args.no_normalize, args.pooling)
  
    ### Train Loop
    device = xm.xla_device()
    modelQ = modelQ.to(device)
    modelA = modelA.to(device)

    # Instantiate optimizer
    optimizerQ = AdamW(params=modelQ.parameters(), lr=2e-5, correct_bias=True)
    optimizerA = AdamW(params=modelA.parameters(), lr=2e-5, correct_bias=True)

    lr_schedulerQ = get_linear_schedule_with_warmup(
        optimizer=optimizerQ,
        num_warmup_steps=500,
        num_training_steps=args.steps,
    )
    lr_schedulerA = get_linear_schedule_with_warmup(
        optimizer=optimizerA,
        num_warmup_steps=500,
        num_training_steps=args.steps,
    )
    
    # Now we train the model
    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1

    modelQ.train()
    modelA.train()
   
    for global_step in tqdm.trange(args.steps, disable=not xm.is_master_ordinal()):
        #### Get the batch data
        batch = queue.get()
        #print(index, "batch {}x{}".format(len(batch), ",".join([str(len(b)) for b in batch])))
        

        if len(batch[0]) == 2: #(anchor, positive)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_q_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_a_length, truncation=True, padding="max_length")

            ### Compute embeddings
            embeddings_a = modelQ(**text1.to(device))
            embeddings_b = modelA(**text2.to(device))
            
            ### Gather all embedings 
            embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
            embeddings_b = torch_xla.core.functions.all_gather(embeddings_b)

            ### Compute similarity scores 512 x 512
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## Symmetric loss as in CLIP
            loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(0, 1), labels)) / 2

        else:   #(anchor, positive, negative)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_q_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_a_length, truncation=True, padding="max_length")
            text3 = tokenizer([b[2] for b in batch], return_tensors="pt", max_length=args.max_a_length, truncation=True, padding="max_length")

            embeddings_a  = modelQ(**text1.to(device))
            embeddings_b1 = modelA(**text2.to(device))
            embeddings_b2 = modelA(**text3.to(device))

            embeddings_a  = torch_xla.core.functions.all_gather(embeddings_a)
            embeddings_b1 = torch_xla.core.functions.all_gather(embeddings_b1)
            embeddings_b2 = torch_xla.core.functions.all_gather(embeddings_b2)

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

            ### Compute similarity scores 512 x 1024
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## One-way loss
            loss = cross_entropy_loss(scores, labels)

        
        # Backward pass
        optimizerQ.zero_grad()
        optimizerA.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelQ.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(modelA.parameters(), max_grad_norm)
        
        xm.optimizer_step(optimizerQ, barrier=True)
        xm.optimizer_step(optimizerA, barrier=True)
        lr_schedulerQ.step()
        lr_schedulerA.step()


        #Save model
        if (global_step+1) % args.save_steps == 0:
            output_pathQ = os.path.join(args.output + "Q", str(global_step+1))
            xm.master_print("save modelQ : "+ output_pathQ)
            modelQ.save_pretrained(output_pathQ)
            output_pathA = os.path.join(args.output + "A", str(global_step+1))
            xm.master_print("save modelA : "+ output_pathQ)
            modelA.save_pretrained(output_pathQ)
          
            
    output_pathQ = os.path.join(args.output + "Q", "final")
    xm.master_print("save modelQ final: "+ output_pathQ)
    modelQ.save_pretrained(output_pathQ)

    output_pathA = os.path.join(args.output + "A", "final")
    xm.master_print("save modelA final: "+ output_pathA)
    modelA.save_pretrained(output_pathA)


def produce_data(args, queue, filepaths, dataset_indices):
    global_batch_size = args.batch_size*args.nprocs    #Global batch size
    size_per_dataset = int(global_batch_size / args.datasets_per_batch)    #How many datasets per batch
    num_same_dataset = int(size_per_dataset / args.batch_size)
    print("producer", "global_batch_size", global_batch_size)
    print("producer", "size_per_dataset", size_per_dataset)
    print("producer", "num_same_dataset", num_same_dataset)
    
    datasets = []
    for filepath in filepaths:
        if "reddit_" in filepath:       #Special dataset class for Reddit files
            data_obj = RedditDataset(filepath)
        else:
            data_obj = Dataset(filepath)
        datasets.append(iter(data_obj)) 
    
    # Store if dataset is in a 2 col or 3 col format
    num_cols = {idx: len(next(dataset)) for idx, dataset in enumerate(datasets)}

    while True:
        texts_in_batch = set()
        batch_format = None     #2 vs 3 col format for this batch
        
        #Add data from several sub datasets
        for _ in range(args.datasets_per_batch):
            valid_dataset = False   #Check that datasets have the same 2/3 col format
            while not valid_dataset:
                data_idx = random.choice(dataset_indices)
                if batch_format is None:
                    batch_format = num_cols[data_idx]
                    valid_dataset = True
                else:   #Check that this dataset has the same format
                    valid_dataset = (batch_format == num_cols[data_idx])
            
            #Get data from this dataset
            dataset = datasets[data_idx]
            for _ in range(num_same_dataset):
                for _ in range(args.nprocs):
                    batch_device = []   #A batch for one device
                    while len(batch_device) < args.batch_size:
                        sample = next(dataset)
                        in_batch = False
                        for text in sample:
                            if text in texts_in_batch:
                                in_batch = True
                                break
                        
                        if not in_batch:
                            for text in sample:
                                texts_in_batch.add(text)
                            batch_device.append(sample)

                    queue.put(batch_device)
                      

class RedditDataset:
    """
    A class that handles the reddit data files
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        while True:
            with gzip.open(self.filepath, "rt") as fIn:
                    for line in fIn:
                        data = json.loads(line)

                        if "response" in data and "context" in data:
                            yield [data["response"], data["context"]]

class Dataset:
    """
    A class that handles one dataset
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        max_dataset_size = 10*1000*1000    #Cache small datasets in memory
        dataset = []
        data_format = None

        while dataset is None or len(dataset) == 0:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        data = data['texts']

                    if data_format is None:
                        data_format = len(data)
                    
                    #Ensure that all entries are of the same 2/3 col format
                    assert len(data) == data_format

                    if dataset is not None:
                        dataset.append(data)
                        if len(dataset) >= max_dataset_size:
                            dataset = None

                    yield data
                
        # Data loaded. Now stream to the queue
        # Shuffle for each epoch
        while True:
            random.shuffle(dataset)
            for data in dataset:
                yield data
                
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nreimers/MiniLM-L6-H384-uncased')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_q_length', type=int, default=128)
    parser.add_argument('--max_a_length', type=int, default=256)
    parser.add_argument('--nprocs', type=int, default=8)
    parser.add_argument('--datasets_per_batch', type=int, default=2, help="Number of datasets per batch")
    parser.add_argument('--scale', type=float, default=20, help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")
    parser.add_argument('--data_folder', default="./data", help="Folder with your dataset files")
    parser.add_argument('--no_normalize', action="store_true", default=False, help="If set: Embeddings are not normalized")
    parser.add_argument('--pooling', default='mean')
    parser.add_argument('--stack_overflow_folder', default="./data/stackexchange_title_best_voted_answer_jsonl")
    parser.add_argument('--stack_overflow_weight', type=int, default="360")
    parser.add_argument('--stack_overflow_body_folder', default="./data/stackexchange_titlebody_best_voted_answer_jsonl")
    parser.add_argument('--stack_overflow_body_weight', type=int, default="180")
    parser.add_argument('--stack_overflow_neg_folder', default="./data/stackexchange_titlebody_best_and_down_voted_answer_jsonl")
    parser.add_argument('--stack_overflow_neg_weight', type=int, default="36")
    parser.add_argument('data_config', help="A data_config.json file")
    parser.add_argument('output')
    args = parser.parse_args()

    # Ensure global batch size is divisble by data_sample_size
    assert (args.batch_size*args.nprocs) % args.datasets_per_batch == 0

    logging.info("Output: "+args.output)
    if os.path.exists(args.output):
        print("Output folder already exists.")
        input("Continue?")

    # Write train script to output path
    os.makedirs(args.output, exist_ok=True)

    data_config_path = os.path.join(args.output, 'data_config.json')
    copyfile(args.data_config, data_config_path)

    train_script_path = os.path.join(args.output, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))



    #Load data config
    with open(args.data_config) as fIn:
        data_config = json.load(fIn)

    queue = mp.Queue(maxsize=100*args.nprocs)
    
    filepaths = []
    dataset_indices = []
    for data in data_config:
        weight = data['weight']
        if weight == 0:
            continue
        filepaths.append(os.path.join(os.path.expanduser(args.data_folder), data['name']))
        dataset_indices.extend([len(filepaths) - 1]*weight)

    so_list = Path(args.stack_overflow_folder).rglob('*.gz')

    total_weight = args.stack_overflow_weight
    if total_weight != 0:
        from subprocess import check_output
        file_length = {}

        total_length = 0
        for f in so_list:
            path = f.absolute().as_posix()
            length = int(check_output(['wc', '-l', path]).split()[0])
            file_length[path] = length
            total_length += length
        print(total_length)
        print(file_length)

        for path in file_length.keys():
            filepaths.append(path)
            so_weight = int((file_length[path] / total_length) * total_weight)
            dataset_indices.extend([len(filepaths) - 1] * so_weight)
            print("{} : {}".format(path, so_weight))

    so_neg_list = Path(args.stack_overflow_neg_folder).rglob('*.gz')

    total_neg_weight = args.stack_overflow_neg_weight
    if total_neg_weight != 0:
        from subprocess import check_output
        file_length = {}

        total_length = 0
        for f in so_neg_list:
            path = f.absolute().as_posix()
            length = int(check_output(['wc', '-l', path]).split()[0])
            file_length[path] = length
            total_length += length
        print(total_length)
        print(file_length)

        for path in file_length.keys():
            filepaths.append(path)
            so_weight = int((file_length[path] / total_length) * total_weight)
            dataset_indices.extend([len(filepaths) - 1] * so_weight)
            print("{} : {}".format(path, so_weight))

    # Start producer
    p = mp.Process(target=produce_data, args=(args, queue, filepaths, dataset_indices))
    p.start()

    # Run training
    print("Start processes:", args.nprocs)
    xmp.spawn(train_function, args=(args, queue), nprocs=args.nprocs, start_method='fork')
    print("Training done")
    print("It might be that not all processes exit automatically. In that case you must manually kill this process.")
    print("With 'pkill python' you can kill all remaining python processes")
    p.kill()
    exit()

