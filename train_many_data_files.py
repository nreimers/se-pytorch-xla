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
from MultiDatasetDataLoader import InputExample, LoggingHandler
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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


from transformers import (
    AdamW,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save_pretrained(self, output_path):
        if xm.is_master_ordinal():
            self.tokenizer.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)

        xm.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
       



def train_function(index, args, queues, weights):
    if args.same_datasets_on_devices:
        rnd = random.Random(42)     #Same datasets in the processes
    else:
        rnd = random.Random(index)  #Different datasets in the processes
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSentenceEmbedding(args.model, tokenizer)
    
  
    ### Train Loop
    device = xm.xla_device()
    model = model.to(device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.steps,
    )
    
    # Now we train the model
    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1

    model.train()
   
    for global_step in tqdm.trange(args.steps, disable=not xm.is_master_ordinal()):
        
        #### Get the batch data
        dataset_idx = rnd.choice(weights)
        text1 = []
        text2 = []
        for _ in range(args.batch_size):
            example = queues[dataset_idx].get()
            text1.append(example[0])
            text2.append(example[1])

        #print(index, f"dataset {dataset_idx}", text1[0:3])

        text1 = tokenizer(text1, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        text2 = tokenizer(text2, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

        ### Compute embeddings
        #print(index, "compute embeddings")
        embeddings_a = model(**text1.to(device))
        embeddings_b = model(**text2.to(device))
        

        ### Gather all embedings 
        embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
        embeddings_b = torch_xla.core.functions.all_gather(embeddings_b)

        ### Compute similarity scores
        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
     
        ### Compute cross-entropy loss
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
        loss = cross_entropy_loss(scores, labels)
      
        ## CLIP loss
        #loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(x, 0, 1), labels)) / 2
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        xm.optimizer_step(optimizer, barrier=True)
        lr_scheduler.step()

        #Save model
        if (global_step+1) % args.save_steps == 0:
            output_path = os.path.join(args.output, str(global_step+1))
            xm.master_print("save model: "+output_path)
            model.save_pretrained(output_path)
          
            
    output_path = os.path.join(args.output)
    xm.master_print("save model final: "+ output_path)
    model.save_pretrained(output_path)



def load_data(path, queue):
    dataset = []

    with gzip.open(path, "rt") as fIn:
        for line in fIn:
            data = json.loads(line)
            if isinstance(data, dict):
                data = data['texts']
            
            #Only use two columns
            dataset.append(data[0:2])
            queue.put(data[0:2])

    # Data loaded. Now stream to the queue
    # Shuffle for each epoch
    while True:
        random.shuffle(dataset)
        for data in dataset:
            queue.put(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nreimers/MiniLM-L6-H384-uncased')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nprocs', type=int, default=8)
    parser.add_argument('--scale', type=float, default=20)
    parser.add_argument('--data_folder', default="~/data", help="Folder with your dataset files")
    parser.add_argument("--same_datasets_on_devices", action="store_true", help="If set, all devices will sample data from the same dataset")
    parser.add_argument('data_config', help="A data_config.json file")
    parser.add_argument('output')
    args = parser.parse_args()


    #Load data config
    with open(args.data_config) as fIn:
        data_config = json.load(fIn)

    threads = []
    queues = []
    weights = []
    for data in data_config:
        data_idx = len(queues)
        queue = mp.Queue(maxsize=args.nprocs*args.batch_size)
        th = threading.Thread(target=load_data, daemon=True, args=(os.path.join(os.path.expanduser(args.data_folder), data['name']), queue))
        th.start()
        threads.append(th)
        queues.append(queue)
        weights.extend([data_idx]*data['weight'])


    print("Start processes:", args.nprocs)
    xmp.spawn(train_function, args=(args, queues, weights), nprocs=args.nprocs, start_method='fork')
    print("Training done")

