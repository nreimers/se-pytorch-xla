"""
Train script for a single file

Need to set the TPU address first:
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
"""

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
       



def train_function(index, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSentenceEmbedding(args.model, tokenizer)
    
    ## Load train data
    dataset = []
    with gzip.open(args.data, 'rt', encoding='utf8') as fIn:
        for line in fIn:
            data = json.loads(line.strip())

            if not isinstance(data, dict):
                data = {'guid': None, 'texts': data}

            dataset.append(InputExample(guid=data.get('guid', None), texts=data['texts']))
            if len(dataset) >= (args.steps * 8 * args.batch_size):
                break


    def collate_fn(batch):
        # On TPU it's best to pad everything to the same length or training will be very slow.
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = tokenizer(texts[idx], return_tensors="pt", max_length=128, truncation=True, padding="max_length") #padding=True, pad_to_multiple_of=128)
            sentence_features.append(tokenized)

        return sentence_features

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler, drop_last=True)
 
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
    para_train_loader = pl.ParallelLoader(train_dataloader, [device]).per_device_loader(device)
    for global_step in tqdm.trange(args.steps, disable=not xm.is_master_ordinal()):
        try:
            batch = next(para_train_loader)
        except StopIteration:
            para_train_loader = pl.ParallelLoader(train_dataloader, [device]).per_device_loader(device)
            batch = next(para_train_loader)
     
        
        if len(batch) == 2:
            text1, text2 = batch
            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))
            
            embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
            embeddings_b = torch_xla.core.functions.all_gather(embeddings_b)
        else:
            text1, text2, text3 = batch
            embeddings_a = model(**text1.to(device))
            embeddings_b1 = model(**text2.to(device))
            embeddings_b2 = model(**text3.to(device))

            embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
            embeddings_b1 = torch_xla.core.functions.all_gather(embeddings_b1)
            embeddings_b2 = torch_xla.core.functions.all_gather(embeddings_b2)

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

        """
        if False and xm.is_master_ordinal():
            for text in batch:
                print(xm.get_ordinal(), tokenizer.convert_ids_to_tokens(text['input_ids'][0])[0:20])

        reps = [model(**text.to(device)) for text in batch]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        #xm.master_print(embeddings_b.shape)

        embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
        embeddings_b = torch_xla.core.functions.all_gather(embeddings_b)
        #xm.master_print(embeddings_b.shape)
        """

        scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
        #xm.master_print(scores.shape)

        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
        loss = cross_entropy_loss(scores, labels)
        #xm.master_print(labels)

        ## CLIP loss
        #loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(x, 0, 1), labels)) / 2
        
        # Computes loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        xm.optimizer_step(optimizer)
        lr_scheduler.step()

        #Save model
        if (global_step+1) % args.save_steps == 0:
            output_path = os.path.join(args.output, str(global_step+1))
            xm.master_print("save model: "+output_path)
            model.save_pretrained(output_path)
          
            
    output_path = os.path.join(args.output)
    xm.master_print("save model final: "+ output_path)
    model.save_pretrained(output_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nreimers/MiniLM-L6-H384-uncased')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nprocs', type=int, default=8)
    parser.add_argument('--scale', type=float, default=20)
    parser.add_argument('data')
    parser.add_argument('output')
    args = parser.parse_args()

    print("Start processes:", args.nprocs)
    xmp.spawn(train_function, args=(args,), nprocs=args.nprocs, start_method='fork')
