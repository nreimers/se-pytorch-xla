# Pytorch TPU Sentence Embedding Model Training


## Requirements

I can recommend to use [conda](https://docs.conda.io/en/latest/miniconda.html) to manage the Python virtual enviroment. You can use [tmux](https://linuxize.com/post/getting-started-with-tmux/) to start a shell that will not stop when you disconnect.

First install conda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

Then create a new virtual env:
```
conda create -n pytorch python=3.8
conda activate pytorch
```` 

First install pytorch
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

Then install [pytorch xla](https://cloud.google.com/tpu/docs/pytorch-xla-ug-tpu-vm):
```
pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-1.9-cp38-cp38-linux_x86_64.whl
```

Then install sentence-transformers:
```
pip install sentence-transformers
```

You also need to set the TPU address:
```
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

## Evaluation

See [https://github.com/nreimers/se-benchmark](https://github.com/nreimers/se-benchmark) how to evaluate models.


## Multi-Dataset File Training

You need a json file that specifies which dataset.jsonl.gz to load. See `train_data_configs` for an example.

Run the code with:
```
python train_many_data_file_v2s.py train_data_configs/small_multi_dataset_train.json output/your_model
```


## Single Dataset File Training
When you want to train on a single data file, you can use `train_single_data_file.py`:
```
python train_single_data_file.py data/your_datafile.jsonl.gz output/your_model
```

Check the other parameters in train_single_data_file.py to modify the model, the batch size, and the number of training steps.


## Base Model & Batch sizes

In the following table I try to collect the max batch sizes per device for different, recommended models. So far I just tested with these values and the models were running with out-of-memory-exceptions. I.e., larger batch sizes are potentially possible for these models. 

| Model | Max Batch size per Device | Commment |
| --- | --- | ---- |
| [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) | 128 is ok | A small and fast model |
| distilroberta-base | | |
| [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) | 64 is ok | Works usually better than bert-base and roberta-base |