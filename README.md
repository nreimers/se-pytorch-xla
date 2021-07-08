# Pytorch TPU Sentence Embedding Model Training

## Requirements

Run these commands to setup the enviroment:

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

To evalaute your model on the STS benchmark test set, run:

```
python eval.py output/your_model
```

## Single Dataset File Training
When you want to train on a single data file, you can use `train_single_data_file.py`:
```
python train_single_data_file.py data/your_datafile.jsonl.gz output/your_model
```

Check the other parameters in train_single_data_file.py to modify the model, the batch size, and the number of training steps.


## Multi-Dataset File Training

You need a json file that specifies which dataset.jsonl.gz to load. See `traon_data_configs` for an example.

Run the code with:
```
python train_many_data_files.py train_data_configs/small_multi_dataset_train.json output/yor_model
```
