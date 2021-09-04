# P-INT
P-INT: A Path-based Interaction Model for Few-shot Knowledge Graph Completion

## Requirements
* ``Python 3.7.9 ``
* ``PyTorch 1.7.0``

## Datasets
We conduct our experiments on two datasets — NELL-One and FB15k237-One. 

You can find original datasets(NELL-One) from [here](https://github.com/xwhan/One-shot-Relational-Learning).

You can download datasets used in this work from [here](https://drive.google.com/drive/folders/16pamNJ-8gDPC2qaObN0pr93xeqdzq4Sq?usp=sharing).

### Pre-trained embeddings
* [NELL-One](https://drive.google.com/file/d/1XXvYpTSTyCnN-PBdUkWBXwXBI99Chbps/view?usp=sharing)
* FB15k237-One   (We use this [repository](https://github.com/thunlp/OpenKE) to get the embeddings used in this work.)

## How to run

### Training
* For NELL-One: ``python train.py --dataset "NELL-One" --few n --max_batches 200000``
* For FB15k237-One: ``python train.py --dataset "FB15k237-One" --few n --max_batches 10000``

In this work, we set n=1 for one-shot and n=5 for five-shot. 

You can set ``--max_batches`` smaller.
After training, the checkpoints will be saved in `./models` and the corresponding results will be printed.

### Test
* For NELL-One: ``python train.py --dataset "NELL-One" --few n --test``
* For FB15k237-One: ``python train.py --dataset "FB15k237-One" --few n --test``

