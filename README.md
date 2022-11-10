# Transfer Learning Enhanced Water-Enabled Electricity Generation in Highly Oriented Graphene Oxide Nanochannels
Code release for "Transfer learning enhanced water-enabled electricity generation in highly oriented graphene oxide nanochannels"

## Prerequisites
- PyTorch >= 1.10.0 (with suitable CUDA and CuDNN version)

- Python3

- Numpy

- Pandas

- SciPy

- argparse

- Matplotlib

    

## Training

Train the source model:

```python
python3 main.py --mode=train --gpu_id=-1 --model=source --n=5 --lr=1e-4 --batch_size=16 --epoch=500 --loss_interval=1000 --valid_interval=1000 --add_noise --noise_std=5e-3 --source_layers=[[16,16],[64,64],[64,128],[64,128],[16,64],[2,2]]
```

Train the opt-model:

```python
python3 main.py --mode=train --gpu_id=-1 --model=target --n=5 --lr=1e-4 --norm_lr=1e-4 --finetune_lr 1e-6 1e-6 3e-6 5e-6 --batch_size=16 --epoches=3000 --loss_interval=100 --valid_interval=1 --trans_num=4 --new_layers=[[32,128],[16,32],[1,1]] --load --noise_std=5e-3
```



## Prediction and Optimization

Prediction of power generation performance for given characteristic parameters:

```python
python3 main.py --mode=test --model=target --gpu_id=-1 --n=5
```

Explore candidate optimal parameter combinations:

```python
python3 main.py --mode=optimize --model=target --gpu_id=-1 --n=5 --T 0 100 --RH 0 100 --P 0 1000 --tau 1 4 --d 10 1000 --l 0 22 --Zeta -80 0 --C 0 1 --optim_algo=DE --DE_mutation=1.0 --DE_recombination=0.7 --DE_maxiter=100000 --DE_popsize=10000 --DE_strategy=randtobest1bin
```



## Citation

If you use this code for your research, please consider citing: Transfer learning enhanced water-enabled electricity generation in highly oriented two-dimensional graphene oxide nanochannels.