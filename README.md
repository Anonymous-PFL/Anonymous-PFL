# Code Appendix
Code for federated learning (FL).

## Requirements to run the code:

Python 3.6-3.10

Numpy

pytorch

cvxopt

## Basic usage
Copy one line of commands in the same folder of `./pfedplat` and run (one example shown as follows).

```
python run.py --seed 1 --device 0 --model CNN_CIFAR10_FedFV --algorithm FedAvg --dataloader DataLoader_cifar10_pat --N 100 --NC 2 --balance True --B 50 --C 1.0 --R 3000 --E 1 --lr 0.1 --decay 0.999
```

All parameters can be seen in `./pfedplat/main.py`.

By setting different parameters and run the command, you can replicate results of all experiments.

Enjoy yourself!

Paper Hash code:
63E7E3D1590F0819B2AC8BBB41E4B68D

Appendix Hash code:
B3820D4F63FADCF994F00633AC906616
