#!/bin/bash

# Proto Net experiments
python -m experiments.proto_nets --dataset omniglot --k-test 5 --n-test 1
python -m experiments.proto_nets --dataset omniglot --k-test 5 --n-test 5
python -m experiments.proto_nets --dataset omniglot --k-test 20 --n-test 1
python -m experiments.proto_nets --dataset omniglot --k-test 20 --n-test 5 --n-train 5

python -m experiments.proto_nets --dataset miniImageNet --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15
python -m experiments.proto_nets --dataset miniImageNet --k-test 5 --n-test 5 --k-train 20 --n-train 5 --q-train 15

# Matching Network experiments
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 5 --n-test 1 --distance l2
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 5 --n-test 5 --distance l2
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 20 --n-test 1 --distance l2
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 20 --n-test 5 --distance l2

python -m experiments.matching_nets --dataset omniglot --fce False --k-test 5 --n-test 1 --distance cosine
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 5 --n-test 5 --distance cosine
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 20 --n-test 1 --distance cosine
python -m experiments.matching_nets --dataset omniglot --fce False --k-test 20 --n-test 5 --distance cosine

python -m experiments.matching_nets --dataset miniImageNet --fce False --k-test 5 --n-test 1 --distance l2
python -m experiments.matching_nets --dataset miniImageNet --fce False --k-test 5 --n-test 5 --distance l2
python -m experiments.matching_nets --dataset miniImageNet --fce True --k-test 5 --n-test 1 --distance l2
python -m experiments.matching_nets --dataset miniImageNet --fce True --k-test 5 --n-test 5 --n-train 5 --distance l2

python -m experiments.matching_nets --dataset miniImageNet --fce False --k-test 5 --n-test 1 --distance cosine
python -m experiments.matching_nets --dataset miniImageNet --fce False --k-test 5 --n-test 5 --distance cosine
python -m experiments.matching_nets --dataset miniImageNet --fce True --k-test 5 --n-test 1 --distance cosine
python -m experiments.matching_nets --dataset miniImageNet --fce True --k-test 5 --n-test 5 --n-train 5 --distance cosine

# 1st order MAML
python -m experiments.maml --dataset omniglot --order 1 --n 1 --k 5 --eval-batches 10 --epoch-len 50
python -m experiments.maml --dataset omniglot --order 1 --n 5 --k 5 --eval-batches 10 --epoch-len 50
python -m experiments.maml --dataset omniglot --order 1 --n 1 --k 20 --meta-batch-size 16 \
    --inner-train-steps 5 --inner-val-steps 5 --inner-lr 0.1 --eval-batches 20 --epoch-len 100
python -m experiments.maml --dataset omniglot --order 1 --n 5 --k 20 --meta-batch-size 16 \
    --inner-train-steps 5 --inner-val-steps 5 --inner-lr 0.1 --eval-batches 20 --epoch-len 100

python -m experiments.maml --dataset miniImageNet --order 1 --n 1 --k 5 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset miniImageNet --order 1 --n 5 --k 5 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400

# 2nd order MAML
python -m experiments.maml --dataset omniglot --order 2 --n 1 --k 5 --eval-batches 10 --epoch-len 50
python -m experiments.maml --dataset omniglot --order 2 --n 5 --k 5 --eval-batches 20 --epoch-len 100 \
    --meta-batch-size 16 --eval-batches 20
python -m experiments.maml --dataset omniglot --order 2 --n 1 --k 20 --meta-batch-size 16 \
    --inner-train-steps 5 --inner-val-steps 5 --inner-lr 0.1 --eval-batches 40 --epoch-len 200
python -m experiments.maml --dataset omniglot --order 2 --n 5 --k 20 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 5 --inner-lr 0.1 --eval-batches 80 --epoch-len 400

python -m experiments.maml --dataset miniImageNet --order 2 --n 1 --k 5 --q 5 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 80 --epoch-len 400
python -m experiments.maml --dataset miniImageNet --order 2 --n 5 --k 5 --q 5 --meta-batch-size 2 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 80 --epoch-len 800


# ../../anaconda3/envs/few-shot/python.exe -m experiments.proto_nets --dataset omniglot --k-test 5 --n-test 1
# ../../anaconda3/envs/few-shot/python.exe -m experiments.proto_nets --dataset omniglot --k-test 5 --n-test 5
