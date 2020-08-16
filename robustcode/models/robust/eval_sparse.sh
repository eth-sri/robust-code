#!/bin/bash

#model=ugraph_transformer
model=ggnn
mkdir -p logs/near-camera-ready/${model}
for i in {12..18}; do
   time python train_sparse.py --config configs/ast_${model}.ini --repeat 1 --n_subtree=100 --n_renames=200 --max_models=$i --eval &> logs/near-camera-ready/${model}/${i}.log
done