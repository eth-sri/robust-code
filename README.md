# Adversarial Robustness for Code

This project contains the source code for the paper [Adversarial Robustness for Code](https://files.sri.inf.ethz.ch/website/papers/icml20_robust.pdf) presented at ICML'20.

## Project Structure

```
├── data/               # preprocessed dataset (needs to be downloaded, see below)
├── robustcode/         # source code
│   ├── analysis/           # scripts to perform static analysis of code. e.g., computing different types of edges 
│   ├── datasets/           # contains scripts to generate datasets (including running TypeScript type inference).
│   ├── dedup/              # scripts to remove duplicates (or near duplicate) files from dataset of programs. Used for creating datasets.
│   ├── parsers/            
│   │   ├── typescript/     # typescript parser + type inference.
│   │   └── parser.py       # helper script to parse files from command line and pretty print the resulting AST
│   ├── models/             # main directory with models and training
│   │   ├── modules/        # contains models and various reusable components 
│   │   ├── robust/         # code for training and evaluating robust models of code
│   │   │   ├── adversary/  # implementation of the program modifications and adversary search
│   │   │   ├── gurobi/     # implementation of the representation refinement via ILP optimization
│   │   │   ├── config/     # configurations files used to run experiments
│   │   │   └── main.py     # main file to train and evaluate models               
└── └── util/               # helper classes
``` 

## Installation

Prepare virtual environment. Tested with python 3.6.8 but any version 3.5+ should work:

```bash
python3.6 -m venv venv
```
or 
```bash 
virtualenv -p python3.6 --system-site-packages venv
```

Install requirements

```bash
source venv/bin/activate
```

Install this project in editable state

```bash
pip install wheel
pip install -e .
```

### Pre-commit Hooks

For developlment, we recomment installing pre-commit hooks

```bash
pip install -e .[dev]
pre-commit install
```

The following hooks are enabled:
 - [Flake8](https://gitlab.com/pycqa/flake8): Python Linter. The configuration can be adjusted in `setup.cfg`
 - [Black](https://github.com/psf/black): Code formatter. Instructions how to integrate it with your IDE can be found [here](https://github.com/psf/black#editor-integration)
 - [Reorder Python Imports](https://github.com/asottile/reorder_python_imports).
 
If necessary, the hooks can be bypassed with the `--no-verify` option passed to `git commit`.

## Datasets

Download and extract datasets using:

```
wget https://files.sri.inf.ethz.ch/data/bigcode/deeptyperast.tar.gz
tar -vxf deeptyperast.tar.gz
```

The models presented in the paper are evaluated on dataset `deeptyperast_1k`.
However, we also include a larger dataset with ~10x more files `deeptyperast_10k`.

Additionally, the instructions to generate the dataset from scratch are provided in `robustcode/datasets/types/README.md`

### Dataset Format (Optional)
_Not necessary for evaluating and training with existing datasets. The information below is provided as a documentation for using the dataset for other task and as a help for debugging purposes_

The preprocessed dataset is stored as four files:

├── deeptyperast_1k/
│   ├── config.json         # configuration file used to generate the dataset 
│   ├── test.json.gz        # test split
│   ├── valid.json.gz       # valid split
│   ├── train.json.gz       # train split

Each split is a compressed gzip archive that contains one JSON file per line.
To read the dataset manually, please refer to the script `robustcode/models/robust/dataset_info.py`.


## Dependencies

For individual project dependencies please consult the projects README file. 

### TypeScript Parser (Optional)
_not necessary for evaluating and training with existing datasets_

The parser requires nodejs and npm installed.
On ubuntu, these can be installed using:

```bash
sudo apt-get install nodejs npm
```

```bash
cd robustcode/parsers/typescript
npm install
npm test
cd ../../..
```

Running `npm test` checks that the parser correctly removes type annotations while parsing.

We provide a separate script that can be used to parse individual files:

```bash
./robustcode/parsers/parser.py robustcode/parsers/typescript/fixtures/fnc.ts

[{'id': 0, 'type': 'SourceFile', 'children': [1]}, {'id': 1, 'type': 'SyntaxList', 'children': [2, 60, 111, 142, 182, 244, 343, 376, 404, 418, 501, 580]}, {'id': 2, 'type': 'FunctionDeclaration', 'children': [3, 4, 5, 6, 16, 17, 18, 19], 'target': '(x: number, y: number) => number'} ...
``` 

To dump AST in a nicer way, use `--pretty` option

```bash
./robustcode/parsers/parser.py robustcode/parsers/typescript/fixtures/fnc.ts --pretty

0                  SourceFile                                         ' '                                                         
 1                 SyntaxList                                         ' '                                                         
  2                FunctionDeclaration                                '(x: number, y: number) => number'                          
   3               FunctionKeyword                function            '(x: number, y: number) => number'                          
   4               Identifier                     add                 '(x: number, y: number) => number'                          
   5               OpenParenToken                 (                   ' '                                                         
   6               SyntaxList                                         ' ' 
```


### Gurobi ILP solver (required for training with learned representation)

The ILP solver used in our work is Gurobi v8.1.1.
To obtain academic licence please visit: https://www.gurobi.com/downloads/

Once the solver is downloaded, install it for the current virtual environment using:

```bash
cd /opt/gurobi811/linux64
sudo <project_dir>/venv/bin/python setup.py install
```

Note that you'll also need to modify the following environment variables for the solver to work correctly:

```
export GUROBI_HOME="/opt/gurobi811/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

The easiest way to do this is to include the above lines in the `~/.bashrc` file.

Note that currently it is required to install the solver to train and evaluate the models. We are planning to remove this dependency in the future.


## Models
The models in our work are located in:
```
robustcode
├── models/             
│   ├── modules/         
│   │   ├── rnn/  # LSTM and DeepTyper Models
└── └── └── dgl/  # Graph Neural Netowrk models (Graph Transformer Model, GCN and GGNN)
``` 

## Configurations
The run configurations are located in:

```
robustcode
├── models/             
│   ├── robust/         
│   │   ├── configs/  
|   │   │   └── ast_rnn.ini                 # LSTM
|   │   │   └── ast_rnn_attn.ini            # LSTM with attention
|   │   │   └── ast_deeptyper.ini           # DeepTyper
|   │   │   └── ast_gcn.ini                 # GCN
|   │   │   └── ast_ggnn.ini                # GGNN
└── └── └── └── ast_ugraph_transformer.ini  # Graph Transformer
```

These include hyperparameters used in our experiments such as learning rate, number of epochs, dimensionality, etc.

## Evaluation

We include model checkpoints and evaluation results in the 'results' folder.
A quick summary of the results can be obtained by running:

```bash
cd robustcode/models/robust/
python summarize_results.py --results_dir results/

   accuracy (median)  accuracy (std)  robustness (median)  robustness (std)  N                                     name
0              87.50        0.361232            51.934713          1.282532  3             ast/rnn_attn/advTrue_valTrue
1              88.17        0.151731            44.952229          1.279452  3            ast/rnn_attn/advFalse_valTrue
2              81.90        0.535413            49.335191          3.057583  3                  ast/gcn/advTrue_valTrue
3              82.56        0.619265            49.112261          1.107310  3                 ast/gcn/advFalse_valTrue
4              88.31        0.380526            50.019904          0.485651  3   ast/ugraph_transformer/advTrue_valTrue
5              89.28        0.890780            47.408439          1.018467  3  ast/ugraph_transformer/advFalse_valTrue
6              87.08        0.318748            55.147293          2.611508  3            ast/deeptyper/advTrue_valTrue
7              88.38        0.151950            52.416401          1.239002  3           ast/deeptyper/advFalse_valTrue
8              86.07        0.245357            57.862261          1.538453  3                 ast/ggnn/advTrue_valTrue
9              86.64        0.445945            52.050159          0.394119  3                ast/ggnn/advFalse_valTrue
```

This corresponds to the results presented in Table 2.

For a more detailed evaluation please run:

```bash
time python std_train.py --config configs/ast_ggnn.ini --adversarial True --n_subtree=300 --n_renames=1000 --repeat 3
```

for adversarial (`--adversarial True`) and standard training (`--adversarial False`).
For training with representation refinement, the following command is needed.

```bash
time python train_sparse.py --config configs/ast_ggnn.ini --n_subtree=300 --n_renames=1000 --repeat 1 --max_models 8 --last_base --eval

Base Accuracy:  22033/ 25120 (87.71%), reject:      0/ 25120 (0.00%)
	   correct                SOUND:  16819/ 22033 (76.34%)
	   correct        SOUND_PRECISE:  16819/ 22033 (76.34%)
	   correct              UNSOUND:   5214/ 22033 (23.66%)
	 incorrect              UNSOUND:   3087/  3087 (100.00%)
	    reject                SOUND:      0/     0 (0.00%)
	    reject        SOUND_PRECISE:      0/     0 (0.00%)
	    reject              UNSOUND:      0/     0 (0.00%)
	     total                SOUND:  16819/ 25120 (66.95%)
	     total              UNSOUND:   8301/ 25120 (33.05%)
```

## Training

#### Experiments: LSTM Baseline
To run the baseline LSTM model, use command is the same as for evaluation (while in the virtual environment):

```bash
cd robustcode/models/robust
python std_train.py --config configs/ast_rnn_attn.ini --adversarial False --n_subtree=300 --n_renames=1000 --repeat 1
```

Make sure that the results folder (changed using `--save_dir`) does not already contain the same model. If it does, the model will be loaded instead of trained from scratch.
On a Nvidia Titan RTX, the training of the LSTM model is very fast and takes ~2 minutes.
You should see the following output:

```
 Training Model...
 	Epoch 0...
 		loss=0.6361, act_loss=0.0000
 		Accuracy:          all   188242/  605683 (31.08%) |   mask_valid   188242/  223912 (84.07%) | mask_constant    29197/   29204 (99.98%)
 		Most common predictions:
 		all
 			                       '<unk>' precision:     130085/    601616  21.62%, recall:     130085/    138736  93.76%
 			                      'string' precision:      36863/    143071  25.77%, recall:      36863/     50015  73.70%
 			                      'number' precision:       9320/     25920  35.96%, recall:       9320/     15899  58.62%
 			                     'boolean' precision:       6352/     20275  31.33%, recall:       6352/      9822  64.67%
 			                        'void' precision:       2822/     13782  20.48%, recall:       2822/      3857  73.17%
 			                  '() => void' precision:       2793/     13722  20.35%, recall:       2793/      4426  63.10%
 			                '() => string' precision:          7/      1530   0.46%, recall:          7/       434   1.61%
 			               '() => boolean' precision:          0/        10   0.00%, recall:          0/       274   0.00%
 			                '() => number' precision:          0/         2   0.00%, recall:          0/       449   0.00%
 			                      '<null>' precision:          0/         0   0.00%, recall:          0/    381771   0.00%
 			                         total  accuracy:     188242/    605683  31.08%
 		mask_valid
 			                       '<unk>' precision:     130085/    153159  84.93%, recall:     130085/    138736  93.76%
 			                      'string' precision:      36863/     43475  84.79%, recall:      36863/     50015  73.70%
 			                      'number' precision:       9320/     10424  89.41%, recall:       9320/     15899  58.62%
 			                     'boolean' precision:       6352/      8718  72.86%, recall:       6352/      9822  64.67%
 			                        'void' precision:       2822/      4446  63.47%, recall:       2822/      3857  73.17%
 			                  '() => void' precision:       2793/      3671  76.08%, recall:       2793/      4426  63.10%
 			                '() => string' precision:          7/        19  36.84%, recall:          7/       434   1.61%
 			                '() => number' precision:          0/         0   0.00%, recall:          0/       449   0.00%
 			               '() => boolean' precision:          0/         0   0.00%, recall:          0/       274   0.00%
 			                     'unsound' precision:          0/         0   0.00%, recall:          0/         0   0.00%
 			                         total  accuracy:     188242/    223912  84.07%
 		valid_prec: 84.06963449926756
 	Done in 4.81 s
 	Epoch 1...
 		loss=0.3557, act_loss=0.0000
 		Accuracy:          all   187840/  605683 (31.01%) |   mask_valid   187840/  223912 (83.89%) | mask_constant    29193/   29204 (99.96%)
 		Most common predictions:
 		all
 			                       '<unk>' precision:     126265/    582565  21.67%, recall:     126265/    138736  91.01%
 			                      'string' precision:      40687/    165211  24.63%, recall:      40687/     50015  81.35%
 			                      'number' precision:       9234/     23523  39.26%, recall:       9234/     15899  58.08%
 			                     'boolean' precision:       5838/     16869  34.61%, recall:       5838/      9822  59.44%
 			                  '() => void' precision:       3134/     15410  20.34%, recall:       3134/      4426  70.81%
 			                        'void' precision:       2624/     13946  18.82%, recall:       2624/      3857  68.03%
 			                '() => string' precision:         58/      2308   2.51%, recall:         58/       434  13.36%
 			               '() => boolean' precision:          0/        95   0.00%, recall:          0/       274   0.00%
 			                '() => number' precision:          0/         1   0.00%, recall:          0/       449   0.00%
 			                      '<null>' precision:          0/         0   0.00%, recall:          0/    381771   0.00%
 			                         total  accuracy:     187840/    605683  31.01%
 		mask_valid
 			                       '<unk>' precision:     126265/    144165  87.58%, recall:     126265/    138736  91.01%
 			                      'string' precision:      40687/     55281  73.60%, recall:      40687/     50015  81.35%
 			                      'number' precision:       9234/      9816  94.07%, recall:       9234/     15899  58.08%
 			                     'boolean' precision:       5838/      6379  91.52%, recall:       5838/      9822  59.44%
 			                  '() => void' precision:       3134/      4052  77.34%, recall:       3134/      4426  70.81%
 			                        'void' precision:       2624/      4039  64.97%, recall:       2624/      3857  68.03%
 			                '() => string' precision:         58/       180  32.22%, recall:         58/       434  13.36%
 			                '() => number' precision:          0/         0   0.00%, recall:          0/       449   0.00%
 			               '() => boolean' precision:          0/         0   0.00%, recall:          0/       274   0.00%
 			                     'unsound' precision:          0/         0   0.00%, recall:          0/         0   0.00%
 			                         total  accuracy:     187840/    223912  83.89%
 		valid_prec: 83.89009968201793
 	Done in 4.78 s
```  

which shows the model performance while training. 
Here, the 'mask_valid' denotes positions for which prediction is made (i.e., identifiers, expressions, etc.) and for which the loss is computed. 
The results in 'all' are not interpretable and can be ignored as predicting types for other statements is not meaning full.


After the model is trained, it will be automatically evaluated in the adversarial setting.
This is much more time consuming and currently takes ~15 minutes for the LSTM model and ~20 for the GNN model.
The adversarial evaluation prints the statistics periodically after each batch. At the end, you should see following result (for the LSTM baseline):

```
Base Accuracy: 106613/121153 (88.00%), reject:      0/121153 (0.00%)
	   correct                SOUND:   5077/106613 (4.76%)
	   correct        SOUND_PRECISE:   5077/106613 (4.76%)
	   correct              UNSOUND: 101536/106613 (95.24%)
	 incorrect              UNSOUND: 116076/     0 (0.00%)
	    reject                SOUND:      0/     0 (0.00%)
	    reject        SOUND_PRECISE:      0/     0 (0.00%)
	    reject              UNSOUND:      0/     0 (0.00%)
	     total                SOUND:   5077/121153 (4.19%)
	     total              UNSOUND: 116076/121153 (95.81%)
``` 

This is a more detailed robustness breakdown that presented in the paper.
Here:
 - `correct` denotes subset of the samples for which the original model makes correct prediction. 
 - `incorrect` denotes subset of the samples which are mispredicted by the original models, and
 - `reject` are samples for which the original model abstains. Since the baseline model is not trained with `--abstain=False`, it never abstains.
 
The `SOUND`, `SOUND_PRECISE` and `UNSOUND` are results when considering adversarial modifications of the original samples.
In particular:
 - `SOUND` denotes that the model is correct (or abstains) for all the adversarial modifications
 - `SOUND_PRECISE` denotes that the model is both correct and does not abstain for all the adversarial modifications.  
 - `UNSOUND` denotes that there exists atleast one adversarial sample that leads to misclasification
 
This experiment replicates the row `LSTM_AST` in Table 1 which reports accuracy 88% and robustness 4.19%.

#### Experiments: Other models

To run other models, simply change the config file. E.g., for ggnn use the following:

```bash
cd robustcode/models/robust
python std_train.py --config configs/ast_ggnn.ini --adversarial False --n_subtree=300 --n_renames=1000 --repeat 1
```

The output will be similar to what is described above.

#### Experiments: Adversarial Training

To enable adversarial training for the baseline models, use `--adversarial=True` flags while training.

#### Experiments: Robust Models

To train robust models, use the following command:

```bash
cd robustcode/models/robust
time python train_sparse.py --config configs/ast_ggnn.ini --n_subtree=300 --n_renames=1000 --repeat 1 --max_models 8 --last_base
```

Here, `max_models=8` denotes that we want to train a chain of 8 models. 
`--last_base` denotes that the last model should cover all the remaining samples.
Note that when running with all the models, training will take several hours.

# Citing This Work

```
@incollection{bielik20robust,
  title = {Adversarial Robustness for Code},
  author = {Bielik, Pavol and Vechev, Martin},
  booktitle = {Proceedings of The 37rd International Conference on Machine Learning},
  year = {2020},
  series = {ICML'20}
}
```

# Contributors

* [Pavol Bielik](https://www.sri.inf.ethz.ch/people/pavol) - pavol.bielik@inf.ethz.ch

# License and Copyright

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](http://www.apache.org/licenses/)
