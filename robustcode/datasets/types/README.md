# TypeScript and JavaScript Dataset

Uses the same set of projects and their version as in [DeepTyper][1]

#### 1. Download the GitHub projects
```bash
./cloner.sh
```

Note that by default, the cloner script downloads only the first 10 projects.
To download all projects, remove `head -n 10` from the `cloner.sh`.


#### 2. Install dependencies

```bash
npm install typescript@3.4.2
npm install mkdirp
npm install argparse
```

#### 3. Create dataset

The following script finds all downloaded projects, initializes their dependencies and runs typescript type inference to extract types of variables and expressions.
Note that running this script takes several hours on the full dataset and uses significant amount of disk space for the downloaded projects.
For example, analyzing the first 10 projects takes 2.6 GB of space and ~5 minutes.  

```bash
python clean_repos.py --num_threads=12
```

The resulting datasets are saved in `data/out` folder:

```
data
├── out/
│   ├── test.json.gz
│   ├── train.json.gz
│   ├── valid.json.gz
```

Each dataset is a file containing one program per line, encoded as JSON.
The format of line is as follows:
```json
{
  "filename:": "<path to the analyzed file>",
  "dependencies": ["list of dependencies use to infer the types. Contains both library files as well as other files from the same project"],
  "ast": ["list of AST nodes"]
}
```

Each AST node is stored as JSON with the following format:
```json
{
  "id": 106,            # node id 
  "type": "Identifier", # type of the AST node
  "value": "x",         # (Optional) value of the AST node  
  "children": [107, 108, 109], # (Optional) ids of the AST children   
  "target": "string"    # (Optional) Inferred type by the TypeScript type inference
  "gold": "string"      # (Optional) Annotation written by the developer (if available)
}
```

 

[1]: https://github.com/DeepTyper/DeepTyper