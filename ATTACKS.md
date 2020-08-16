# Attacks

Here we provide intuition how the attacks work and then discuss the relevant implementation details.
To start with, let's assume we are given a simple program as input:

```bash 
$ cat tmp.ts
let a = 1;
let b = 3;
let c = a + b;
```

As a first step, the program is parsed and converted into its abstract syntex tree (AST) representation.
This can be illustrated by runnining the parser.

```bash
python robustcode/parsers/parser.py tmp.ts --pretty
0           SourceFile                                         ' '                                                         
 1          SyntaxList                                         ' '                                                         
  2         VariableStatement                                  ' '                                                         
   3        VariableDeclarationList                            ' '                                                         
    4       LetKeyword                     let                 ' '                                                         
    5       SyntaxList                                         ' '                                                         
     6      VariableDeclaration                                number                                                      
      7     Identifier                     a                   number                                                      
      8     FirstAssignment                =                   ' '                                                         
      9     FirstLiteralToken              1                   number                                                      
   10       SemicolonToken                 ;                   ' '                                                         
  11        VariableStatement                                  ' '                                                         
   12       VariableDeclarationList                            ' '                                                         
    13      LetKeyword                     let                 ' '                                                         
    14      SyntaxList                                         ' '                                                         
     15     VariableDeclaration                                number                                                      
      16    Identifier                     b                   number                                                      
      17    FirstAssignment                =                   ' '                                                         
      18    FirstLiteralToken              3                   number                                                      
   19       SemicolonToken                 ;                   ' '                                                         
  20        VariableStatement                                  ' '                                                         
   21       VariableDeclarationList                            ' '                                                         
    22      LetKeyword                     let                 ' '                                                         
    23      SyntaxList                                         ' '                                                         
     24     VariableDeclaration                                number                                                      
      25    Identifier                     c                   number                                                      
      26    FirstAssignment                =                   ' '                                                         
      27    BinaryExpression                                   number                                                      
       28   Identifier                     a                   number                                                      
       29   PlusToken                      +                   ' '                                                         
       30   Identifier                     b                   number                                                      
   31       SemicolonToken                 ;                   ' '  
```

Here, since we are interested in predicting types, the parser also runs a type inference the results of which are shown as the last column.

## Node Replacement Attacks

The node replacement attacks replace are used to change the one or mode node values, without changing the code structure.
Example of node replacement attacks include:
* Constant Replacement: i.e., changing `let a = 1` to `let a = 14`.
* Variable Renaming: i.e., renaming `a` to `x` which results in the following program:
 
```
let x = 1;
let b = 3;
let c = x + b;
```

Note that when renaming a variable we much ensure that:
* all occurrences of the variable are renamed together. In our case, this corresponds to renaming two AST nodes 7 and 28.
* variable is not renamed to another variable in the same scope. In our case, the variables in scope are `b` and `c` defined at nodes 16 and 25, respectively.

To apply node replacement fast during training and evaluation, we precompute all possible renamings and their associated constraints.
Concretely, in the example above, the precomputed 'rules' might look as follows:

```
# denotes rule for variable renaming
node: 7, conflicts: [16, 26], usages: [7, 28], original value: 'a', target values: ['a', 'b', 'c', 'x', ..., '<unk>']

# denotes rule for replacing an integer constant 
node: 9, conflicts: [],       usages: [9],     original value: 1,   target values: [0, 1, 2, ..., '<unk>']
```  

The first rule denotes variable renaming with the original value `a`, set of possible target values (these are computed using the model vocabulary, including the unknown word `<unk>`), set of usages (that need to be renamed together) and a set of conflicting nodes.
Note that we keep ids of the conflicting nodes, and not their actual values as these can change (i.e., other modifications are applied).
The second rule denotes replacing an integer constant. The main difference here is that the set of conflicts is empty. 

### Implementation: Representing Set of Valid Modifications (Rules)

Each valid node replacement is represented in `NodeRenameRule` class which contains:
* Stores pre-computed information about to which program (AST tree) and location in the program (AST node) the change applies
* Checks if the change can be applied (i.e., ensures that two variables are not renamed to the same value). For node replacement this is implemented as pre-computing a set of 'conflicting' nodes. The change is then applicable only if the new value is different than values at all 'conflicting' nodes.
* Applies the change

The class `AdversarialNodeReplacement` in `robustcode/models/robust/adversary/rules.py` computes an index 
of valid modifications that can be applied to each program (represented as AST).

Currently, the supported modification rules are:

```
rules.update(self.compute_constant_replacement(tree_id, tree))
rules.update(self.compute_variable_renaming(tree_id, tree))
rules.update(self.compute_prop_declaration_renaming(tree_id, tree))
rules.update(self.compute_property_assignment_renaming(tree_id, tree))
``` 

More information about the rules is provided in the comments.

### Implementation: Applying Modifications (Rules)

The class `RenameAdversary` in `robustcode/models/robust/adversary/adversary.py` applies subset of the selected modifications to a programs.
For performance reasons, class is defined to work over already batched programs (both is sequence of tokens and batch of trees).

Currently, the renaming rules are simply applied to all the valid positions in the program.
 

## Subtree Replacement

The class `AdversarialSubtreeReplacement` in `robustcode/models/robust/adversary/tree_rules.py` computes an index 
of valid tree replacement modifications. 
The corresponding adversary class that applies the modifications is `SubtreeAdversary` and defined in `robustcode/models/robust/adversary/adversary.py`.


### Usage:

A usage of how the adversaries are used and applied to evaluate a model can be seen in evaluating `baseline` models in `robustcode/models/robust/main.py`.

## Attack configuration
### Value renaming attack
In this attack, values in the graph are renamed in a way that doesn't change the meaning of the code.
Each time, N values are sampled according to a strategy and used in N distinct adversarial examples. 
There are multiple ways to choose the best adversarial value. These are:
* random - values are chosen randomly N times
* gradient greedy - values are sorted in order of their one-hot embedding's gradient and sampled sequentially
* gradient ascent - values with maximal gradient are sampled iteratively and the example is kept between iterations
* gradient "boosted" - in each iteration, loss is masked to only correctly classified positions, values with maximal gradient are sampled, and the input is reset to original
* N-individual gradients - adversarial examples selected individually for each predicted position, much slower, optimal when N = # predicted positions

Usage:
```bash
--adv_mode=[MODE], default: RANDOM
--n_renames=INT, default: 200

Where [MODE] is one of:
- RANDOM
- BATCH_GRADIENT_GREEDY
- BATCH_GRADIENT_ASCENT
- BATCH_GRADIENT_BOOSTING
- INDIVIDUAL_GRADIENT
```
### Subtree replacement attack
To do
