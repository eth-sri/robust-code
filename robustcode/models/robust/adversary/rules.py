import collections
import random
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import dgl
import numpy as np
import torch

from robustcode.analysis.graph import AstNode
from robustcode.analysis.graph import AstTree
from robustcode.analysis.graph import TypeScriptGraphAnalyzer
from robustcode.models.robust.dataset import Dataset
from robustcode.util.misc import acc
from robustcode.util.misc import trim


class NodeRenameRule:
    def __init__(
        self,
        tree_id: int,
        node_id: int,
        usages: Iterable[int],
        conflits: Iterable[int],
        value: str,
        candidate_values: Iterable[int],
        fixed_value_offset=None,
        decl_id=None,
    ):
        self.tree_id = tree_id
        self.node_id = node_id
        # declaration site for variables or None
        self.decl_id = decl_id

        "Do not replace nodes with value >= than the fixed_value_offset"
        self.fixed_value_offset = fixed_value_offset

        """
        Original value of the rename position and a list of candidate values to which it can be renamed
        """
        self.value = value
        self.original_value = value
        self.candidate_values = list(
            candidate_values
        )  # make a copy to shuffle the values separately for each rule

        """
        Positions in the tree where the declaration is used (including the declaration position)
        """
        self.usages = list(usages)

        """
        Positions in the tree that cannot have the same name
        """
        self.conflicts = conflits

    def __repr__(self):
        return "Node: {:6d}, Value: {:<20s}, Tree: {:6d}, Conflits: {}\n\t#candidates: {:6d}, usages: {}".format(
            self.node_id,
            trim(self.value, 20),
            self.tree_id,
            self.conflicts,
            len(self.candidate_values),
            self.usages,
        )

    def is_applicable(self, value: int, tree: Union[AstTree, dgl.DGLGraph], offset=0):
        """
        value is either unknown (0) or different to all the conflicting nodes
        """
        if value == 0 or len(self.conflicts) == 0:
            return True
        if isinstance(tree, AstTree):
            return all(tree.nodes[node_id].value != value for node_id in self.conflicts)
        elif isinstance(tree, dgl.DGLGraph):
            conflicts = [c + int(offset) for c in self.conflicts]
            return (tree.ndata["values"][conflicts].cpu() != value).all()
        else:
            conflicts = [c + int(offset) for c in self.conflicts]
            return all(tree[conflicts] != value)

    def apply_first_valid(self, start_idx, target, usage_offset=0):
        """

        :param start_idx:     index to generate distinct adversarial examples
        :param target:        numpy tensor of graph values for entire batch
        :param usage_offset:  offset of this example in batch
        :return:
        """
        usages = (
            self.usages
            if usage_offset == 0
            else [u + usage_offset for u in self.usages]
        )
        if self.fixed_value_offset is not None:
            usages = [u for u in usages if target[u] < self.fixed_value_offset]
        if not usages:
            return

        # iterate self.candidate_values circularly until an applicable value is found
        for idx_offset in range(len(self.candidate_values)):
            value = int(
                self.candidate_values[
                    (start_idx + idx_offset) % len(self.candidate_values)
                ]
            )
            if self.is_applicable(value, target, offset=usage_offset):
                target[usages] = value
                # print("Using", value)
                return

    def apply_value(self, value, target, usage_offset=0):
        usages = (
            self.usages
            if usage_offset == 0
            else [u + usage_offset for u in self.usages]
        )
        if self.fixed_value_offset is not None:
            usages = [u for u in usages if target[u] < self.fixed_value_offset]
        if not usages:
            return
        # if not self.is_applicable(value, target, offset=usage_offset):
        #     raise Exception("Value not applicable!")
        target[usages] = value

    def apply(self, value, tree: AstTree):
        assert value in self.candidate_values
        assert self.is_applicable(value, tree)
        for node_id in self.usages:
            tree.nodes[node_id].value = value

    def reset(self):
        self.value = self.original_value

    def __hash__(self):
        """
        Well defined only if rules for the same tree are indexed
        """
        return self.decl_id


class ShuffleStrategy(ABC):
    def initialize(self, model, batch, position_mask=None):
        pass

    @abstractmethod
    def shuffle_candidates(self, rule: NodeRenameRule):
        raise NotImplementedError


class ShuffleStrategyIndividualGradient(ShuffleStrategy):
    def __init__(self, max_samples):
        self.max_samples = max_samples  # TODO: max_samples should be the actual maximal number of predictable lines

    def initialize(self, model, batch, position_mask=None):
        """Computes and saves gradients of individual loss w.r.t. each input position."""
        self.batch_ids = (
            batch.ids.tolist() if isinstance(batch.ids, torch.Tensor) else batch.ids
        )
        self.tree_sizes = batch.lengths.tolist()

        # TODO: is this correct?
        self.x_grads_generator = self._merged_gradient_generator(model, batch)

    def _single_gradient_generator(self, model, batch, minibatch):
        gen = model.get_input_gradients_individual(
            minibatch, mask_field="mask_valid", max_samples=self.max_samples
        )
        while True:
            yield torch.split(next(gen), self.tree_sizes)

    def _merged_gradient_generator(self, model, batch):
        """If the graph is split into multiple network inputs, we need to merge the gradients after backpropagation."""
        generators = [
            model.get_input_gradients_individual(
                minibatch,
                mask_field="mask_valid",
                max_samples=self.max_samples,
                is_batched=True,
            )
            for minibatch in batch
        ]
        while True:  # for i in range(self.max_samples) inside
            # try:
            x_grads = []
            for gen in generators:
                try:
                    x_grad_partial = next(gen)
                    x_grads.append(x_grad_partial)
                except StopIteration:
                    continue
            if len(x_grads) <= 0:
                print("Iteration ended with ", len(x_grads), " items")
                raise StopIteration()
            x_grad = torch.cat(x_grads, dim=0).transpose(0, 1)
            yield x_grad
        # except StopIteration:
        #     raise

    def for_next_position(self):
        try:
            x_grad = next(self.x_grads_generator)
            self.tree_grads = {
                tree_id: x_grad[off].detach()
                for off, tree_id in enumerate(self.batch_ids)
            }
        except StopIteration:
            print(
                "Warning: Gradients exhausted. Ensure that max_samples == len(valid_mask)"
            )

    def shuffle_candidates(self, rule: NodeRenameRule):
        """
        Ranks candidate tokens for an input position based on previously computed gradients.
        :param rule:
        """
        x_tree_grad = self.tree_grads[rule.tree_id]
        # here the grads might not have been computed? (why?)
        # print(x_tree_grad.shape, rule.usages)
        usages = [
            i for i in rule.usages if i < x_tree_grad.shape[0]
        ]  # TODO: sometimes it is >= length, why?
        grads_per_position = x_tree_grad[usages, :]
        grads_per_value = torch.sum(grads_per_position, dim=0).cpu().numpy()
        rule.candidate_values.sort(
            key=lambda value: grads_per_value[value], reverse=True
        )


class ShuffleStrategyGradient(ShuffleStrategy):
    def initialize(self, model, batch, minibatch, position_mask=None):
        """
        Computes and saves gradients of batch loss w.r.t. each input position.
        :param model
        :param batch
        :param minibatch: required to ensure that changes in graph are reflected
        :param position_mask: (optional) mask to apply over loss before backprop
        """
        tree_sizes = batch.lengths.tolist()
        batch_ids = (
            batch.ids.tolist() if isinstance(batch.ids, torch.Tensor) else batch.ids
        )

        if minibatch is not None:
            # compute gradient only for one minibatch
            x_inputs, x_grad = model.get_input_gradients_batch(
                model.loss_function,
                minibatch,
                mask_field="mask_valid",
                adversarial_mask=position_mask,
            )
            self.x_grad = torch.split(x_grad, tree_sizes)

        else:
            x_grads = []
            for i, minibatch in enumerate(batch):
                # RNN model: (batch_size, max(tree_length), embedding_dim)
                # print("Input:", len(minibatch.X))
                x_inputs, x_grad_part = model.get_input_gradients_batch(
                    model.loss_function,
                    minibatch,
                    mask_field="mask_valid",
                    adversarial_mask=position_mask[i] if position_mask else None,
                )
                if x_grad_part is None:
                    # print("WARNING: NO GRADIENT!!!!")
                    continue
                x_grads.append(x_grad_part)
                # print("Part grad:", x_grad_part.shape)
            x_grad = torch.cat(x_grads, dim=0)  # TODO: is this OK?
            self.x_grad = x_grad.transpose(0, 1)

        self.tree_grads = {
            tree_id: self.x_grad[off].detach() for off, tree_id in enumerate(batch_ids)
        }

    def shuffle_candidates(self, rule: NodeRenameRule):
        """
        Ranks candidate tokens for an input position based on previously computed gradients.
        :param rule:
        """
        x_tree_grad = self.tree_grads[rule.tree_id]
        if rule.usages[-1] >= x_tree_grad.shape[0]:
            print("WARNING: usage >= max. position in gradient, skipping!")
            return
            # TODO: why does this happen??
        grads_per_position = x_tree_grad[rule.usages, :]
        grads_per_value = torch.sum(grads_per_position, dim=0).cpu().numpy()
        rule.candidate_values.sort(
            key=lambda value: grads_per_value[value], reverse=True
        )


class ShuffleStrategyRandom(ShuffleStrategy):
    def shuffle_candidates(self, rule: NodeRenameRule):
        random.shuffle(rule.candidate_values)


class ShuffleStrategyGuided(ShuffleStrategy):
    def __init__(self, value_attribution, target_label: int):
        super().__init__()
        self.value_attribution = value_attribution
        self.target_label = target_label

        """
        We want nodes to predict target_label 
        """
        self.values = value_attribution.get_values(target_label, is_positive=True)

    def shuffle_candidates(self, rule: NodeRenameRule):
        scored_values = []
        unscored_values = []
        for v in rule.candidate_values:
            score = self.values.get(v, 0)
            if score > 0:
                scored_values.append((score, v))
            else:
                unscored_values.append(v)
        # scored_values = [(self.values.get(v), v) for v in rule.candidate_values if v in self.values]
        # unscored_values = [v for v in rule.candidate_values if v not in self.values]
        random.shuffle(unscored_values)
        scored_values.sort(reverse=True)
        rule.candidate_values = [v for (score, v) in scored_values] + unscored_values


class RenameRulesForTree:
    def __init__(self, tree_id: int, tree: AstTree):
        self.tree_id = tree_id
        # should be a numericalized tree
        self.tree = tree
        self.rules: List[NodeRenameRule] = []
        self.nodes_to_rule: Dict[int, NodeRenameRule] = {}

    def update(self, rules: Iterable[NodeRenameRule]):
        for rule in rules:
            if len(rule.candidate_values) == 1:
                # skip rules with unique values
                continue
            self.add(rule)

    def add(self, rule: NodeRenameRule):
        self.rules.append(rule)
        for node_id in rule.usages:
            assert (
                node_id not in self.nodes_to_rule
            ), "node {} already added\n{}\n{}".format(
                node_id, rule, self.nodes_to_rule[node_id]
            )
            self.nodes_to_rule[node_id] = rule

    def print_tree(self):
        def node_label(node: AstNode):
            pos = self.nodes_to_rule.get(node.id, None)
            if pos is not None:
                return "{:>6s} {:6d} {}".format(
                    str(pos.decl_id), len(pos.candidate_values), pos.usages
                )
            return ""

        print(self.tree.root.dumpAsString(label=node_label))

    def init_rules(
        self,
        nodes: Optional[Iterable[int]] = None,
        num_samples=10,
        shuffle: Optional[ShuffleStrategy] = None,
    ):
        rules = (
            self.rules
            if nodes is None
            else set(self.nodes_to_rule[i] for i in nodes if i in self.nodes_to_rule)
        )
        if not rules:
            return [], 0

        if shuffle is not None:
            for rule in rules:
                # random.shuffle(rule.candidate_values)
                shuffle.shuffle_candidates(rule)

        max_samples = max(len(rule.candidate_values) for rule in rules)
        num_samples = (
            min(num_samples, max_samples) if num_samples != -1 else max_samples
        )
        return rules, num_samples

    def forEachTreeRuleApply(
        self, nodes: Optional[Iterable[int]] = None, num_samples=10, shuffle=True
    ):
        rules, num_samples = self.init_rules(nodes, num_samples, shuffle)
        if not rules:
            return

        for idx in range(num_samples):
            for rule in rules:
                value = rule.candidate_values[idx % len(rule.candidate_values)]
                if rule.is_applicable(value, self.tree):
                    rule.apply(value, self.tree)
            yield self.tree

    def forEachGraphRuleApply(
        self, g: dgl.DGLGraph, mask: torch.Tensor = None, num_samples=10, shuffle=True
    ):
        nodes = np.flatnonzero(mask.cpu().numpy()) if mask is not None else None
        rules, num_samples = self.init_rules(nodes, num_samples, shuffle)
        if not rules:
            return

        original_values = g.ndata["values"]
        for idx in range(num_samples):
            values = g.ndata["values"].cpu().numpy()

            for rule in rules:
                value = int(rule.candidate_values[idx % len(rule.candidate_values)])
                applicable = rule.is_applicable(value, values)
                if applicable:
                    values[rule.usages] = value

            g.ndata["values"] = torch.tensor(
                values, dtype=torch.long, device=original_values.device
            )
            yield g
        g.ndata["values"] = original_values


class RenameRulesIndex:
    def __init__(self):
        self.per_tree_rules: Dict[int, RenameRulesForTree] = {}

    def add(self, rule: RenameRulesForTree):
        self.per_tree_rules[rule.tree_id] = rule

    def get(self, tree_id) -> RenameRulesForTree:
        assert tree_id in self.per_tree_rules
        return self.per_tree_rules.get(tree_id, None)


class NodeValueIndexStr:
    def __init__(self, dataset: Dataset, trees: Dict[int, AstTree]):
        self.dataset = dataset

        self.per_type_values = collections.defaultdict(set)
        for tree in trees.values():
            for node in tree.nodes:
                self.per_type_values[node.type].add(node.value)

        "convert to list to allow indexing"
        self.per_type_values = {
            key: list(values) for key, values in self.per_type_values.items()
        }

        "binary operators used for generating adversary examples"
        self.bin_ops = collections.defaultdict(list)
        bin_expr = "BinaryExpression"
        allowed_bin_ops = [
            "PlusToken",
            #            'FirstAssignment',
            "GreaterThanToken",
            "EqualsEqualsEqualsToken",
            #            'AmpersandAmpersandToken',
            "ExclamationEqualsEqualsToken",
            #            'InKeyword',
            "FirstBinaryOperator",
            "BarBarToken",
            #            'SlashToken',
            "MinusToken",
            "GreaterThanEqualsToken",
            #            'FirstCompoundAssignment',
            #            'InstanceOfKeyword',
            "LessThanEqualsToken",
            "EqualsEqualsToken",
            #            'AsteriskToken',
            #            'PercentToken',
            "BarToken",
            #            'CommaToken',
            "ExclamationEqualsToken",
            #            'AsteriskEqualsToken',
            "MinusEqualsToken",
            #            'AmpersandToken',
            "GreaterThanGreaterThanGreaterThanToken",
            "GreaterThanGreaterThanEqualsToken",
        ]
        # allowed_bin_ops = [self.dataset.TYPES.vocab.stoi[op] for op in allowed_bin_ops]

        for tree in trees.values():
            for node in tree.nodes:
                if node.type != bin_expr or len(node.children) != 3:
                    continue
                op = node.down_first().right()
                if op.value not in self.bin_ops[op.type] and op.type in allowed_bin_ops:
                    self.bin_ops[op.type].append(op.value)

        self.bin_ops = [
            (op_type, op_value)
            for op_type, op_values in self.bin_ops.items()
            for op_value in op_values
        ]

    def values_for_type(self, node_type):
        # if isinstance(node_type, str):
        #     node_type = self.dataset.TYPES.vocab.stoi[node_type]
        return self.per_type_values.get(node_type, [])


class NodeValueIndex:
    """
    Collects list of values that were seen for a given type in the training dataset
    """

    def __init__(self, dataset: Dataset, trees: Dict[int, AstTree]):
        self.dataset = dataset

        self.per_type_values = collections.defaultdict(set)
        for tree in trees.values():
            for node in tree.nodes:
                assert not isinstance(
                    node.value, str
                ), "Expects numericalized trees, got: {} {}".format(
                    type(node.value), node
                )
                self.per_type_values[node.type].add(node.value)

    def values_for_type(self, node_type):
        if isinstance(node_type, str):
            node_type = self.dataset.TYPES.vocab.stoi[node_type]
        return self.per_type_values.get(node_type, [])


class AdversarialNodeReplacement:
    def __init__(self, value_index: NodeValueIndex, fixed_value_offset):
        # self.node_types = node_types

        self.analyzer = TypeScriptGraphAnalyzer()
        self.value_index = value_index

        self.fixed_value_offset = fixed_value_offset

    @staticmethod
    def property_declaration(node):
        "private x = ..."
        return (
            node.type == "Identifier"
            and node.pos_in_parent() == 1
            and node.has_up()
            and node.up().type == "PropertyDeclaration"
        )

    @staticmethod
    def property_assignment_left(node):
        "{x: _}"
        return (
            node.has_up()
            and node.up().type == "PropertyAssignment"
            and node.is_first_child()
        )

    @staticmethod
    def property_access(node):
        "_.x"
        return (
            node.pos_in_parent() == 2
            and node.left().type == "DotToken"
            and node.has_up()
            and node.up().type == "PropertyAccessExpression"
        )

    CONSTANTS = set(
        [
            "StringLiteral",
            "TemplateExpression",
            "FirstTemplateToken",  # string
            "TrueKeyword",
            "FalseKeyword",  # boolean
            "FirstLiteralToken",  # numbers
        ]
    )

    @staticmethod
    def is_constant(node):
        return node.type in AdversarialNodeReplacement.CONSTANTS

    def compute_property_assignment_renaming(self, tree_id, tree: AstTree):
        rename_nodes = {}
        declarations = [
            node
            for node in tree.nodes
            if AdversarialNodeReplacement.property_assignment_left(node)
        ]
        decl_scopes = {decl.id: self.analyzer.get_scope(decl) for decl in declarations}

        sample = self.value_index.dataset.get_sample_by_id(tree_id)
        for declaration in declarations:
            scope: AstNode = decl_scopes[declaration.id]
            rename_nodes[declaration.id] = declaration

            for node in scope.forEachNode():
                if node.value != declaration.value:
                    continue

                if not AdversarialNodeReplacement.property_access(node):
                    continue

                target_type = sample.target[node.left().left().id]
                if not (
                    target_type[0] == "{"
                    and target_type[-1] == "}"
                    and declaration.value in target_type
                ):
                    continue

                rename_nodes[node.id] = declaration

        return self.__process_decl(decl_scopes, rename_nodes, tree_id, tree)

    def compute_constant_replacement(self, tree_id, tree: AstTree):
        blacklist = ["number", "string", "boolean", "function"]
        blacklist = ['"' + v + '"' for v in blacklist] + [
            "'" + v + "'" for v in blacklist
        ]
        constants = [
            node
            for node in tree.nodes
            if AdversarialNodeReplacement.is_constant(node)
            and not AdversarialNodeReplacement.property_assignment_left(node)
            and node.value not in blacklist
        ]
        res = []

        # print('-' * 40)
        # print('Tree: {}'.format(tree_id))
        for node in constants:
            pos = NodeRenameRule(
                tree_id,
                node.id,
                [node.id],
                [],
                value="{:3d} {}".format(
                    self.value_index.dataset.VALUES.vocab.stoi[node.value], node.value
                ),
                candidate_values=self.value_index.values_for_type(node.type),
                fixed_value_offset=self.fixed_value_offset,
            )
            res.append(pos)
        return res

    def compute_prop_declaration_renaming(self, tree_id, tree: AstTree):
        rename_nodes = {}
        declarations = [
            node
            for node in tree.nodes
            if AdversarialNodeReplacement.property_declaration(node)
        ]
        decl_scopes = {decl.id: self.analyzer.get_scope(decl) for decl in declarations}

        sample = self.value_index.dataset.get_sample_by_id(tree_id)
        assert sample is not None

        def get_scope_type(node: AstNode):
            if node.type == "ClassExpression":
                raw_type = sample.target[node.id]
            else:
                node = node.down_first()
                while node.type not in ["ClassKeyword", "InterfaceKeyword"]:
                    if not node.has_right():
                        break
                    node = node.right()
                raw_type = sample.target[node.id]
            if "typeof" in raw_type:
                """
                converts
                "'typeof Foo'"
                to 
                ["'typeof Foo'", "Foo"]
                """
                return [raw_type, raw_type.split(" ")[-1]]
            # assert 'typeof' in raw_type, '{}\n{}, raw_type: {}'.format(base_node, node, raw_type)
            return [raw_type]

        for declaration in declarations:
            scope: AstNode = decl_scopes[declaration.id]
            assert scope.type in [
                "ClassDeclaration",
                "InterfaceDeclaration",
                "ClassExpression",
            ], (str(scope) + "\n" + tree.dumpAsString())
            allowed_types = get_scope_type(scope) + ["any"]

            # replace all occurrences that match the class type in the whole file
            scope = tree.root

            rename_nodes[declaration.id] = declaration
            for node in scope.forEachNode():
                if node.value != declaration.value:
                    continue

                if not AdversarialNodeReplacement.property_access(node):
                    continue

                if sample.target[node.left().left().id] not in allowed_types:
                    # Logger.debug('skipping with {} wrong type {} != {} (scope: {})'.format(
                    #     node.id, sample.target[node.left().left().id], allowed_types, scope.id))
                    # input()
                    continue

                rename_nodes[node.id] = declaration

        return self.__process_decl(decl_scopes, rename_nodes, tree_id, tree)

    def compute_variable_renaming(self, tree_id, tree: AstTree):
        rename_nodes = {}
        declarations = [
            node
            for node in tree.nodes
            if self.analyzer.is_declaration(node) and not node.type.endswith("Keyword")
        ]
        decl_scopes = {decl.id: self.analyzer.get_scope(decl) for decl in declarations}
        for declaration in declarations:
            scope: AstNode = decl_scopes[
                declaration.id
            ]  # self.analyzer.get_scope(declaration)

            for node in scope.forEachNode():
                if node.value != declaration.value:
                    continue
                if AdversarialNodeReplacement.property_access(node):
                    continue
                if AdversarialNodeReplacement.property_declaration(node):
                    continue
                """
                all property assignments are strings, even if they have the same name
                e.g. {foo: 5} is equivalent to {'foo': 5}
                regardless of whether variable named foo exists in the scope
                """
                if AdversarialNodeReplacement.property_assignment_left(node):
                    continue

                if self.analyzer.method_decl_name(node):
                    continue

                if node.type.endswith("Keyword"):
                    continue

                rename_nodes[node.id] = declaration

        return self.__process_decl(decl_scopes, rename_nodes, tree_id, tree)

    def __process_decl(self, decl_scopes, rename_nodes, tree_id, tree: AstTree):
        """
        Int -> List[Int]

        Stores for each declaration list of other declarations that are in the same scope
        Each renaming should preserve the invariant the variable names in the same scope
        do not conflict with each other.

        Note that it is possible that the same variable name is declared multiple times, e.g.:

        var x = ...
        ...
        var x = ...

        in this case, we merge the rules into a single rules that always renames both variables into the same value.
        This is a safe option (compared to removing the conflicting constraints) that keeps the program semantics unchanged.
        """
        merged_declarations = {}
        per_scope_decls = collections.defaultdict(set)

        def try_merge_declaration(idx, other_decl_ids):
            for other_decl_id in other_decl_ids:
                if tree.nodes[idx].value == tree.nodes[other_decl_id].value:
                    merged_declarations[idx] = other_decl_id
                    return True
            return False

        for decl_id, scope in decl_scopes.items():
            if not try_merge_declaration(decl_id, per_scope_decls[scope.id]):
                per_scope_decls[scope.id].add(decl_id)

        # print('per_scope_decls', per_scope_decls)
        per_decl_conflits = collections.defaultdict(set)
        for scope_id, decl_ids in per_scope_decls.items():
            for decl_id in decl_ids:
                per_decl_conflits[decl_id].update(decl_ids - {decl_id})

        """
        Note that the declaration can shadow each other in which case rename_nodes 
        can be reassigned to match the inner most scope.
        As a result, the per_decl_usages are computed only at the end

        per_decl_usages: Int -> List[Int]
            Map from a declaration site to all usage positions
        """

        per_decl_usages = collections.defaultdict(set)
        for node_id, declaration in rename_nodes.items():
            # print(node_id, declaration.id)
            merged_declaration_id = merged_declarations.get(
                declaration.id, declaration.id
            )
            per_decl_usages[merged_declaration_id].add(node_id)

        # print('merged_decl', merged_declarations)

        """
        Consistency Check
        """

        def check_consistency():
            for decl_id, conflicts in per_decl_conflits.items():
                assert all(
                    tree.nodes[idx].value != tree.nodes[decl_id].value
                    for idx in conflicts
                ), "conflicts not satisfiable {}: {}".format(
                    decl_id,
                    ",".join(
                        "{}: {}".format(idx, tree.nodes[idx].value) for idx in conflicts
                    ),
                )

            assert all(
                decl_id in usage_ids for decl_id, usage_ids in per_decl_usages.items()
            )
            for decl_id, usage_ids in per_decl_usages.items():
                decl = tree.nodes[decl_id]
                assert all(
                    tree.nodes[uid].value == decl.value
                    and tree.nodes[uid].type == decl.type
                    for uid in usage_ids
                ), "{} vs {}".format(
                    decl, " ".join(str(tree.nodes[uid]) for uid in usage_ids)
                )

            dataset = self.value_index.dataset
            sample = dataset.id_to_sample[tree_id]
            for decl_id, node_ids in per_decl_usages.items():
                if (
                    dataset.TARGET.vocab.stoi[sample.target[decl_id]]
                    == dataset.unk_token_id
                ):
                    continue
                """
                Currently, the cases that fail the following consistency check are either due to:
                  - type refinement
                  - assignment of different type. This is often done by shadowing variable from the same scope. e.g.:
                    def foo(x: int):
                        x = 'test'
                """
                if not all(
                    sample.target[node_id] == sample.target[decl_id]
                    or sample.target[node_id]
                    == "never"  # type refinement in switch statement
                    for node_id in node_ids
                ):
                    # print(tree.root.dumpAsString(label=node_label))
                    # print('{} {}'.format(decl_id, node_ids))
                    # input()
                    # num_inconsistencies += 1
                    pass

        check_consistency()

        res = []
        for decl_id, usage_ids in per_decl_usages.items():
            decl = tree.nodes[decl_id]
            candidate_values = self.value_index.values_for_type(decl.type)
            value = "{:3d} {}".format(
                self.value_index.dataset.VALUES.vocab.stoi[decl.value], decl.value
            )

            pos = NodeRenameRule(
                tree_id,
                decl_id,
                usage_ids,
                per_decl_conflits[decl_id],
                value=value,
                candidate_values=candidate_values,
                decl_id=decl_id,
                fixed_value_offset=self.fixed_value_offset,
            )
            res.append(pos)

        return res

    def make_rules(
        self, dataset: Dataset, trees: Dict[int, AstTree], trees_num: Dict[int, AstTree]
    ):
        num_valid = 0
        num_pos = 0
        index = RenameRulesIndex()
        for tree_id, tree in trees.items():
            sample = dataset.id_to_sample[tree_id]
            rules = RenameRulesForTree(tree_id, trees_num[tree_id])
            try:
                rules.update(self.compute_constant_replacement(tree_id, tree))
                rules.update(self.compute_variable_renaming(tree_id, tree))
                rules.update(self.compute_prop_declaration_renaming(tree_id, tree))
                rules.update(self.compute_property_assignment_renaming(tree_id, tree))
                index.add(rules)
            except Exception as e:
                print(
                    tree.root.dumpAsString(
                        label=lambda node: "{:<20s}".format(
                            trim(sample.target[node.id], 20)
                        )
                    )
                )
                raise e

            num_valid += sum(sample.mask_valid)
            for valid, node in zip(sample.mask_valid, tree.nodes):
                if valid and node.id in rules.nodes_to_rule:
                    num_pos += 1

            def node_label(node: AstNode):
                pos = rules.nodes_to_rule.get(node.id, None)
                s = "{:<20s}".format(trim(sample.target[node.id], 20))
                if pos is not None:
                    return s + "{:>6s} {:6d} {} {}".format(
                        str(pos.decl_id),
                        len(pos.candidate_values),
                        pos.usages,
                        [
                            self.value_index.dataset.VALUES.vocab.itos[c]
                            for c in pos.candidate_values
                        ],
                    )
                return s

        print(
            "{:7d}/{:7d} ({:.2f}%)".format(num_pos, num_valid, acc(num_pos, num_valid))
        )
        return index
