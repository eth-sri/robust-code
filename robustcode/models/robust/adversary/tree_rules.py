import random
from typing import Dict

from robustcode.analysis.graph import AstNode
from robustcode.analysis.graph import AstTree
from robustcode.analysis.graph import TypeScriptGraphAnalyzer
from robustcode.models.robust.adversary.rules import AdversarialNodeReplacement
from robustcode.models.robust.adversary.rules import NodeValueIndexStr
from robustcode.models.robust.dataset import Dataset
from robustcode.util.misc import trim


class PositionIDs:
    ADVERSARIAL_CONSTANT = -2


class ExpressionGenerator:
    def __init__(self, value_index: NodeValueIndexStr):
        self.value_index = value_index

        self.constant_types_to_node_type = {
            "string": ["StringLiteral", "TemplateExpression", "FirstTemplateToken"],
            "boolean": ["TrueKeyword", "FalseKeyword"],
            "number": ["FirstLiteralToken"],
        }

        self.constant_types = list(self.constant_types_to_node_type.keys())

        self.null_id = "<null>"
        self.bin_expr_type = "BinaryExpression"

        self.cond_expr = AstNode(
            idx=-1, type="ConditionalExpression", value=self.null_id
        )
        self.paren_expr = AstNode(
            idx=-1, type="ParenthesizedExpression", value=self.null_id
        )
        self.open_paren_token = AstNode(idx=-1, type="OpenParenToken", value="(")
        self.close_paren_token = AstNode(idx=-1, type="CloseParenToken", value=")")
        self.question_token = AstNode(idx=-1, type="QuestionToken", value="?")
        self.colon_token = AstNode(idx=-1, type="ColonToken", value=":")

    def gen_constant(self, target_type=None, parent=None):
        if target_type is None:
            target_type = random.choice(self.constant_types)

        node_type = random.choice(self.constant_types_to_node_type[target_type])
        node_value = random.choice(self.value_index.per_type_values[node_type])
        astnode = AstNode(
            idx=PositionIDs.ADVERSARIAL_CONSTANT,
            parent=parent,
            type=node_type,
            value=node_value,
        )
        astnode.origin = PositionIDs.ADVERSARIAL_CONSTANT
        return astnode

    def gen_bin_expr(self, depth=0, target_type=None, parent=None):
        if depth == 0:
            return self.gen_constant(target_type=target_type, parent=parent)

        if target_type is None:
            target_type = random.choice(self.constant_types)

        bin_op = AstNode(
            idx=-1, parent=parent, type=self.bin_expr_type, value=self.null_id
        )
        bin_operand_type, bin_operand_value = random.choice(self.value_index.bin_ops)
        if depth == 0:
            bin_op.children = [
                self.gen_constant(target_type=target_type, parent=bin_op),
                AstNode(
                    idx=-1,
                    parent=bin_op,
                    type=bin_operand_type,
                    value=bin_operand_value,
                ),
                self.gen_constant(target_type=target_type, parent=bin_op),
            ]
        else:
            bin_op.children = [
                self.gen_bin_expr(
                    random.randint(0, depth - 1), target_type=target_type, parent=bin_op
                ),
                AstNode(
                    idx=-1,
                    parent=bin_op,
                    type=bin_operand_type,
                    value=bin_operand_value,
                ),
                self.gen_bin_expr(
                    random.randint(0, depth - 1), target_type=target_type, parent=bin_op
                ),
            ]
        return bin_op

    def ternary_expr(
        self, left: AstNode, right: AstNode, depth=0, target_type=None, parent=None
    ):
        root = self.cond_expr.copyNoChildren(parent=parent)

        paren_expr = self.paren_expr.copyNoChildren(parent=root)
        paren_expr.children = [
            self.open_paren_token.copyNoChildren(parent=paren_expr),
            self.gen_bin_expr(depth=depth, target_type=target_type, parent=paren_expr),
            self.close_paren_token.copyNoChildren(parent=paren_expr),
        ]

        left = left.deepCopy(parent=root)
        right = right.deepCopy(parent=root)
        root.children = [
            paren_expr,
            self.question_token.copyNoChildren(parent=root),
            left,
            self.colon_token.copyNoChildren(parent=root),
            right,
        ]
        """
        pick randomly which of left/right should keep target values
        this ensures that the number of nodes to predict and their order is the same for the original and modified tree
        """
        clear_node = random.choice([left, right])
        for node in clear_node.forEachNode():
            node.id = -1
            node.fields.clear()
        return root


class ShuffleDictKeysRule:
    """
    shuffle dictionary keys:

    {
        'x': 1,
        'y': 2,
    }

    ->

    {
        'y': 2,
        'x': 1,
    }
    """

    def __init__(self):
        self.name = "ShuffleDictKeys"
        self.applied_positions = {}

        self.null_id = "<null>"

    def revert_all_changes(self):
        for original_node, swapped_node in self.applied_positions.values():
            swapped_node.swapNodes(original_node)
            for child in original_node.children:
                child.parent = original_node
        self.applied_positions.clear()

    def matches(self, node: AstNode):
        if node.type != "SyntaxList":
            return False
        node = node.left()
        if node is None or node.type != "FirstPunctuation":
            return False
        node = node.up()
        if node is None or node.type != "ObjectLiteralExpression":
            return False
        return True

    def apply(self, tree_id, node: AstNode):
        assert self.matches(node)
        assert node.id != -1
        key = (tree_id, node.id)
        assert key not in self.applied_positions

        assert False, "unsound implementation"
        """
        TODO: the current implementation assumes that structural modifications do not change
        the predictions order. As a result, reordering structural changes will results in wrong evaluation.
        To fix this, we would need to compute a permutation that reorders the predictions in the original order and
        apply it during evaluation.
        """

        properties = [child for child in node.children if child.type != "CommaToken"]
        random.shuffle(properties)

        new_block = AstNode(idx=-1, type="SyntaxList", value=self.null_id)
        for prop in properties:
            if new_block.children:
                comma_token = AstNode(idx=-1, type="CommaToken", value=",")
                new_block.add_child(comma_token)
            new_block.add_child(prop)

        # "remember where the change was applied such that it can be reverted later"
        self.applied_positions[key] = (node, new_block)
        node.swapNodes(new_block)


class AddMethodCallArgumentsRule:
    """
    adds additional unused method call arguments

    console.log("hello world!")

    ->

    console.log("hello world!", expr1, expr2)

    where expr1, expr2 are randomly generated expressions.
    In JavaScript, additional arguments are ignored at runtime.
    Note however this change is not semantic preserving since the method could
    declare additional parameters in which case the generated expressions will be used.

    """

    def __init__(self, expr_gen: ExpressionGenerator):
        self.name = "AddMethodCallArguments"
        self.applied_positions = {}
        self.expr_gen = expr_gen

        self.null_id = "<null>"
        self.analyzer = TypeScriptGraphAnalyzer()

    def revert_all_changes(self):
        for original_node, swapped_node in self.applied_positions.values():
            swapped_node.swapNodes(original_node)
        self.applied_positions.clear()

    def matches(self, node: AstNode):
        return self.analyzer.is_function_def(node)

    """
===================    146       CallExpression	                    	                                     
====================   147 PropertyAccessExpr..	                    	                                     
=====================  148           Identifier	                 div	                 any            div  
=====================  149             DotToken	                   .	                                  .  
=====================  150           Identifier	             getText	                            getText  
====================   151       OpenParenToken	                   (	                                  (  
====================   152           SyntaxList	                    	                                     
====================   153      CloseParenToken	                   )	                                  )

===================    122       CallExpression	                    	                                     
====================   123 PropertyAccessExpr..	                    	                                     
=====================  124           Identifier	                  by	                 any          <unk>  
=====================  125             DotToken	                   .	                                  .  
=====================  126           Identifier	                 css	                                css  
====================   127       OpenParenToken	                   (	                                  (  
====================   128           SyntaxList	                    	                                     
=====================  129        StringLiteral	             'input'	              string        'input'  
====================   130      CloseParenToken	                   )	                                  )
    """

    def apply(self, tree_id, node: AstNode):
        assert self.matches(node)

        def find_args(function_def: AstNode):
            assert function_def.down_first() is not None
            c = function_def.down_first()
            while c.type != "OpenParenToken":
                c = c.right()
            assert c.has_right() and c.right().type == "SyntaxList"
            return c.right()

        args = find_args(node)
        new_args = args.deepCopy(parent=args.parent)
        for i in range(random.randint(1, 2)):
            if new_args.children:
                comma_token = AstNode(idx=-1, type="CommaToken", value=",")
                new_args.add_child(comma_token)
            expr = self.expr_gen.gen_bin_expr(depth=random.randint(0, 2))
            new_args.add_child(expr)

        assert args.id != -1
        key = (tree_id, args.id)
        assert key not in self.applied_positions
        "remember where the change was applied such that it can be reverted later"
        self.applied_positions[key] = (args, new_args)
        args.swapNodes(new_args)


class AddFunctionArgumentRule:

    """
    adds additional unused function arguments

    function foo(x) {
      ...
    }

    ->

    function foo(x, arg1, arg2) {
       ...
    }
    """

    def __init__(self):
        self.name = "AddFunctionArgument"
        self.applied_positions = {}
        self.analyzer = TypeScriptGraphAnalyzer()

        self.null_id = "<null>"

    def revert_all_changes(self):
        for original_node, swapped_node in self.applied_positions.values():
            swapped_node.swapNodes(original_node)
        self.applied_positions.clear()

    def matches(self, node: AstNode):
        return self.analyzer.is_function_def(node)

    def apply(self, tree_id, node: AstNode):
        assert self.matches(node)

        def find_params(function_def: AstNode):
            assert function_def.down_first() is not None
            c = function_def.down_first()
            while c.type != "OpenParenToken":
                c = c.right()
            assert c.has_right() and c.right().type == "SyntaxList"
            return c.right()

        params = find_params(node)
        new_params = params.deepCopy(parent=params.parent)
        for i in range(random.randint(1, 3)):
            param = AstNode(idx=-1, type="Parameter", value=self.null_id)
            # TODO: enable adversarial modifications of the added identifier
            ident = AstNode(idx=-1, type="Identifier", value="param{}".format(i))
            param.add_child(ident)

            if new_params.children:
                comma_token = AstNode(idx=-1, type="CommaToken", value=",")
                new_params.add_child(comma_token)
            new_params.add_child(param)

        assert params.id != -1
        key = (tree_id, params.id)
        assert key not in self.applied_positions
        "remember where the change was applied such that it can be reverted later"
        self.applied_positions[key] = (params, new_params)
        params.swapNodes(new_params)


class AddExpressionStatementRule:

    """
    adds side-effect free expression statement

    """

    def __init__(self, expr_gen: ExpressionGenerator):
        self.name = "AddExpressionStatement"
        """
         we need to remove modifications in the reverse order they were added
         given a code, the new statement is added either before of after the existing one. E.g,:
         
         stmt1         
         
         ->
         
         stmt1
         expr1
         
         or
         
         stmt1         
         
         ->
         
         expr1
         stmt1          
         
         where stmt1 is the original statement and expr1 is newly generated
         
        """
        self.applied_modifications = []
        self.applied_positions = set()
        self.analyzer = TypeScriptGraphAnalyzer()

        self.null_id = "<null>"

        self.expr_gen = expr_gen

    def revert_all_changes(self):
        for original_node, right_sibling_node, is_after in reversed(
            self.applied_modifications
        ):
            if is_after:
                original_node.parent.remove_child(original_node.pos_in_parent() + 1)
            else:
                original_node.parent.remove_child(original_node.pos_in_parent() - 1)
        self.applied_modifications.clear()
        self.applied_positions.clear()

    def matches(self, node: AstNode):
        if not node.type.endswith("Statement"):
            return False
        block = node.parent
        return (
            block.type == "SyntaxList"
            and block.parent is not None
            and block.parent.type in ["Block", "SourceFile"]
        )

    def apply(self, tree_id, node: AstNode):
        assert self.matches(node)

        # whether the expression is added before or after
        is_after = random.choice([True, False])

        root = AstNode(idx=-1, type="ExpressionStatement", value=self.null_id)
        expr = self.expr_gen.gen_bin_expr(depth=random.randint(0, 2))
        colon = AstNode(idx=-1, type="SemicolonToken", value=";")
        root.add_child(expr)
        root.add_child(colon)

        assert node.id != -1
        key = (tree_id, node.id)
        assert key not in self.applied_positions
        # "remember where the change was applied such that it can be reverted later"
        self.applied_modifications.append((node, root, is_after))
        node.parent.add_child(root, pos=node.pos_in_parent() + 1 * is_after)


class AddObjExpressionRule:
    """
    adds side-effect objection expression statement

    {x: y}

    """

    def __init__(self, expr_gen: ExpressionGenerator):
        self.name = "AddObjExpressionStatement"
        self.applied_modifications = []
        self.applied_positions = set()
        self.analyzer = TypeScriptGraphAnalyzer()

        self.null_id = "<null>"

        self.expr_gen = expr_gen

    def revert_all_changes(self):
        for original_node, right_sibling_node, is_after in reversed(
            self.applied_modifications
        ):
            if is_after:
                original_node.parent.remove_child(original_node.pos_in_parent() + 1)
            else:
                original_node.parent.remove_child(original_node.pos_in_parent() - 1)
        self.applied_modifications.clear()
        self.applied_positions.clear()

    def matches(self, node: AstNode):
        if not node.type.endswith("Statement"):
            return False
        block = node.parent
        return (
            block.type == "SyntaxList"
            and block.parent is not None
            and block.parent.type in ["Block", "SourceFile"]
        )

    def apply(self, tree_id, node: AstNode):
        assert self.matches(node)

        # whether the expression is added before or after
        is_after = random.choice([True, False])

        values = set()
        for n in node.parent.forEachNode():
            if n.type == "Identifier":
                values.add(n.value)
        values = list(values)

        root = AstNode(idx=-1, type="ExpressionStatement", value=self.null_id)
        obj = AstNode(idx=-1, type="ObjectLiteralExpression", value=self.null_id)
        root.add_child(obj)
        n = AstNode(idx=-1, type="FirstPunctuation", value="{")
        obj.add_child(n)
        block = AstNode(idx=-1, type="SyntaxList", value=self.null_id)
        obj.add_child(block)
        for i in range(random.randint(1, 5)):
            if i != 0:
                n = AstNode(idx=-1, type="CommaToken", value=",")
                block.add_child(n)

            prop = AstNode(idx=-1, type="PropertyAssignment", value=self.null_id)
            if not values or random.random() > 0.5:
                n = self.expr_gen.gen_constant()
                prop.add_child(n)
            else:
                n = AstNode(idx=-1, type="Identifier", value=random.choice(values))
                prop.add_child(n)
            n = AstNode(idx=-1, type="ColonToken", value=":")
            prop.add_child(n)
            if not values or random.random() > 0.5:
                n = self.expr_gen.gen_constant()
                prop.add_child(n)
            else:
                n = AstNode(idx=-1, type="Identifier", value=random.choice(values))
                prop.add_child(n)
            block.add_child(prop)

        n = AstNode(idx=-1, type="CloseBraceToken", value="}")
        block.add_child(n)

        colon = AstNode(idx=-1, type="SemicolonToken", value=";")
        root.add_child(colon)

        assert node.id != -1
        key = (tree_id, node.id)
        assert key not in self.applied_positions
        # "remember where the change was applied such that it can be reverted later"
        self.applied_modifications.append((node, root, is_after))
        node.parent.add_child(root, pos=node.pos_in_parent() + 1 * is_after)


class TernaryWrapperRule:

    """
    wraps an expresion x into
    (bin_expr) ? x : x
    where bin_expr is a random boolean expression
    """

    """                   
                   313     ConditionalExpression                                            string ['TernaryWrapperRule']
                    314    ParenthesizedExpression                                         boolean []
                     315   OpenParenToken                 (                                        []
                     316   BinaryExpression                                                boolean ['TernaryWrapperRule']
                      317  Identifier                     value                                any ['TernaryWrapperRule']
                      318  FirstBinaryOperator            <                                        []
                      319  FirstLiteralToken              0                                 number ['TernaryWrapperRule']
                     320   CloseParenToken                )                                        []
                    321    QuestionToken                  ?                                        []
                    322    StringLiteral                  'red'                             string ['TernaryWrapperRule']
                    323    ColonToken                     :                                        []
                    324    StringLiteral                  'green'                           string ['TernaryWrapperRule']
    
    """

    def __init__(self, expr_gen: ExpressionGenerator):
        self.name = "TernaryWrapperRule"
        self.applied_positions = {}
        self.expr_gen = expr_gen

        # self.type_counts = collections.Counter()

    def revert_all_changes(self):
        for original_node, swapped_node in self.applied_positions.values():
            swapped_node.swapNodes(original_node)
        self.applied_positions.clear()

    def apply(self, tree_id, node: AstNode):
        # assert self.matches(node)
        assert node.id != -1
        key = (tree_id, node.id)
        assert key not in self.applied_positions

        cond_depth = random.randint(0, 3)
        expr = self.expr_gen.ternary_expr(node, node, cond_depth)
        "remember where the change was applied such that it can be reverted later"
        self.applied_positions[key] = (node, expr)
        node.swapNodes(expr)
        return expr

    def matches(self, node: AstNode):
        # is_null = str(node.fields.get('target', '<null>') == '<null>')
        # if node.fields.get('target', '<null>') != '<null>':
        #     return False

        if AdversarialNodeReplacement.is_constant(node):
            # self.type_counts['constant' + is_null] += 1
            return True
        if node.type == "PropertyAccessExpression":
            # self.type_counts['PropertyAccessExpression' + is_null] += 1
            return True

        parent = node.up()
        if parent is not None:
            pos = node.pos_in_parent()
            "x.y -> ((...) ? x : x).y"
            if pos == 0 and parent.type == "PropertyAccessExpression":
                # self.type_counts['PropertyAccessExpression v2' + is_null] += 1
                return True

            " x + y -> ((...) ? x : x) + y"
            if pos != 1 and parent.type == "BinaryExpression":
                # self.type_counts['BinaryExpression v2' + is_null] += 1
                return True

            if node.type == "BinaryExpression" and parent.type != "ExpressionStatement":
                # self.type_counts['BinaryExpression' + is_null] += 1
                return True

            "{ y : x }"
            if pos == 2 and parent.type == "PropertyAssignment":
                # self.type_counts['PropertyAssignment' + is_null] += 1
                return True

        "return x -> return (...) ? x : x"
        if node.has_left() and node.left().type in ["ReturnKeyword", "FirstAssignment"]:
            # self.type_counts['left' + is_null] += 1
            return True

        return False


class ArrayAccessWrapperRule(TernaryWrapperRule):
    """
    wraps an expression x into
    [x, x, x][1]

    a possible extension is to replace some of x with a different expression of the same type

   24         ElementAccessExpression                            ' '
    25        ArrayLiteralExpression                             any[]
     26       OpenBracketToken               [                   ' '
     27       SyntaxList                                         ' '
      28      Identifier                     a                   any
     29       CloseBracketToken              ]                   ' '
    30        OpenBracketToken               [                   ' '
    31        FirstLiteralToken              0                   number
    32        CloseBracketToken              ]                   ' '
    """

    def __init__(self, expr_gen: ExpressionGenerator):
        super(ArrayAccessWrapperRule, self).__init__(expr_gen)
        self.name = "ArrayAccessWrapper"

        self.null_id = "<null>"

    def gen_array_expression(self, node, num_elem, select_idx):
        assert 0 <= select_idx < num_elem
        elem_access = AstNode(
            idx=-1, type="ElementAccessExpression", value=self.null_id
        )
        array_lit = AstNode(idx=-1, type="ArrayLiteralExpression", value=self.null_id)

        syntax_list = AstNode(idx=-1, type="SyntaxList", value=self.null_id)
        for idx in range(num_elem):
            array_node = node.deepCopy()
            syntax_list.add_child(array_node)

            if idx != select_idx:
                """
                this ensures that the number of nodes to predict and their order 
                is the same for the original and modified tree
                """
                for n in array_node.forEachNode():
                    n.id = -1
                    n.fields.clear()

        array_lit.add_child(AstNode(idx=-1, type="OpenBracketToken", value="["))
        array_lit.add_child(syntax_list)
        array_lit.add_child(AstNode(idx=-1, type="CloseBracketToken", value="]"))
        elem_access.add_child(array_lit)
        elem_access.add_child(AstNode(idx=-1, type="OpenBracketToken", value="["))
        elem_access.add_child(
            AstNode(idx=-1, type="FirstLiteralToken", value=select_idx)
        )
        elem_access.add_child(AstNode(idx=-1, type="CloseBracketToken", value="]"))
        return elem_access

    def apply(self, tree_id, node: AstNode):
        # assert self.matches(node)
        assert node.id != -1
        key = (tree_id, node.id)
        assert key not in self.applied_positions

        size = random.randint(1, 4)
        expr = self.gen_array_expression(node, size, random.randint(0, size - 1))
        "remember where the change was applied such that it can be reverted later"
        self.applied_positions[key] = (node, expr)
        node.swapNodes(expr)


class AdversarialSubtreeReplacement:
    def __init__(self, expr_gen: ExpressionGenerator):
        self.analyzer = TypeScriptGraphAnalyzer()
        self.expr_gen = expr_gen

    def make_rules(
        self, dataset: Dataset, trees: Dict[int, AstTree], trees_num: Dict[int, AstTree]
    ):
        rules = [
            TernaryWrapperRule(self.expr_gen),
            AddFunctionArgumentRule(),
            AddMethodCallArgumentsRule(self.expr_gen),
            ArrayAccessWrapperRule(self.expr_gen),
            AddExpressionStatementRule(self.expr_gen),
            AddObjExpressionRule(self.expr_gen),
        ]

        def visualize():
            for tree_id, tree in trees.items():
                sample = dataset.id_to_sample[tree_id]

                if not any(node.value == "?" for node in tree.nodes):
                    continue

                def node_label(n: AstNode):
                    valid_rules = [rule for rule in rules if rule.matches(n)]
                    return "{:>20s} {}".format(
                        trim(sample.target[n.id], 20),
                        str([rule.name for rule in valid_rules]),
                    )

                print(tree.root.dumpAsString(label=node_label))

                num_changes = 0
                for node in tree.nodes:
                    for rule in rules:
                        if rule.matches(node):  # and random.randint(0, 20) == 0:
                            rule.apply(tree_id, node)
                            num_changes += 1
                            print(tree.root.dumpAsString(label=node_label))
                            input()

                print("done {} changes".format(num_changes))
                print("reverting changes")
                input()
                rules[0].revert_all_changes()
                print(tree.root.dumpAsString(label=node_label))
                input()

        # visualize()

        return rules
