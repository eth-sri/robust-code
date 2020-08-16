import collections
import copy
import io
import itertools
import math
import sys
import traceback
from abc import ABC
from abc import abstractmethod
from typing import Iterable
from typing import Optional
from typing import Sequence

from robustcode.util.misc import trim


class GraphAnalyzer(ABC):
    @abstractmethod
    def function_defs(self, tree):
        pass

    @abstractmethod
    def method_calls(self, tree):
        pass

    @abstractmethod
    def function_arguments(self, tree):
        pass

    """
    Returns [(lhs, rhs)] for all assignments in the tree
    """

    @abstractmethod
    def assignments(self, tree):
        pass

    """
    Returns [(return, function_decl)] where function_decl is the function declaration associated with return
    """

    @abstractmethod
    def returns(self, tree):
        pass

    """
    Return true if the node is a parameter of a method declaration.
    
    def foo(x, y):
        ...
        
    should return true for x and y
    """

    @abstractmethod
    def is_parameter(self, node):
        pass

    @abstractmethod
    def method_decl_name(self, node):
        pass

    """
    Returns the scope in which given variable or parameter is defined.
    
    def foo(x):
        ...
        y = 5;
        ...
        
    should return node corresponding to the method declaration for both x, y.
    Typically this would be implemented only approximately (e.g., not modelling semantics of let vs var)
    """

    @abstractmethod
    def get_scope(self, node):
        pass

    @abstractmethod
    def is_variable(self, tree):
        pass

    """
    Return true if the statement declares a new variable
    """

    @abstractmethod
    def is_declaration(self, node):
        pass

    @abstractmethod
    def is_class_property(self, node):
        pass

    @abstractmethod
    def is_variable_read(self, tree):
        pass

    @abstractmethod
    def is_variable_write(self, tree):
        pass

    """
    Returns [(guard, true_block, else_block | None)]
    """

    @abstractmethod
    def guards(self, tree):
        pass

    def variables(self, root: "AstNode"):
        for node in root.forEachNode():
            if self.is_variable(node):
                yield node


class TypeScriptGraphAnalyzer(GraphAnalyzer):
    def __init__(self):
        super(TypeScriptGraphAnalyzer, self).__init__()

    def function_defs(self, tree: "AstTree"):
        return [node for node in tree.nodes if self.is_function_def(node)]

    """
      2                FunctionDeclaration                                '(x: number, y: number) => number'                          
       3               FunctionKeyword                function            '(x: number, y: number) => number'                          
       4               Identifier                     add                 '(x: number, y: number) => number'                          
       5               OpenParenToken                 (                   ' '                                                         
       6               SyntaxList                                         ' '                                                         
        7              Parameter                                          number                                                      
         8             Identifier                     x                   number                                                                                                                 
        11             CommaToken                     ,                   ' '                                                         
        12             Parameter                                          number                                                      
         13            Identifier                     y                   number      
         
          83           FunctionExpression                                 '(x: number, y: number) => number'                          
           84          FunctionKeyword                function            '(x: number, y: number) => number'                          
           85          OpenParenToken                 (                   ' '                                                         
           86          SyntaxList                                         ' '                                                         
            87         Parameter                                          number                                                      
             88        Identifier                     x                   number                                                                                                                     
            91         CommaToken                     ,                   ' '                                                         
            92         Parameter                                          number                                                      
             93        Identifier                     y                   number                                                                                                                   
           96          CloseParenToken                )                   ' '                                                         
    
        147            MethodSignature                                    '(onclick: (this: any[], e: Event[]) => void) => void'      
         148           Identifier                     addClickListener    '(onclick: (this: any[], e: Event[]) => void) => void'      
         149           OpenParenToken                 (                   ' '                                                         
         150           SyntaxList                                         ' '                                                         
          151          Parameter                                          '(this: any[], e: Event[]) => void'                         
           152         Identifier                     onclick             '(this: any[], e: Event[]) => void'
           
        192            MethodDeclaration                                  '(this: void, e: Event) => void'                            
         193           Identifier                     onClickGood         '(this: void, e: Event) => void'                            
         194           OpenParenToken                 (                   ' '                                                         
         195           SyntaxList                                         ' '                                                         
          196          Parameter                                          void                                                        
           197         Identifier                     this                void                                                                                                                       
          200          CommaToken                     ,                   ' '                                                         
          201          Parameter                                          Event                                                       
           202         Identifier                     e                   Event 
           
        491            MethodDeclaration                                  '() => void'                                                
         492           SyntaxList                                         ' '                                                         
          493          FirstContextualKeyword         abstract            ' '                                                         
         494           Identifier                     makeSound           '() => void'                                                
         495           OpenParenToken                 (                   ' '                                                         
         496           SyntaxList                                         ' '                                                         
         497           CloseParenToken                )                   ' '                                                         
         498           ColonToken                     :                   ' '                                                         
         499           VoidKeyword                    void                void          
           
        432            Constructor                                        ' '                                                         
         433           ConstructorKeyword             constructor         'typeof Animal'                                             
         434           OpenParenToken                 (                   ' '                                                         
         435           SyntaxList                                         ' '                                                         
          436          Parameter                                          string                                                      
           437         Identifier                     theName             string                                      
    """

    def is_function_def(self, node: "AstNode"):
        return node.type in [
            "MethodDeclaration",
            "MethodSignature",
            "FunctionExpression",
            "FunctionDeclaration",
            "Constructor",
        ]

    def method_calls(self, tree: "AstTree"):
        return [node for node in tree.nodes if self.is_method_call(node)]

    def is_method_call(self, node: "AstNode"):
        return node.type == "CallExpression"

    def function_arguments(self, tree):
        assert False

    def method_decl_name(self, node: "AstNode"):
        return (
            node.type == "Identifier"
            and node.has_up()
            and node.up().type
            in [
                "MethodDeclaration",
                "ClassDeclaration",
                "InterfaceDeclaration",
                "FunctionDeclaration",
                "MethodSignature",
                "ArrowFunction",
                "ClassExpression",
                "FunctionExpression",
                "SetAccessor",
                "GetAccessor",
                "FunctionType",
            ]
        )

    def get_scope(self, node: "AstNode"):
        cur = node
        while cur.has_up():
            cur = cur.up()
            if cur.type in [
                "MethodDeclaration",
                "ClassDeclaration",
                "InterfaceDeclaration",
                "FunctionDeclaration",
                "MethodSignature",
                "ArrowFunction",
                "ClassExpression",
                "FunctionExpression",
                "SetAccessor",
                "GetAccessor",
                "FunctionType",
            ]:
                break
        return cur

    """
        894                                 MethodDeclaration                                  
         895                                SyntaxList                                         
          896                               PublicKeyword                  public              
          897                               AsyncKeyword                   async               
         898                                Identifier                     parse               
         899                                OpenParenToken                 (                   
         900                                SyntaxList                                         
          901                               Parameter                                          
           902                              Identifier                     document             902                  Uri document
          903                               CommaToken                     ,                   
          904                               Parameter                                          
           905                              Identifier                     source               905               string source
         906                                CloseParenToken                ) 
         
        1517             Identifier                     values              1485          CSSRecord[] values
        1518             DotToken                       .                   
        1519             Identifier                     map                 
       1520              OpenParenToken                 (                   
       1521              SyntaxList                                         
        1522             ArrowFunction                                      
         1523            SyntaxList                                         
          1524           Parameter                                          
           1525          Identifier                     property            1455            CSSRecord property
         1526            EqualsGreaterThanToken         =>                  
         1527            Block         
         
    # ignore params as shown below. Not clear why are they marked as Params 
         871                     OpenBracketToken               [                   
         872                     SyntaxList                                         
          873                    Parameter                                          
           874                   Identifier                     key                  874               string key
         875                     CloseBracketToken              ]
    """

    def is_parameter(self, node: "AstNode"):
        return (
            node.type == "Identifier"
            and node.has_up()
            and node.up().type == "Parameter"
            and node.up().has_up()
            and node.up().up().type == "SyntaxList"
            and (
                (
                    node.up().up().has_left()
                    and node.up().up().left().type == "OpenParenToken"
                )
                or (
                    node.up().up().has_up()
                    and node.up().up().up().type in ["ArrowFunction", "SetAccessor"]
                )
            )
        )

    def is_variable(self, node: "AstNode"):
        return node.type == "Identifier"

    def is_declaration(self, node: "AstNode"):
        "foo(x)"
        if self.is_parameter(node):
            return True
        "x = ..."
        if (
            node.is_first_child()
            and node.type == "Identifier"
            and node.has_up()
            and node.up().type == "VariableDeclaration"
        ):
            return True
        "import * as x from ..."
        if (
            node.type == "Identifier"
            and node.has_left()
            and node.left().type == "AsKeyword"
        ):
            return True

        # "{x: foo}"
        # if node.up() is not None and node.up().type == 'PropertyAssignment' and node.pos_in_parent() == 0:
        #     return True
        return False

    def is_class_property(self, node: "AstNode"):
        if (
            node.type != "Identifier"
            or not node.has_up()
            or not node.up().type == "PropertyDeclaration"
        ):
            return False

        while node.has_left():
            node = node.left()
            if node.type != "SyntaxList":
                return False
        return True

    def is_variable_read(self, node: "AstNode"):
        return not self.is_variable_write(node)

    def is_variable_write(self, node: "AstNode"):
        "x      = foo + x"
        if node.has_right() and node.right().type.endswith("Assignment"):
            return True

        if node.is_first_child() and node.has_up():
            node = node.up()
            "{x: foo}"
            if node.type == "PropertyAssignment":
                return True

            "x[i++] = foo + x"
            "x.foo = y"
            if node.has_right() and node.right().type.endswith("Assignment"):
                return True
        return False

    def assignments(self, tree: "AstTree"):
        res = []
        for node in tree.nodes:
            if not node.type.endswith("Assignment") or node.type in [
                "SpreadAssignment",
                "ExportAssignment",
            ]:
                continue

            if node.type == "PropertyAssignment":
                # assert len(node.children) == 3, '{}\n{}'.format(node.id, tree.dumpAsString())
                if len(node.children) == 3:
                    res.append((node.down_first(), node.down_last()))
            elif node.type == "ShorthandPropertyAssignment":
                if node.down_first() is None:
                    continue
                res.append((node.down_first(), node.down_first()))
            else:
                # assert node.left() is not None, '{}\n{}'.format(node.id, tree.dumpAsString())
                # assert node.right() is not None, '{}\n{}'.format(node.id, tree.dumpAsString())
                if not node.has_left() or not node.has_right():
                    continue
                res.append((node.left(), node.right()))
        return res

    def returns(self, tree: "AstTree"):
        res = []
        for node in tree.nodes:
            if node.type != "ReturnStatement":
                continue

            fnc_def = node
            while fnc_def is not None and fnc_def.type not in [
                "FunctionDeclaration",
                "FunctionExpression",
                "ArrowFunction",
            ]:
                fnc_def = fnc_def.up()
            if fnc_def is not None:
                res.append((node, fnc_def))
            else:
                # possible for top level returns
                # sys.stderr.write('Unable to find function declaration for: ' + str(node.id) + '\n')
                pass
        return res

    def guards(self, tree: "AstTree"):
        res = []
        for node in tree.nodes:
            if node.type == "IfStatement":
                if node.down_first() is None or node.down_first().right() is None:
                    continue
                condition = node.down_first().right().right()
                if condition is None or condition.right() is None:
                    continue

                block_true = condition.right().right()
                if block_true is None:
                    continue
                block_false = block_true.right()
                if block_false is not None:
                    block_false = block_false.right()

                res.append((condition, block_true, block_false))
            elif node.type == "WhileStatement":
                if node.down_first() is None or node.down_first().right() is None:
                    continue
                condition = node.down_first().right().right()
                if (
                    condition is None
                    or condition.right() is None
                    or condition.right().right() is None
                ):
                    continue
                block = condition.right().right()
                res.append((condition, block, None))
        return res


class AstTree:
    edge_types = [
        "child_edges",
        "next_token_edges",
        "last_lexical_usage_edges",
        "computed_from_edges",
        "returns_to_edges",
        "guard_by_edges",
        "guard_by_negation_edges",
        "last_write_edges",
        "last_read_edges",
    ]

    @staticmethod
    def fromJson(json_root, analyzer=None, field_names: Optional[Iterable[str]] = None):
        return AstTree(
            AstNode.fromJson(json_root, field_names=field_names), analyzer=analyzer
        )

    @staticmethod
    def fromTensor(types, values, depths, fields={}):
        nodes = [
            AstNode(
                idx,
                t,
                v,
                fields={
                    key: values[idx]
                    for key, values in fields.items()
                    if values[idx] != "<null>"
                },
            )
            for idx, t, v in zip(itertools.count(), types, values)
        ]
        parents = []
        for idx, depth in zip(itertools.count(1), depths[1:]):
            if depth > depths[idx - 1]:
                parents.append(nodes[idx - 1])
            while parents and depth <= depths[parents[-1].id]:
                parents.pop()
            if parents:
                nodes[idx].parent = parents[-1]
                parents[-1].children.append(nodes[idx])

        return AstTree(nodes[0], nodes=nodes)

    def number_nodes(self):
        for idx, node in enumerate(self.nodes):
            node.id = idx

    def refresh(self):
        """
        Reloads the list of nodes and renumbers them.
        Used to reflect changes after adversarial attacks.
        """
        self.nodes = list(self.root.forEachNode())
        self.number_nodes()

    def __len__(self):
        return len(self.nodes)

    def __init__(self, root, analyzer=None, nodes=None):
        self.root: AstNode = root
        self.nodes: Sequence[AstNode] = nodes if nodes is not None else list(
            root.forEachNode()
        )
        self.analyzer = analyzer

    def dumpFieldsAsString(self, fields):
        return self.root.dumpAsString(
            label=lambda node: " ".join(
                "{:<60s}".format(trim(node.fields.get(field, " "), 60))
                for field in fields
            )
        )

    def dumpAsString(self):
        if not self.analyzer:
            return self.root.dumpAsString()

        per_node_edges = [{key: [] for key in AstTree.edge_types} for _ in self.nodes]
        for edge_type in AstTree.edge_types:
            edges = getattr(self, edge_type)()
            for src, tgt in edges:
                per_node_edges[src][edge_type].append(tgt)

        return (
            " ".join(AstTree.edge_types)
            + "\n"
            + self.root.dumpAsString(
                label=lambda node: " ".join(
                    "{:<12s}".format(trim(per_node_edges[node.id][edge_type], 12))
                    for edge_type in AstTree.edge_types
                )
            )
        )

    def compute_all_edges(self):
        assert self.analyzer is not None
        return {
            edge_type: getattr(self, edge_type)() for edge_type in AstTree.edge_types
        }

    """
    Edges connecting child to parent node
    """

    def child_edges(self):
        edges = []
        for node in self.nodes:
            edges.extend([(node.id, child.id) for child in node.children])
        return edges

    """
    Edges connecting pairs of terminal nodes visited in pre-order traversal 
    """

    def next_token_edges(self):
        edges = []
        last_leaf = None
        for node in self.nodes:
            if not node.children:
                if last_leaf is not None:
                    edges.append((last_leaf.id, node.id))
                last_leaf = node
        return edges

    """
    Edges connecting variables to their last lexical usage
    """

    def last_lexical_usage_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        last_usages = {}
        for node in self.nodes:
            if not self.analyzer.is_variable(node):
                continue
            if node.value in last_usages:
                edges.append((last_usages[node.value].id, node.id))
            last_usages[node.value] = node
        return edges

    def computed_from_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        assignments = self.analyzer.assignments(self)
        for lhs, rhs in assignments:
            lhs_vars = list(self.analyzer.variables(lhs))[:5]
            rhs_vars = list(self.analyzer.variables(rhs))[:10]
            for src, tgt in itertools.product(lhs_vars, rhs_vars):
                edges.append((src.id, tgt.id))

        return edges

    def returns_to_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        returns = self.analyzer.returns(self)
        for ret, fnc_def in returns:
            edges.append((fnc_def.id, ret.id))

        return edges

    def guard_by_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        guards = self.analyzer.guards(self)
        for condition, block_true, _ in guards:
            condition_vars_names = set(
                node.value for node in self.analyzer.variables(condition)
            )

            for node in self.analyzer.variables(block_true):
                if node.value in condition_vars_names:
                    edges.append((condition.id, node.id))

        return edges

    def guard_by_negation_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        guards = self.analyzer.guards(self)
        for condition, _, block_false in guards:
            condition_vars_names = set(
                node.value for node in self.analyzer.variables(condition)
            )

            if block_false is None:
                continue

            for node in self.analyzer.variables(block_false):
                if node.value in condition_vars_names:
                    edges.append((condition.id, node.id))

        return edges

    def last_read_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        last_reads = {}
        for node in self.nodes:
            if not self.analyzer.is_variable(node):
                continue
            if not self.analyzer.is_variable_read(node):
                continue

            if node.value in last_reads:
                edges.append((last_reads[node.value].id, node.id))
            last_reads[node.value] = node

        return edges

    def last_write_edges(self):
        edges = []
        if self.analyzer is None:
            return edges

        last_writes = {}
        for node in self.nodes:
            if not self.analyzer.is_variable(node):
                continue
            if not self.analyzer.is_variable_write(node):
                continue

            if node.value in last_writes:
                edges.append((last_writes[node.value].id, node.id))
            last_writes[node.value] = node

        return edges


class SwapNodes:
    def __init__(self, src_node, tgt_node):
        self.src_node = src_node
        self.tgt_node = tgt_node

    def __enter__(self):
        self.src_node.swapNodes(self.tgt_node)
        return self.tgt_node

    def __exit__(self, type, value, traceback):
        self.tgt_node.swapNodes(self.src_node)


class AstNode:
    @staticmethod
    def getType(json_node):
        assert "type" in json_node
        return json_node["type"]

    @staticmethod
    def getValue(json_node):
        if "value" in json_node:
            return json_node["value"]
        return None

    @staticmethod
    def getChildren(json_root, json_node, parent, field_names):
        if "children" not in json_node:
            return []
        assert all(child_id > parent.id for child_id in json_node["children"])
        return [
            AstNode.__fromJson(json_root, child_id, parent, field_names)
            for child_id in json_node["children"]
        ]

    @staticmethod
    def getFields(json_node, fields: Optional[Iterable[str]]):
        if fields is None:
            return {}
        return {field: json_node[field] for field in fields if field in json_node}

    @staticmethod
    def __fromJson(json_root, idx, parent, field_names: Optional[Iterable[str]]):
        assert 0 <= idx < len(json_root)
        json_node = json_root[idx]
        node = AstNode(
            idx,
            AstNode.getType(json_node),
            AstNode.getValue(json_node),
            parent=parent,
            fields=AstNode.getFields(json_node, field_names),
        )
        node.children = AstNode.getChildren(json_root, json_node, node, field_names)
        return node

    @staticmethod
    def fromJson(json_root, field_names: Optional[Iterable[str]] = None):
        assert len(json_root) > 0
        return AstNode.__fromJson(json_root, 0, None, field_names)

    @staticmethod
    def fromTensor(types, values, depths):
        tree = AstTree.fromTensor(types, values, depths)
        return tree.root

    def __init__(self, idx, type, value=None, children=None, parent=None, fields=None):
        self.id = idx
        self.type = type
        self.value = value
        self.children = children if children is not None else []
        self.parent = parent
        self.fields = fields if fields is not None else {}
        self.origin = 0  # "how did this node get here?" - used in nested attacks

    def forEachNode(self):
        # implemented without recursion to avoid stack size limits for deep trees
        stack = collections.deque([self])
        while stack:
            node = stack.pop()
            yield node
            stack.extend(child for child in reversed(node.children))

    def __len__(self):
        # implemented without recursion to avoid stack size limits for deep trees
        size = 0
        stack = collections.deque([self])
        while stack:
            size += 1
            node = stack.pop()
            yield node
            stack.extend(child for child in reversed(node.children))
        return size

    def add_child(self, node, pos=None):
        if pos is None:
            self.children.append(node)
        else:
            self.children.insert(pos, node)
        node.parent = self

    def remove_child(self, pos):
        del self.children[pos]

    def has_up(self) -> bool:
        return self.parent is not None

    def up(self) -> Optional["AstNode"]:
        return self.parent

    def down_first(self) -> Optional["AstNode"]:
        if not self.children:
            return None
        return self.children[0]

    def down_last(self) -> Optional["AstNode"]:
        if not self.children:
            return None
        return self.children[-1]

    def right(self) -> Optional["AstNode"]:
        if self.parent is None:
            return None

        i = self.pos_in_parent()
        if i < len(self.parent.children) - 1:
            return self.parent.children[i + 1]

        return None

    def has_right(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.children[-1].id != self.id

    def is_last_child(self) -> bool:
        return not self.has_right()

    def left(self) -> Optional["AstNode"]:
        if self.parent is None:
            return None

        i = self.pos_in_parent()
        if i > 0:
            return self.parent.children[i - 1]

        return None

    def has_left(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.children[0].id != self.id

    def is_first_child(self) -> bool:
        return not self.has_left()

    def copyNoChildren(self, parent=None):
        return AstNode(
            self.id,
            self.type,
            self.value,
            parent=parent if parent is not None else self.parent,
        )

    def deepCopy(self, parent=None):
        res = AstNode(
            self.id, self.type, self.value, parent=parent, fields=copy.copy(self.fields)
        )
        res.children = [node.deepCopy(parent=res) for node in self.children]
        return res

    def swapNodes(self, other):
        assert self.parent is not None, "Unable to swap root"
        other.parent = self.parent
        for idx, child in enumerate(self.parent.children):
            if child == self:
                self.parent.children[idx] = other
                break
        self.parent = None

    def get_value(self) -> str:
        """
        An empty node values can be denoted in two ways:
          - using None if the tree is parsed from json
          - using '<null>' if the tree is parsed from a batch
        """
        return self.value if self.value else "<null>"

    def has_value(self) -> bool:
        return self.value is not None and self.value != "<null>"

    def node_equal(self, other: "AstNode"):
        return (
            self.type == other.type
            and self.get_value() == other.get_value()
            and self.fields == other.fields
            and len(self.children) == len(other.children)
        )

    def num_tree_diffs(self, other: "AstNode"):
        diff = 0 if self.node_equal(other) else 1
        return diff + sum(
            [
                child.num_tree_diffs(other_child)
                for child, other_child in zip(self.children, other.children)
            ]
        )

    def tree_equal(self, other, verbose=False):
        if verbose and not self.node_equal(other):
            sys.stderr.write(
                '\n\t\t%d, %s vs %s, type: %s, value: %s ("%s" == "%s"), children: %s, fields %s vs %s\n'
                % (
                    self.id,
                    self,
                    other,
                    self.type == other.type,
                    self.get_value() == other.get_value(),
                    self.get_value(),
                    other.get_value(),
                    len(self.children) == len(other.children),
                    self.fields,
                    other.fields,
                )
            )
        return self.node_equal(other) and all(
            [
                child.tree_equal(other_child, verbose=verbose)
                for child, other_child in zip(self.children, other.children)
            ]
        )

    def depth(self):
        res = 0
        node = self
        while node.parent is not None:
            res += 1
            node = node.parent
        return res

    def max_depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.max_depth() for child in self.children)

    def pos_in_parent(self) -> int:
        if self.parent is None:
            return 0
        """
        It is possible that id == -1 if the tree is being modified
        call AstTree.number_nodes() to reassign ids. 
        """
        assert self.id != -1

        for i, node in enumerate(self.parent.children):
            if node.id == self.id:
                return i
        assert False

    def get_root(self):
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def __str__(self):
        if self.value:
            return "type: {}, value: {}".format(self.type, self.value)
        return "type: {}".format(self.type)

    def dumpFieldsAsString(self, fields):
        return self.dumpAsString(
            label=lambda node: " ".join(
                "{:<60s}".format(trim(node.fields.get(field, " "), 60))
                for field in fields
            )
        )

    def dumpAsString(self, depth=0, max_depth=None, length=None, label=None):
        max_depth = self.max_depth() if max_depth is None else max_depth
        length = len(self) if length is None else length

        s = "{}{:<{}d}{} {:<30s} {:<20s}".format(
            depth * " ",
            self.id,
            math.ceil(math.log10(min(2, length))) + 2,
            " " * (max_depth - depth),
            trim(self.type, 30),
            trim(self.value, 20),
        )
        if label is not None:
            s += label(self)
        s += "\n"

        for child in self.children:
            s += child.dumpAsString(depth + 1, max_depth, length, label)
        return s


class AstPrinter:
    @staticmethod
    def toTS(node, include_types=False):
        out = io.StringIO()
        try:
            AstPrinter.__toTS(out, node, 0, include_types)
            return out.getvalue()
        except Exception:
            print("Unexpected error:", sys.exc_info()[0])
            traceback.print_exc(file=sys.stdout)
        return None

    @staticmethod
    def __toTS(out, node, depth, include_types):

        if node.type == "SourceFile":
            assert len(node.children) == 1
            AstPrinter.__toTS(out, node.children[0], depth, include_types)
        elif node.type == "SyntaxList":
            for child in node.children:
                AstPrinter.__toTS(out, child, depth, include_types)
        elif node.type in ["Block", "ModuleBlock", "CaseBlock"]:
            assert len(node.children) == 3, node.id
            AstPrinter.__toTS(out, node.children[0], depth, include_types)
            out.write("\n")
            AstPrinter.__toTS(out, node.children[1], depth + 1, include_types)
            out.write(2 * depth * " ")
            AstPrinter.__toTS(out, node.children[2], depth, include_types)
            out.write("\n")
        elif node.type.endswith("Statement"):
            out.write(2 * depth * " ")
            for child in node.children:
                AstPrinter.__toTS(out, child, depth, include_types)
            out.write("\n")
        elif node.type in [
            "FirstFutureReservedWord",
            "LastFutureReservedWord",
            "LastContextualKeyword",
            "ExtendsKeyword",
            "AsKeyword",
            "IsKeyword",
            "InKeyword",
            "FromKeyword",
        ] or (
            node.type
            in ["FirstAssignment", "FirstNodeAssignment", "FirstCompoundAssignment"]
            and node.parent.type != "BinaryExpression"
        ):
            assert not node.children, node.id
            assert node.value
            # remove new line (if there is one)
            if out.seek(out.tell() - 1) and out.read(1) == "\n":
                out.seek(out.tell() - 1)
            out.write(" ")
            out.write(node.value)
            out.write(" ")
        elif node.type.endswith("Keyword") and node.type not in [
            "ThisKeyword",
            "SuperKeyword",
        ]:
            assert not node.children
            assert node.value
            if node.type in ["ElseKeyword", "FinallyKeyword"]:
                out.write(2 * depth * " ")
            out.write(node.value)
            out.write(" ")
        elif node.type == "BinaryExpression":
            assert len(node.children) == 3
            AstPrinter.__toTS(out, node.children[0], depth, include_types)
            out.write(" ")
            AstPrinter.__toTS(out, node.children[1], depth, include_types)
            out.write(" ")
            AstPrinter.__toTS(out, node.children[2], depth, include_types)
        elif node.type in [
            "HeritageClause",
            "ImportClause",
            "CatchClause",
            "CaseClause",
            "DefaultClause",
        ]:
            out.write(2 * depth * " ")
            for child in node.children:
                AstPrinter.__toTS(out, child, depth, include_types)
        elif node.type.endswith("Expression") or node.type in [
            "VariableDeclaration",
            "VariableDeclarationList",
            "MetaProperty",
            # 'HeritageClause', 'ImportClause', 'CatchClause', 'CaseClause', 'DefaultClause',
            "ExpressionWithTypeArguments",
            "NamedImports",
            "ImportSpecifier",
            "NamespaceImport",
            "NamedExports",
            "ExportSpecifier",
            "Parameter",
            "TypeParameter",
            "TypeReference",
            "PropertyAssignment",
            "SpreadAssignment",
            "ArrowFunction",
            "ComputedPropertyName",
            "IntersectionType",
            "UnionType",
            "ArrayType",
            "FunctionType",
            "LiteralType",
            "ThisType",
            "ParenthesizedType",
            "MappedType",
            "TupleType",
            "ConstructorType",
            "IndexedAccessType",
            "ConditionalType",
            "RestType",
            "InferType",
            "OptionalType",
            "TemplateSpan",
            "TypeOperator",
            "ExternalModuleReference",
            "FirstTypeNode",
            "LastTypeNode",
            "FirstNode",
            "SemicolonClassElement",
            "TypeLiteral",
            "SpreadElement",
            "ArrayBindingPattern",
            "ObjectBindingPattern",
            "BindingElement",
            "ShorthandPropertyAssignment",
            "TypeQuery",
            "SetAccessor",
            "GetAccessor",
            "EnumMember",
        ]:
            for child in node.children:
                AstPrinter.__toTS(out, child, depth, include_types)

            if (
                not include_types
                and node.type == "Parameter"
                and len(node.children) == 1
                and node.parent.parent.type == "IndexSignature"
            ):
                # if types are removed we add a dummy type to ensure that IndexSignature is parsed correctly
                out.write(": any")

        elif node.type in [
            "EnumDeclaration",
            "PropertyDeclaration",
            "MethodDeclaration",
            "TypeAliasDeclaration",
            "ModuleDeclaration",
            "ImportDeclaration",
            "ExportDeclaration",
            "ImportEqualsDeclaration",
            "NamespaceExportDeclaration",
            "PropertySignature",
            "CallSignature",
            "MethodSignature",
            "ConstructSignature",
            "IndexSignature",
            "Constructor",
            "Decorator",
            "ExportAssignment",
            "FunctionDeclaration",
        ]:
            out.write(2 * depth * " ")
            for pos, child in enumerate(node.children):
                AstPrinter.__toTS(out, child, depth, include_types)

                # if pos == 1 and node.type == 'IndexSignature' and len(child) == 3:
                #     # if types are removed we add a dummy type to ensure that IndexSignature is parsed correctly
                #     out.write(': any')
            out.write("\n")

        elif node.type in ["ClassDeclaration", "InterfaceDeclaration"]:
            out.write(2 * depth * " ")
            cdepth = depth
            for child in node.children:
                if child.type == "CloseBraceToken" and child.value == "}":
                    cdepth -= 1
                    out.write(2 * depth * " ")
                AstPrinter.__toTS(out, child, cdepth, include_types)
                if child.type == "FirstPunctuation" and child.value == "{":
                    cdepth += 1
                    out.write("\n")
                if child.type == "CloseBraceToken" and child.value == "}":
                    out.write("\n")

        elif node.value is not None:
            assert not node.children
            out.write(node.value)

        else:
            assert False, (
                "unhandled type: "
                + str(node.id)
                + ", "
                + str(node)
                + ", num children: "
                + str(len(node.children))
            )

        if include_types and "gold" in node.fields:
            out.write(": {}".format(node.fields["gold"]))
