import unittest

from robustcode.analysis.graph import AstNode
from robustcode.analysis.graph import AstTree


class TestGraphFromJSON(unittest.TestCase):
    def test_single_node(self):
        node = AstNode.fromJson([{"id": 0, "type": "Root"}])
        self.assertEqual(len(node), 1)
        self.assertEqual(node.type, "Root")
        self.assertEqual(node.value, None)
        self.assertEqual(len(node.children), 0)
        self.assertEqual(node.parent, None)

    def test_single_child(self):
        node = AstNode.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1]},
                {"id": 1, "type": "Identifier", "value": "x"},
            ]
        )
        self.assertEqual(len(node), 2)
        self.assertEqual(node.type, "Root")
        self.assertEqual(node.value, None)
        self.assertEqual(len(node.children), 1)
        self.assertEqual(node.parent, None)

        child = node.children[0]
        self.assertEqual(child.type, "Identifier")
        self.assertEqual(child.value, "x")
        self.assertEqual(child.parent, node)

    def test_asttree(self):
        tree = AstTree.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1]},
                {"id": 1, "type": "Identifier", "value": "x"},
            ]
        )
        self.assertEqual(len(tree.nodes), 2)


class TestGraphChildEdges(unittest.TestCase):
    def test_empty(self):
        tree = AstTree.fromJson([{"id": 0, "type": "Root"}])
        edges = tree.child_edges()
        self.assertEqual(len(edges), 0)

    def test_single(self):
        tree = AstTree.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1]},
                {"id": 1, "type": "Identifier", "value": "x"},
            ]
        )
        edges = tree.child_edges()
        self.assertEqual(edges, [(0, 1)])

    def test_multi(self):
        tree = AstTree.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1, 2]},
                {"id": 1, "type": "Identifier", "value": "x"},
                {"id": 2, "type": "Identifier", "value": "x", "children": [3]},
                {"id": 3, "type": "Identifier", "value": "x"},
            ]
        )
        edges = tree.child_edges()
        self.assertEqual(sorted(edges), [(0, 1), (0, 2), (2, 3)])


class TestGraphNextTokenEdges(unittest.TestCase):
    def test_empty(self):
        tree = AstTree.fromJson([{"id": 0, "type": "Root"}])
        edges = tree.next_token_edges()
        self.assertEqual(len(edges), 0)

    def test_empty2(self):
        tree = AstTree.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1]},
                {"id": 1, "type": "Identifier", "value": "x"},
            ]
        )
        edges = tree.next_token_edges()
        self.assertEqual(len(edges), 0)

    def test_single(self):
        tree = AstTree.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1, 2]},
                {"id": 1, "type": "Identifier", "value": "x"},
                {"id": 2, "type": "Identifier", "value": "x", "children": [3]},
                {"id": 3, "type": "Identifier", "value": "x"},
            ]
        )
        edges = tree.next_token_edges()
        self.assertEqual(sorted(edges), [(1, 3)])

    def test_multi(self):
        tree = AstTree.fromJson(
            [
                {"id": 0, "type": "Root", "children": [1, 2, 4]},
                {"id": 1, "type": "Identifier", "value": "x"},
                {"id": 2, "type": "Identifier", "value": "x", "children": [3]},
                {"id": 3, "type": "Identifier", "value": "x"},
                {"id": 4, "type": "Identifier", "value": "x"},
            ]
        )
        edges = tree.next_token_edges()
        self.assertEqual(sorted(edges), [(1, 3), (3, 4)])
        print("\n" + tree.dumpAsString())


if __name__ == "__main__":
    unittest.main()
