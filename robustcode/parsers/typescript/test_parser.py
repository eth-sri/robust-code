import unittest

from robustcode.analysis.graph import AstTree
from robustcode.parsers.parser import parse_file


class TestTypeAnottations(unittest.TestCase):
    def test_annotation_escaping(self):
        ast_ref = AstTree.fromJson(parse_file('fixtures/fnc_notypes.ts'))
        ast = AstTree.fromJson(parse_file('fixtures/fnc.ts', args=['--remove_types']))
        self.assertEqual(len(ast_ref), len(ast))
        for node, node_ref in zip(ast.nodes, ast_ref.nodes):
            self.assertEqual(node.type, node_ref.type)
            self.assertEqual(node.value, node_ref.value)


if __name__ == '__main__':
    unittest.main()
