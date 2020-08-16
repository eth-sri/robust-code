import gzip
import json
import os

from robustcode.analysis.graph import AstTree
from robustcode.analysis.graph import TypeScriptGraphAnalyzer


def main():
    with gzip.open(
        os.path.expanduser("../../../data/deeptyperast_1k/train.json.gz"), "rb"
    ) as f:
        for line in f:
            data = json.loads(line)

            # Print the content
            for key, values in data.items():
                print(key, values)

            """
            Each json samples corresponds to a single source file.
            The json format is as follows (printed by running the above):
            
            # path of the source file, including the project
            'id': 'SharePoint/sp-dev-fx-webparts/ICustomBusinessObjectsPnPJsState.ts'
            
            # ast values
            'ast_values': ['<null>', '<null>', '<null>', 'import', '<null>', '<null>', '{', '<null>', '<null>', 'MyDocument', '}', 'from', '"../model/MyDocument"', ';', '<null>', '<null>', 'export', 'interface', 'ICustomBusinessObjectsPnPJsState', '{', '<null>', '<null>', 'myDocuments', ';', '<null>', 'errors', ';', '}']
            
            # ast types:
            'ast_types': ['SourceFile', 'SyntaxList', 'ImportDeclaration', 'ImportKeyword', 'ImportClause', 'NamedImports', 'FirstPunctuation', 'SyntaxList', 'ImportSpecifier', 'Identifier', 'CloseBraceToken', 'FromKeyword', 'StringLiteral', 'SemicolonToken', 'InterfaceDeclaration', 'SyntaxList', 'ExportKeyword', 'InterfaceKeyword', 'Identifier', 'FirstPunctuation', 'SyntaxList', 'PropertySignature', 'Identifier', 'SemicolonToken', 'PropertySignature', 'Identifier', 'SemicolonToken', 'CloseBraceToken']
            
            # inferred type when running typescript analyzer on the full project
            target_full ['<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', 'typeof MyDocument', 'typeof MyDocument', '<null>', '<null>', 'string', '<null>', 'ICustomBusinessObjectsPnPJsState', '<null>', '<null>', '<null>', 'any', '<null>', '<null>', 'MyDocument[]', 'MyDocument[]', '<null>', 'string[]', 'string[]', '<null>', '<null>']
            
            # mask containing 1 for locations with inferred types
            mask_valid_full [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
            
            # mask containing 1 for locations manually annotated by users
            mask_gold [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
            
            # user annotations, note that it does not necessarily matches the inferred type
            gold_type ['<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', '<null>', 'MyDocument[]', '<null>', '<null>', 'string[]', '<null>', '<null>']
            
            # position in the parent. 0 denotes the first child, 1 the second child, etc.
            pos [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 2, 3, 4, 1, 0, 0, 1, 2, 3, 4, 0, 0, 1, 1, 0, 1, 5]
            
            # depth in the tree. 0 denotes the root
            depth [0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 5, 3, 3, 3, 2, 3, 4, 3, 3, 3, 3, 4, 5, 5, 4, 5, 5, 3]
            
            # dependencies used by type inference to obtain ground truth files.
            # useful when the type inference file is run as part of the evaluation/training to make it faster             
            dependencies ['Repos/SharePoint/sp-dev-fx-webparts/samples/react-sp-pnp-js-property-decorators/src/webparts/customBusinessObjectsPnPJs/components/ICustomBusinessObjectsPnPJsState.ts', 'SharePoint/sp-dev-fx-webparts/samples/react-sp-pnp-js-property-decorators/src/webparts/customBusinessObjectsPnPJs/model/MyDocument.ts', 'typescript/node_modules/typescript/lib/lib.es5.d.ts']
            
            # various edge types supported by the current analyzer
            child_edges_src [0, 1, 1, 2, 2, 2, 2, 2, 4, 5, 5, 5, 7, 8, 14, 14, 14, 14, 14, 14, 15, 20, 20, 21, 21, 24, 24]
            child_edges_tgt [1, 2, 14, 3, 4, 11, 12, 13, 5, 6, 7, 10, 8, 9, 15, 17, 18, 19, 20, 27, 16, 21, 24, 22, 23, 25, 26]
            
            next_token_edges_src [3, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 25, 26]
            next_token_edges_tgt [6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 22, 23, 25, 26, 27]
            
            last_lexical_usage_edges_src []
            last_lexical_usage_edges_tgt []
            
            computed_from_edges_src []
            computed_from_edges_tgt []
            
            returns_to_edges_src []
            returns_to_edges_tgt []
            
            guard_by_edges_src []
            guard_by_edges_tgt []
            
            guard_by_negation_edges_src []
            guard_by_negation_edges_tgt []
            
            last_write_edges_src []
            last_write_edges_tgt []
            
            last_read_edges_src []
            last_read_edges_tgt []
            """

            # parse that sample as AST Tree
            tree = AstTree.fromTensor(
                data["ast_types"],
                data["ast_values"],
                data["depth"],
                {"target": data["target_full"]},
            )  # 'gold': sample.gold,
            tree.analyzer = TypeScriptGraphAnalyzer()

            # use analyzer to compute edges
            tree.number_nodes()
            per_type_edges = tree.compute_all_edges()
            for edge_type, values in per_type_edges.items():
                print("edge_type", edge_type)
                print("\t source nodes:", [v[0] for v in values])
                print("\t target nodes:", [v[1] for v in values])


if __name__ == "__main__":
    main()
