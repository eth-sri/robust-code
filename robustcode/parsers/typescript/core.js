#!/usr/bin/env nodejs

"use strict";
const ts = require("typescript");

function print(x) { console.log(x); }

let removableLexicalKinds = [
	ts.SyntaxKind.EndOfFileToken,
	ts.SyntaxKind.NewLineTrivia,
	ts.SyntaxKind.WhitespaceTrivia
];

module.exports = {
	'is_encrypted': is_encrypted,
	'treeToJson': treeToJson,
	'numberNodes': numberNodes,
	'printTree': printTree
};

function is_encrypted(data) {
	return (data.length > 5 && data[4].hasOwnProperty('value') && data[4]['value'] === 'GITCRYPT');
}

function rightSibling(node, parent) {
	if (node === undefined) return undefined;
	let found = false;
	for (const child of parent.getChildren()) {
		if (found) return child;
		if (child === node) {
			found = true;
		}
	}
	return undefined;
}

function leftSibling(node, parent) {
	if (node === undefined) return undefined;
	let last_child = undefined;
	for (const child of parent.getChildren()) {
		if (child === node) {
			return last_child;
		}
		last_child = child;
	}
	return undefined;
}

function isTypeNode(node) {
	return ts.SyntaxKind.ThisKeyword !== node.kind &&
		ts.SyntaxKind.NullKeyword !== node.kind &&
		ts.isTypeNode(node);
}

function isUserTypeAnnotation(node, parent) {
	/*
	 Covers two cases:
	 1)   : type
			  <-- node points to type

	 */
	if (isTypeNode(node) && leftSibling(node, parent) !== undefined &&
		    ts.SyntaxKind.ColonToken === leftSibling(node, parent).kind &&
		 	(
		 		leftSibling(leftSibling(leftSibling(node, parent), parent), parent) === undefined ||
				ts.SyntaxKind.QuestionToken !== leftSibling(leftSibling(leftSibling(node, parent), parent), parent).kind
			)
		   ) {
		return true;
	}
	/*
	2)   : type
	      <-- node points to colon
	*/
	if (
		   	ts.SyntaxKind.ColonToken === node.kind && rightSibling(node, parent) !== undefined &&
			isTypeNode(rightSibling(node, parent)) && (
				leftSibling(leftSibling(node, parent), parent) === undefined ||
				ts.SyntaxKind.QuestionToken !== leftSibling(leftSibling(node, parent), parent).kind
			)
		   ) {
		leftSibling(node, parent).instrGoldTypeNode = rightSibling(node, parent);
		return true;
	}
	return false;
}

function skipNode(node, parent, remove_types) {
	return (
		removableLexicalKinds.indexOf(node.kind) !== -1 ||
		ts.SyntaxKind[node.kind].indexOf("JSDoc") !== -1 ||
		(remove_types && isUserTypeAnnotation(node, parent))
	);
}

function* forEachChild(node, remove_types) {
	for (const child of node.getChildren()) {
		if (!skipNode(child, node, remove_types)) {
			yield child;
		}
	}
}

function numberNodes(tree, remove_types, id=0) {
	tree.instrID = id;
	for (const child of forEachChild(tree, remove_types)) {
		id = numberNodes(child, remove_types,id + 1);
	}
	return id;
}

function treeToJson(tree, checker, remove_types, ref_paths=undefined, extended=false) {
	let data = [];
	try {
		__treeToJson(tree, checker, data, remove_types, ref_paths, extended);
	} catch (err) {
		print(err);
		return [];
	}
	return data;
}

function getChildrenIds(node, remove_types) {
	let ids = [];
	for (const child of forEachChild(node, remove_types)) {
		ids.push(child.instrID);
	}
	return ids;
}

function __treeToJson(tree, checker, data, remove_types, ref_paths, extended) {
	let node_data = {
		'id': tree.instrID,
		'type': ts.SyntaxKind[tree.kind],
	};

	let children = getChildrenIds(tree, remove_types);
	if (children.length > 0) {
		node_data['children'] = children;
	} else {
		node_data['value'] = tree.getText();
	}

	if (extended) {
		if (tree.instrID === 0) {
			// skip root type, which is: 'typeof import(<path>)' as this causes mismatch based on file name
			// node_data['target'] = 'O';
			// node_data['gold'] = 0;
		} else {
			let [symbol_type, full_type, gold_type] = getType(checker, tree, ref_paths);
			// node_data['target'] = symbol_type;
			if (full_type !== undefined) {
				node_data['target'] = full_type;
			}
			// node_data['gold'] = is_gold ? 1 : 0;
			if (gold_type !== undefined) {
				node_data['gold'] = gold_type;
			}
		}
	}

	data.push(node_data);
	for (const child of forEachChild(tree, remove_types)) {
		__treeToJson(child, checker, data, remove_types, ref_paths, extended);
	}
}

function normalizeType(type) {
	if (type === undefined) return type;
	type = type.trim();
	if (type.startsWith("\"") || type.startsWith("'")) {
		return 'string';
	}

	if (type.match("^-?\\.?[0-9][0-9,\\.]*$")) {
		return 'number';
	}

	if (type.match("^-?[0-9]+n$")) {
		return "bigint";
	}

	if (type === 'true' || type === 'false') {
		return 'boolean';
	}

	if (type === 'String') {
		return 'string';
	}

	return type;
}

function getRoot(node) {
	let root = node;
	while (root.parent !== undefined) {
		root = root.parent;
	}
	return root;
}

function getType(checker, node, ref_paths=undefined) {
	let symbol_target = undefined;
	// let is_gold = false;

	if (node.kind === ts.SyntaxKind.NumericLiteral) {
		symbol_target = 'number';
	}
	if (node.kind === ts.SyntaxKind.StringLiteral) {
		symbol_target = 'string';
	}
	if (node.kind === ts.SyntaxKind.RegularExpressionLiteral) {
		symbol_target = 'RegExp';
	}
	if (node.kind === ts.SyntaxKind.BigIntLiteral) {
		symbol_target = 'bigint';
	}
	if (node.kind === ts.SyntaxKind.TrueKeyword || node.kind === ts.SyntaxKind.FalseKeyword) {
		symbol_target = 'boolean';
	}

	try {
		let symbol = checker.getSymbolAtLocation(node);
		if (symbol) {
			if (symbol_target === undefined) {
				let type = checker.getTypeOfSymbolAtLocation(symbol, node);
				symbol_target = checker.typeToString(type);

				if (type.hasOwnProperty('symbol') && type.symbol.hasOwnProperty('valueDeclaration') &&
					type.symbol.valueDeclaration !== undefined) {
					ref_paths.add(getRoot(type.symbol.valueDeclaration).path);
				}
			}

			if (symbol.valueDeclaration !== undefined) {
				let symbol_root = getRoot(symbol.valueDeclaration);
				if (ref_paths !== undefined) {
					ref_paths.add(symbol_root.path);
				}

				// if the roots are different, then the declaration is included from different source file
				// if (symbol_root === getRoot(node)) {
				// 	if (!symbol.hasOwnProperty('instrGold')) {
				// 		is_gold = true;
				// 		symbol.instrGold = true;
				// 	}
				//
				// 	// symbol_target = symbol.valueDeclaration.type.getText();
				// }
			}
		}
	} catch (e) { }


	let full_target = symbol_target;
	try {
		if (full_target === undefined) {
			full_target = checker.typeToString(checker.getTypeAtLocation(node));
			if (full_target === 'any') {
				full_target = undefined;
			}
		}
	} catch (err) { }

	let gold_type = node.hasOwnProperty('instrGoldTypeNode') ? node.instrGoldTypeNode.getText() : undefined;
	return [normalizeType(symbol_target), normalizeType(full_target), gold_type];
}

function printTree(tree, checker, remove_types, depth=0) {
	let [symbol_type, full_type, is_gold] = getType(checker, tree);
	print(' '.repeat(depth)  + 'id:' + tree.instrID  + ", " + ts.isTypeNode(tree) + ", " +
		ts.SyntaxKind[tree.kind] + ":" + ((tree.getChildCount() === 0) ? tree.getText() : "") + " : " + symbol_type + ", " + full_type + ", " + is_gold);
	for (const child of forEachChild(tree, remove_types)) {
		printTree(child, checker, remove_types,depth + 1);
	}
}

