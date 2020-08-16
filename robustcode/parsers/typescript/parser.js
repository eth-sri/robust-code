#!/usr/bin/env nodejs

"use strict";
const ts = require("typescript");
const core = require("./core");

let ArgumentParser = require('argparse').ArgumentParser;
let parser = new ArgumentParser({
  addHelp:true,
  description: 'TypeScript Parser'
});
parser.addArgument(
  [ 'file' ],
  {
    help: 'File to parse'
  }
);
parser.addArgument(
  [ '--remove_types' ],
  {
    action: 'storeTrue',
	default: false,
    help: 'Whether to remove user defined type annotaitons'
  }
);
let args = parser.parseArgs();

function print(x) { console.log(x); }
// let removableLexicalKinds = [
// 	ts.SyntaxKind.EndOfFileToken,
// 	ts.SyntaxKind.NewLineTrivia,
// 	ts.SyntaxKind.WhitespaceTrivia
// ];

let deps = [];

let program = ts.createProgram(deps.concat([args.file]), {
		target: ts.ScriptTarget.Latest,
		module: ts.ModuleKind.CommonJS,
		// noImplicitAny: false,
		// types: [],
		// noResolve: true,
		// typeRoots: [__dirname + '/node_modules/@types/'],
		// noLib: true,
        // declaration: true,
        // declarationDir: 'tmp',
		// skipLibCheck: true,
		checkJs: true,
		allowJs: true });


let checker = null;
try {
	checker = program.getTypeChecker();
} catch (err) {
	return null;
}

// print('#source files: ' + program.getSourceFiles().length);
// for (const sourceFile of program.getSourceFiles()) {
// 	print('\t'+ sourceFile.getSourceFile().fileName);
// }
for (const sourceFile of program.getSourceFiles()) {
	let filename = sourceFile.getSourceFile().fileName;
	// if (filename.endsWith('.d.ts')) continue;
	if (filename !== args.file) continue;

	core.numberNodes(sourceFile, args.remove_types);

	// core.printTree(sourceFile, checker, args.remove_types);
	let ref_paths = new Set();
	let data = core.treeToJson(sourceFile, checker, args.remove_types, ref_paths, true);
	// ref_paths.delete(filename);
	// print(ref_paths);

	if (!core.is_encrypted(data)) {
		print(JSON.stringify(data));
	}

	// Do not call process.exit() as it does not guarantee that the asynchronous print will finish
	process.exitCode = 0;
	return;
}
