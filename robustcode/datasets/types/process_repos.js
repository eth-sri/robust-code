#!/usr/bin/env nodejs

"use strict";
const ts = require("typescript");
const fs = require("fs");
const path = require("path");
const assert = require('assert');
// const child_process = require('child_process');
const mkdirp = require('mkdirp');
const zlib = require('zlib');

const core = require('../../parsers/typescript/core');


let ArgumentParser = require('argparse').ArgumentParser;
let parser = new ArgumentParser({
  addHelp:true,
  description: 'Parse and run Type Inference on a TypeScript project'
});
parser.addArgument(
  [ 'repos' ],
  {
    help: 'Path the dataset of typescript repositories'
  }
);
parser.addArgument(
  [ 'cleaned' ],
  {
    help: 'Directory where to write results'
  }
);
parser.addArgument(
  [ 'project' ],
  {
    help: 'Path to project inside "repos" to analyze'
  }
);

let args = parser.parseArgs();

function print(x) { console.log(x); }

function analyzeFile(filename, dir) {
	if (filename.endsWith('.d.ts')) return [0, 0, 0];
	if (path.relative(dir, filename).startsWith("..")) return [0, 0, 0];

	let program = ts.createProgram([filename], {
		target: ts.ScriptTarget.Latest,
		module: ts.ModuleKind.CommonJS,
		noImplicitAny: false,
		// types: [],
        // skipLibCheck: true,
		// lib: ["es6", "es2015", "dom", "scripthost"],
		// lib: [],
		checkJs: true,
		allowJs: true });

	let checker = null;
	try {
		checker = program.getTypeChecker();
    } catch (err) {
		return null;
	}

	let num_inferred_types = 0;
	let num_any_types = 0;
	let num_total = 0;
	for (const sourceFile of program.getSourceFiles()) {
		if (filename !== sourceFile.getSourceFile().fileName) continue;
		try {
			core.numberNodes(sourceFile, true);

			let data = core.treeToJson(sourceFile, checker, true, true);
			if (core.is_encrypted(data)) continue;

			for (let node of data) {
				num_total++;
				if (node['target'] !== 'O') {
					num_inferred_types++;
					if (node['target'] === 'any') {
						num_any_types++;
					}
				}
			}
		}
		catch (e) {
			console.log(e);
			console.log("Error parsing file " + filename);
		}
	}
	return [num_inferred_types, num_any_types, num_total]
}

analyzeProject(args.repos, args.cleaned, args.project);
function analyzeProject(repos, cleaned, dir) {
	let children = fs.readdirSync(dir);
	assert.ok(children.find(value => value === "tsconfig.json"));
	print("Config in: " + args.project);

	let files = [];
	walkSync(dir, files);
	print('number of files: ' + files.length);

	let program = ts.createProgram(files, {
		target: ts.ScriptTarget.Latest,
		module: ts.ModuleKind.CommonJS,
		noImplicitAny: false,
		checkJs: true,
		allowJs: true });

	let checker = null;
	try {
		checker = program.getTypeChecker();
    } catch (err) {
		return null;
	}

	let per_file_results = [];
	let source_files = [];
	for (const sourceFile of program.getSourceFiles()) {
		source_files.push(sourceFile.getSourceFile().fileName);
	}

	let num_inferred_types = 0;
	let num_any_types = 0;
	let num_total = 0;
	for (const sourceFile of program.getSourceFiles()) {
		let filename = sourceFile.getSourceFile().fileName;
		if (filename.endsWith('.d.ts')) continue;
		try {
			let relativePath = path.relative(dir, filename);
			if (relativePath.startsWith(".."))
				continue;

			core.numberNodes(sourceFile, true);

			// printTree(sourceFile, checker);
            let ref_paths = new Set();
			let data = core.treeToJson(sourceFile, checker, true, ref_paths, true);
			// print(JSON.stringify(data));
			if (core.is_encrypted(data)) continue;

			for (let node of data) {
				num_total++;
				if (node.hasOwnProperty('target')) { //node['target'] !== 'O'
					num_inferred_types++;
					if (node['target'] === 'any') {
						num_any_types++;
					}
				}
			}

			per_file_results.push(
                {
                    'filename': filename,
                    'dependencies': Array.from(ref_paths),
					'source_files': source_files,
                    'ast': data
                }
            );

		}
		catch (e) {
			console.log(e);
			console.log("Error parsing file " + filename);
		}
	}
	print('Inferred Types: ' + num_inferred_types + '/' + (num_any_types + num_inferred_types) + ", total: " + num_total);

    let out_dir = path.dirname(dir.replace(repos, cleaned));
    if (!fs.existsSync(out_dir)) {
        mkdirp.sync(out_dir);
    }
    // fs.writeFileSync(dir.replace(repos, cleaned) + ".json", JSON.stringify(per_file_results), 'utf-8');

    let output = fs.createWriteStream(dir.replace(repos, cleaned) + ".json.gz");
	let compress = zlib.createGzip();
	compress.pipe(output);
	compress.write(JSON.stringify(per_file_results));
	compress.end();
}


function walkSync(dir, filelist) {
	var fs = fs || require('fs'), files = fs.readdirSync(dir);
	filelist = filelist || [];
	files.forEach(function (file) {
		let fullPath = path.join(dir, file);
		try {
			if (fs.statSync(fullPath).isDirectory()) {
				if (file !== ".git" && file !== 'node_modules')
					filelist = walkSync(dir + '/' + file, filelist);
			}
			else if (file.endsWith('.js') || file.endsWith('.ts')) {
				if (fs.statSync(fullPath).size < 1000*1000)
					filelist.push(fullPath);
			}
		} catch (e) {
			console.error("Error processing " + file);
		}
	});
	return filelist;
}
