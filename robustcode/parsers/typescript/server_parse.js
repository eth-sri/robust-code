const ts = require("typescript");
const core = require('./core');

module.exports = {
	'parse_file': parse_file
};

function parse_file(filename, remove_types, dependencies) {
    // console.log(filename);
    let program = ts.createProgram(dependencies.concat([filename]).sort(), {
		target: ts.ScriptTarget.Latest,
		module: ts.ModuleKind.CommonJS,
		// noImplicitAny: false,
        types: [],
		checkJs: true,
		allowJs: true,
        noLib: true,
        noResolve: true,
        // skipLibCheck: true,
    });

    let checker = null;
    try {
        checker = program.getTypeChecker();
    } catch (err) {
        return "NullTypeChecker";
    }

    for (const sourceFile of program.getSourceFiles()) {
	    if (sourceFile.getSourceFile().fileName !== filename) continue;

	    let num_syntax_errors = program.getSyntacticDiagnostics(sourceFile).length;
    	core.numberNodes(sourceFile, remove_types);

	    let data = {
	        'syntactic_errors': num_syntax_errors,
            // 'semantic_errors': program.getSemanticDiagnostics(sourceFile).length,
	        'ast': core.treeToJson(sourceFile, checker, remove_types, undefined, true)
        };

        return JSON.stringify(data);
    }
    return "Error";
}
