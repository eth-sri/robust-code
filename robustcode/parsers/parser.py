#!/usr/bin/env python
import argparse
import collections
import json
import os
import random
import subprocess
import sys

import requests

from robustcode.analysis.graph import AstTree


def FindSubBinary(binary):
    for path in sys.path:
        abs_binary_path = os.path.join(os.path.abspath(path), binary)
        if os.path.isfile(abs_binary_path):
            return abs_binary_path
    return binary


Parser = collections.namedtuple(
    "Parser", ["language", "file_extension", "path", "prefix"]
)

parsers = [
    Parser(
        "typescript",
        ".ts",
        FindSubBinary("robustcode/parsers/typescript/parser.js"),
        [],
    ),
]

parsers_server = [] + [
    Parser("typescript", ".ts", "http://localhost:{}/api/v1/parse".format(port), None)
    for port in range(3000, 3016)
]


def print_ast(nodes, idx=0, depth=0):
    print(" " * depth, "id: {}".format(idx), nodes[idx])
    if "children" not in nodes[idx]:
        return

    for child_id in nodes[idx]["children"]:
        print_ast(nodes, child_id, depth + 1)


def get_lang_for_filename(filename):
    for parser in parsers:
        if filename.endswith(parser.file_extension):
            return parser.language
    return None


def shuffle(data):
    random.shuffle(data)
    return data


def get_parser_for_filename(filename, server=False):
    for parser in parsers if not server else shuffle(parsers_server):
        if filename.endswith(parser.file_extension):
            return parser
    return None


def get_parser_by_name(name, server=False):
    for parser in parsers if not server else shuffle(parsers_server):
        if name == parser.language:
            return parser
    return None


"""
Avoid spawning new process whenever file is parsed
The server needs to be started manually before the first call
"""


def parse_file_server(filename, data=None, parser_name=None):
    if parser_name is None:
        parser = get_parser_for_filename(filename, server=True)
    else:
        parser = get_parser_by_name(parser_name, server=True)
    if parser is None:
        return None

    headers = {"Content-type": "application/json"}
    data = {} if data is None else data
    data["filename"] = filename
    r = requests.post(url=parser.path, data=json.dumps(data), headers=headers)
    if r.text == "SyntaxError":
        return None
    return json.loads(r.text)


def parse_file(filename, args=None):
    parser = get_parser_for_filename(filename)
    if parser is None:
        return None

    args = [] if args is None else args
    proc = subprocess.Popen(
        parser.prefix + [parser.path, filename] + args, stdout=subprocess.PIPE
    )
    try:
        outs, errs = proc.communicate(timeout=15)
        if not outs:
            return None
        return json.loads(outs)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="File to parse")
    parser.add_argument("--use_server", default=False, action="store_true")
    parser.add_argument(
        "--options", default=[], nargs="*", help="Options passed to the parser"
    )
    parser.add_argument("--pretty", default=False, action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print("File '" + args.file + "' not found!")
        exit(1)

    if not args.use_server:
        ast_json = parse_file(args.file, ["--{}".format(key) for key in args.options])
    else:
        ast_json = parse_file_server(args.file, {key: True for key in args.options})

    if not args.pretty:
        print(ast_json)
    else:
        tree = AstTree.fromJson(ast_json, fields=["target"])
        print(tree.dumpFieldsAsString(fields=["target"]))


if __name__ == "__main__":
    main()
