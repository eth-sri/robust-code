import argparse
import collections
import glob
import gzip
import heapq
import itertools
import json
import math
import multiprocessing
import os
import random
import re
import shutil
import subprocess
import sys
import time

from robustcode.analysis.graph import AstNode
from robustcode.parsers.parser import parse_file_server
from robustcode.util.misc import is_file_empty
from robustcode.util.misc import Logger


def find_top_level_projects(path):
    if (
        "node_modules" in path
        or "DefinitelyTyped" in path
        or "TypeScript/tests" in path
        or ".git" in path
    ):
        return

    if any(filename == "tsconfig.json" for filename in os.listdir(path)):
        yield path
        return

    for dir_name in os.listdir(path):
        dir_path = os.path.join(path, dir_name)
        if not os.path.isdir(dir_path):
            continue
        yield from find_top_level_projects(dir_path)


def process_project(repo_path, repo_cleaned_path, path):
    print(repo_path, repo_cleaned_path, path)
    NpmPackageInstaller.install_dependencies(path)
    subprocess.call(
        [
            "nodejs",
            "--max-old-space-size=8000",
            "process_repos.js",
            repo_path,
            repo_cleaned_path,
            path,
        ]
    )


def collect_project_results(project_path, out_dir, include_js):
    org, project = project_path.split("/")[-3:-1]

    out_dir_gold = os.path.join(out_dir, "outputs-gold")
    out_dir_all = os.path.join(out_dir, "outputs-all")
    out_dir_asts = os.path.join(out_dir, "outputs-asts")

    whitespace_pattern = re.compile(r"\s+")

    # find files to process
    files = glob.glob("{}**/*.ts.ast".format(project_path), recursive=True)
    if include_js:
        files += glob.glob("{}**/*.js.ast".format(project_path), recursive=True)
    if not any(os.path.getsize(file_name) > 0 for file_name in files):
        return

    files = [file_name[:-4] for file_name in files]

    print("num files: {:5d}\t{} {}".format(len(files), org, project))

    f_gold = open("{}/{}_{}".format(out_dir_gold, org, project), "w", encoding="utf-8")
    f_all = open("{}/{}_{}".format(out_dir_all, org, project), "w", encoding="utf-8")
    f_asts = open("{}/{}_{}".format(out_dir_asts, org, project), "w", encoding="utf-8")

    for file_name in files:
        with open("{}.ast".format(file_name), "r", encoding="utf-8") as f:
            ast = json.loads(f.read())
            if not ast:
                continue
            ast = {"data": ast, "id": file_name, "js": file_name.endswith("js")}

        tokens = []
        gold = []
        types = []
        for node in ast["data"]:
            if "value" not in node:
                continue

            tokens.append(node["value"])
            types.append(node["target"])
            gold.append(node["target"] if node["gold"] else "O")
        # remove whitespace from tokens
        # this is required as DeepType splits input on spaces
        tokens = [re.sub(whitespace_pattern, "", t) for t in tokens]

        if file_name.endswith("js"):
            # Identify JS files specifically; these should not be used as oracles
            gold.insert(0, "O")
            types.insert(0, "O")
            tokens.insert(0, "'js'")

        f_gold.write(" ".join(tokens) + "\t" + " ".join(gold) + "\n")
        f_all.write(" ".join(tokens) + "\t" + " ".join(types) + "\n")
        f_asts.write(json.dumps(ast) + "\n")

    f_gold.close()
    f_all.close()
    f_asts.close()


def collect_results(path, out_dir, num_threads, include_js=False):
    out_dir_gold = os.path.join(out_dir, "outputs-gold")
    out_dir_all = os.path.join(out_dir, "outputs-all")
    out_dir_asts = os.path.join(out_dir, "outputs-asts")
    os.makedirs(out_dir_gold, exist_ok=True)
    os.makedirs(out_dir_all, exist_ok=True)
    os.makedirs(out_dir_asts, exist_ok=True)

    project_paths = [
        (project_path, out_dir, include_js)
        for project_path in glob.iglob("{}/*/*/".format(path))
    ]
    with multiprocessing.Pool(num_threads) as pool:
        pool.starmap(collect_project_results, project_paths)


"""
Optimize project dependencies
Each file has a list of dependencies required by the type inference.
This is however just an overapproximation which includes many files that are not used.
"""


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


"""
Optimize number of dependencies required for type inference.
Useful to speed-up type inference if the files are re-evaluated as part of adversarial search. 
"""


def optimize_deps(filename, deps, base_deps, ref_json, base_time):
    t = time.time()
    opt_deps = set(deps)
    removal_candidates = list(set(deps) - set(base_deps))
    random.shuffle(removal_candidates)

    opt_time = None
    queue = PriorityHeap()
    queue.add(removal_candidates)
    while len(queue) > 0:
        data = queue.pop()
        for to_remove in chunks(data, max(1, math.ceil(len(data) / 2))):
            start = time.time()
            ast_json = parse_file_server(
                filename,
                parser_name="typescript",
                data={
                    "remove_types": True,
                    "deps": sorted([d for d in opt_deps if d not in to_remove]),
                },
            )
            opt_time = time.time() - start
            assert ast_json is not None

            if ast_json == ref_json:
                print(
                    "\ttook: {}, remove: {}".format(time.time() - start, len(to_remove))
                )
                opt_deps.difference_update(to_remove)
            elif len(to_remove) != 1:
                print("\ttook: {}, recurse".format(time.time() - start))
                queue.add(to_remove)

    Logger.debug(
        "Original Size: #{} ({:.2f}s), Base Size: #{}, Optimized Size: #{} ({:.2f}s), Total Time: {:.2f}".format(
            len(deps),
            base_time,
            len(base_deps),
            len(opt_deps),
            opt_time,
            time.time() - t,
        )
    )
    return list(opt_deps)


class PriorityHeap:
    def __init__(self):
        self.pq = []
        self.counter = itertools.count()

    def add(self, data):
        entry = [-len(data), next(self.counter), data]
        heapq.heappush(self.pq, entry)

    def pop(self):
        assert self.pq
        priority, count, data = heapq.heappop(self.pq)
        return data

    def __len__(self):
        return len(self.pq)


def optimize_project(path, pool, include_js=False):
    if os.path.exists(path + ".opt"):
        return
    if is_file_empty(path):
        return

    with gzip.open(path, "rb") as f:
        entries = json.loads(f.read())

    if not include_js:
        entries = [entry for entry in entries if entry["filename"].endswith(".ts")]

    Logger.start_scope("Optimizing {}".format(path))
    Logger.debug("#Entries: {}".format(len(entries)))

    num_diffs = 0
    opt_entries = []
    for idx, entry in enumerate(pool.imap_unordered(optimize_file, entries)):
        # for idx, entry in enumerate(entries):
        #     entry = optimize_file(entry)
        sys.stderr.write("\r{}/{}".format(idx, len(entries)))
        num_diffs += entry["num_diffs"]
        opt_entries.append(entry)
    sys.stderr.write("\r{}/{}\n".format(len(entries), len(entries)))
    Logger.debug("#Diffs: {}".format(num_diffs))
    Logger.end_scope()

    print("write: ", path + ".opt")
    with gzip.open(path + ".opt", "wb") as f:
        f.write(json.dumps(opt_entries).encode("utf-8"))


def optimize_file(data):
    start = time.time()
    ast_json = parse_file_server(
        data["filename"],
        parser_name="typescript",
        data={"remove_types": True, "deps": data["source_files"]},
    )
    base_time = time.time() - start
    assert ast_json is not None
    ref_root = AstNode.fromJson(data["ast"], fields=["gold", "target"])
    root = AstNode.fromJson(ast_json, fields=["gold", "target"])
    num_diffs = ref_root.num_tree_diffs(root)
    # print(ref_root.tree_equal(root, verbose=True))

    data["dependencies"] = optimize_deps(
        data["filename"],
        data["source_files"],
        data["dependencies"],
        ast_json,
        base_time,
    )
    data["ast"] = ast_json
    data["num_diffs"] = num_diffs
    return data


"""
Collect and install npm packages
"""


class NpmPackageInstaller:
    @staticmethod
    def install_dependencies(dir_path):
        if os.path.exists(os.path.join(dir_path, "node_modules")):
            return

        packages = NpmPackageInstaller.collect_dependencies(dir_path)
        packages = [
            package
            for package in packages
            if NpmPackageInstaller.package_exists(package)
        ]

        for chunk in chunks(packages, 100):
            # nodejs --stack_size=4048 $(which npm)
            args = ["{}@{}".format(name, version) for name, version in chunk]
            p = subprocess.Popen(["npm", "install"] + args, cwd=dir_path)
            p.wait()

    @staticmethod
    def collect_dependencies(dir_path):
        deps = collections.defaultdict(collections.Counter)
        for file in NpmPackageInstaller.find_package_json(os.path.abspath(dir_path)):
            with open(file, "r") as f:
                content = json.loads(f.read())
                dependencies = {}
                if "devDependencies" in content:
                    dependencies = content["devDependencies"]
                elif "dependencies" in content:
                    dependencies = content["dependencies"]

                for key, value in dependencies.items():
                    deps[key][value] += 1

        candidate_packages = []
        for idx, name in enumerate(sorted(deps.keys())):
            versions = deps[name]
            for version, count in versions.most_common(1):
                candidate_packages.append((name, version))
        return candidate_packages

    @staticmethod
    def find_package_json(root_path):
        if (
            "node_modules" in root_path
            or "DefinitelyTyped" in root_path
            or "TypeScript/tests" in root_path
            or ".git" in root_path
        ):
            return

        for name in os.listdir(root_path):
            path = os.path.join(root_path, name)
            if os.path.isdir(path):
                yield from NpmPackageInstaller.find_package_json(path)
            if name == "package.json":
                yield path

    @staticmethod
    def package_exists(entry):
        name, version = entry

        proc = subprocess.Popen(
            ["npm", "info", "{}@{}".format(name, version)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # print(' '.join(['npm', 'info', '{}@{}'.format(name, version)]))
        try:
            outs, errs = proc.communicate(timeout=15)
            return proc.returncode == 0 and len(outs.strip()) > 0
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()
            return False


"""
Split dataset into train/valid/test
"""


def save_dataset(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # collect all projects
    paths = glob.glob("{}/**/*.json.gz".format(args.repos_cleaned), recursive=True)
    random.shuffle(paths)
    train_paths = paths[: (8 * len(paths)) // 10]
    valid_paths = paths[(8 * len(paths)) // 10 : (9 * len(paths)) // 10]
    test_paths = paths[(9 * len(paths)) // 10 :]

    __save_paths(args, train_paths, "train")
    __save_paths(args, valid_paths, "valid")
    __save_paths(args, test_paths, "test")


def __save_paths(args, paths, name):
    num_entries = 0
    with gzip.open(os.path.join(args.out_dir, name + ".json.gz"), "wb") as f_out:
        for path in paths:
            optimized = False
            if os.path.exists(path + ".opt"):
                optimized = True
                path = path + ".opt"

            if is_file_empty(path):
                continue
            print(num_entries, path)

            with gzip.open(path, "rb") as f:
                for entry in json.loads(f.read()):
                    if not (args.include_js or entry["filename"].endswith(".ts")):
                        continue

                    num_entries += 1
                    if not optimized:
                        entry["dependencies"] = entry["source_files"]
                    del entry["source_files"]
                    assert None not in entry["dependencies"]

                    f_out.write(json.dumps(entry).encode("utf-8"))
                    f_out.write("\n".encode("utf-8"))

    Logger.debug("{}, num files: {}".format(name, num_entries))


def main():
    parser = argparse.ArgumentParser(
        "Run TypeScript Type Checker on a dataset of project"
    )
    parser.add_argument("--repos", default="data/Repos")
    parser.add_argument("--repos_cleaned", default="data/Repos-processed")
    parser.add_argument("--out_dir", default="data/out")
    parser.add_argument("--num_threads", default=12)
    parser.add_argument("--include_js", default=False, action="store_true")
    args = parser.parse_args()

    Logger.init()
    random.seed(42)

    args.repos = os.path.abspath(args.repos)
    args.repos_cleaned = os.path.abspath(args.repos_cleaned)
    args.out_dir = os.path.abspath(args.out_dir)

    paths = []
    for path in os.listdir(args.repos):
        if path == "SAP":
            continue

        for p in find_top_level_projects(os.path.join(args.repos, path)):
            paths.append((args.repos, args.repos_cleaned, p))

    if os.path.exists(args.repos_cleaned):
        shutil.rmtree(args.repos_cleaned)
    if not os.path.exists(args.repos_cleaned):
        os.makedirs(args.repos_cleaned)
        with multiprocessing.Pool(args.num_threads) as pool:
            pool.starmap(process_project, paths)

    # (optional) optimize dependencies
    # paths = glob.glob('{}/**/*.json.gz'.format(args.repos_cleaned), recursive=True)
    # with multiprocessing.Pool(args.num_threads) as pool:
    #     for path in paths:
    #         optimize_project(path, pool)

    save_dataset(args)


if __name__ == "__main__":
    main()
