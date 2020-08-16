#!/usr/bin/env python
import argparse
import collections
import glob
import multiprocessing
import os
import random
import re
import sys
import time

import pygments
from datasketch import MinHash
from datasketch import MinHashLSH
from datasketch.hashfunc import sha1_hash32
from nltk.util import ngrams
from pygments.lexers import guess_lexer_for_filename

from robustcode.dedup.fastdupes import multiglob_compile


def file_to_tokens(filename, stop_tokens):
    return lex_file(filename, stop_tokens)


def lex_file(filename, stop_tokens=None):
    stop_tokens = stop_tokens if stop_tokens else []
    with open(filename, "r") as f:
        content = f.read()
        lexer = guess_lexer_for_filename(filename, content)
        tokens = pygments.lex(content, lexer)
        tokens = [
            v
            for k, v in tokens
            if not repr(k).startswith("Token.Comment.") and v not in stop_tokens
        ]
        return tokens


def lex_file_and_hash(filename, stop_tokens):
    unique_tokens = set(lex_file(filename, stop_tokens))
    return [sha1_hash32(v.encode("utf-8")) for v in unique_tokens]


def hash_file_tokens(tokens, ngram=2):
    unique_tokens = set(str(x) for x in ngrams(tokens, ngram))
    return [sha1_hash32(v.encode("utf-8")) for v in unique_tokens]


def read_lines(filename):
    with open(filename, "r") as f:
        return re.split(",| |\n|\t", f.read())


def jaccard_similarity(A, B):
    A = set(A)
    B = set(B)
    return len(A.intersection(B)) / len(A.union(B))


def generate_stop_tokens(files):
    sys.stderr.write("Generating Stop Tokens...\n")
    start = time.time()

    with multiprocessing.Pool(16) as pool:
        per_file_tokens = pool.starmap(file_to_tokens, [(file, []) for file in files])

    counter = collections.Counter()
    for tokens in per_file_tokens:
        for token in set(tokens):
            counter[token] += 1

    print("\tNumber of Unique values: {}".format(len(counter)))
    print("Stop Tokens:")
    stop_tokens = set()
    for v, count in counter.most_common():
        if count < len(files) * 0.6:
            break

        stop_tokens.add(v)
        print("\t{}: {}".format(repr(v), count))
    print("\tNumber of stop tokens: {}".format(len(stop_tokens)))
    sys.stderr.write("Done in {} s\n".format(time.time() - start))
    return stop_tokens


class Dataset:
    def __init__(self, file_names, per_file_tokens):
        self.file_names = file_names
        self.per_file_tokens = per_file_tokens

        empty_files = [
            idx for idx, tokens in enumerate(self.per_file_tokens) if len(tokens) == 0
        ]
        if empty_files:
            print("Removing {} empty files".format(len(empty_files)))
            self.remove_idxs(empty_files)

    def jaccard_similarity(self, first, second):
        return jaccard_similarity(
            self.per_file_tokens[first], self.per_file_tokens[second]
        )

    def remove_idxs(self, idxs):
        self.file_names = [
            v for idx, v in enumerate(self.file_names) if idx not in idxs
        ]
        self.per_file_tokens = [
            v for idx, v in enumerate(self.per_file_tokens) if idx not in idxs
        ]


class ParsedDataset(Dataset):
    def __init__(self, root, extension, exclude_re):
        file_names = sorted(ParsedDataset.__find_files(root, extension, exclude_re))
        stop_tokens = generate_stop_tokens(random.sample(file_names, 200))
        file_names, per_file_tokens = ParsedDataset.index_files(file_names, stop_tokens)
        super(ParsedDataset, self).__init__(file_names, per_file_tokens)

    @staticmethod
    def __find_files(root, ext, exclude_re):
        files = []
        for idx, file in enumerate(
            glob.iglob(os.path.join(root, "**/*." + ext), recursive=True)
        ):
            if not os.path.isfile(file):
                continue
            if exclude_re.match(file):
                continue
            files.append(file)
            if idx % 16 == 0:
                sys.stderr.write(
                    "\rGathering file paths to compare... ({} files examined)".format(
                        idx
                    )
                )
        sys.stderr.write(
            "\rGathering file paths to compare... ({} files examined)\n".format(
                len(files)
            )
        )
        return files

    @staticmethod
    def index_files(file_names, stop_tokens):
        start = time.time()
        sys.stderr.write("Indexing Dataset...\n")
        sys.stderr.write("\tTokenizing Files...\n")
        with multiprocessing.Pool(16) as pool:
            per_file_tokens = pool.starmap(
                lex_file_and_hash, [(file, stop_tokens) for file in file_names]
            )

        sys.stderr.write("\tIndexing Tokens...\n")
        empty_files_idx = set()
        for idx, tokens in enumerate(per_file_tokens):
            if not tokens:
                empty_files_idx.add(idx)

        sys.stderr.write(
            "\tRemoving {} files with empty tokens\n".format(len(empty_files_idx))
        )
        file_names = [
            v for idx, v in enumerate(file_names) if idx not in empty_files_idx
        ]
        per_file_tokens = [
            v for idx, v in enumerate(per_file_tokens) if idx not in empty_files_idx
        ]
        sys.stderr.write("Done in {} s\n".format(time.time() - start))
        return file_names, per_file_tokens


def hash_fnc(x):
    return x


def compute_min_hash(tokens, num_perm, seed):
    m = MinHash(num_perm=num_perm, hashfunc=hash_fnc, seed=seed)
    for s in tokens:
        m.update(s)
    return m


def compute_min_hashes(dataset, num_perm, seed):
    sys.stderr.write("Computing MinHashes...\n")
    start = time.time()
    with multiprocessing.Pool(16) as pool:
        minhashes = pool.starmap(
            compute_min_hash,
            [(tokens, num_perm, seed) for tokens in dataset.per_file_tokens],
        )
    sys.stderr.write("Done in {} s\n".format(time.time() - start))
    return minhashes


SimilarFiles = collections.namedtuple(
    "SimilarFiles", ["score", "first_idx", "second_idx"]
)


def has_similar_files(lsh_index, dataset, idx, min_hash, threshold, stats):
    return (
        find_similar_file(lsh_index, dataset, idx, min_hash, threshold, stats)
        is not None
    )


def find_similar_file(lsh_index, dataset, idx, min_hash, threshold, stats):
    for candidate_idx in lsh_index.query(min_hash):
        stats["candidates"] += 1
        if dataset.jaccard_similarity(idx, candidate_idx) > threshold:
            stats["~similar"] += 1
            return candidate_idx
    return None


def compute_similar_files(dataset, num_perm, seed, threshold):
    minhashes = compute_min_hashes(dataset, num_perm, seed)
    lsh_index = MinHashLSH(num_perm=num_perm, threshold=threshold - 0.2)

    stats = collections.Counter()
    similar_idxs = set()
    similar_files = []
    for idx, m in enumerate(minhashes):
        similar_idx = find_similar_file(lsh_index, dataset, idx, m, threshold, stats)
        if similar_idx is not None:
            similar_files.append(
                (dataset.file_names[idx], dataset.file_names[similar_idx])
            )
            similar_idxs.add(idx)
            continue
        else:
            lsh_index.insert(idx, m)

    print(
        "Removing {} similar files from {} files".format(
            len(similar_idxs), len(minhashes)
        )
    )
    print("Stats: {}".format(stats))
    removed_idxs = [dataset.file_names[idx] for idx in similar_idxs]
    dataset.remove_idxs(similar_idxs)
    return removed_idxs, similar_files


def compute_similar_files_exact(dataset, threshold):
    stats = collections.Counter()
    similar_idxs = set()
    for first in range(len(dataset.files)):
        for second in range(0, first):
            if second in similar_idxs:
                continue
            stats["candidates"] += 1
            if dataset.jaccard_similarity(first, second) > threshold:
                stats["similar"] += 1
                similar_idxs.add(second)
                # print('\t', first, second)
                break

        if first % 16 == 0:
            sys.stderr.write(
                "\rComputing exact similarity... ({} files processed)".format(first)
            )
    sys.stderr.write(
        "\rComputing exact similarity... ({} files examined)\n".format(
            len(dataset.files)
        )
    )

    print(
        "Removing {} similar files from {} files".format(
            len(similar_idxs), len(dataset.files)
        )
    )
    print("Stats: {}".format(stats))


def main():
    parser = argparse.ArgumentParser(
        "Find similar files using Locality-Sensitive Hashing"
    )
    parser.add_argument("dataset", type=str, help="Directory to search for files")
    parser.add_argument("language", choices=["java", "js", "coffee", "py"])
    parser.add_argument(
        "--num_perm",
        type=int,
        default=32,
        help="The number of permutation functions used by the MinHash to be indexed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="The Jaccard similarity threshold between 0.0 and 1.0",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        action="append",
        help='Patterns (supporting regex) to exclude from the dataset. Example (--exclude "*[.-]min.js")',
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print('Error! Directory "{}" does not exists.'.format(args.dataset))
        exit(0)

    ignore_re = multiglob_compile(args.exclude if args.exclude else [], prefix=False)
    dataset = ParsedDataset(args.dataset, args.language, ignore_re)
    # compute_similar_files_exact(dataset, args.threshold)

    i = 0
    while True:
        print("Iteration #{}".format(i))
        removed_idxs, _ = compute_similar_files(
            dataset, args.num_perm, i, args.threshold
        )
        if len(removed_idxs) == 0:
            break
        i += 1

    with open(args.language + ".files", "w") as f:
        for filename in dataset.files:
            f.write("{}\n".format(os.path.abspath(filename)))


if __name__ == "__main__":
    main()
