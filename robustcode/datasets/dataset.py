import gzip
import json
import multiprocessing
import os
import sys
import tempfile
from abc import ABC
from abc import abstractmethod

import torchtext

import robustcode.parsers.parser as parser
from robustcode.analysis.graph import AstPrinter
from robustcode.analysis.graph import AstTree
from robustcode.analysis.graph import TypeScriptGraphAnalyzer
from robustcode.util.misc import Logger


class Config:

    DATASETS = {
        # Type Inference
        "deeptyperast": {"loader": "deeptyperast"},
        "deeptyperast_10k": {"loader": "deeptyperast", "num_samples": 10000},
        "deeptyperast_4k": {"loader": "deeptyperast", "num_samples": 4000},
        "deeptyperast_2k": {"loader": "deeptyperast", "num_samples": 2000},
        "deeptyperast_1k": {"loader": "deeptyperast", "num_samples": 1000},
    }

    @staticmethod
    def list_datasets():
        return Config.DATASETS.keys()

    @staticmethod
    def get_dataset(name):
        assert name in Config.DATASETS
        params = Config.DATASETS[name]
        params["name"] = name
        return Config(**params)

    def __init__(
        self,
        name,
        loader=None,
        shuffle=False,
        num_samples=None,
        valid_ratio=0.1,
        test_ratio=0.1,
        train="train.json",
        valid="valid.json",
        test="test.json",
        eval_first=True,
        compressed=True,
        language=None,
    ):
        assert loader in DATA_LOADERS, 'Unknown loader "{}". Known loaders: {}'.format(
            loader, DATA_LOADERS.keys()
        )
        self.name = name
        self.shuffle = shuffle
        self.loader = loader
        self.num_samples = num_samples
        self.eval_first = eval_first
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.compressed = compressed
        self.train = train
        self.valid = valid
        self.test = test
        if compressed:
            if not self.train.endswith(".gz"):
                self.train = self.train + ".gz"
            if not self.valid.endswith(".gz"):
                self.valid = self.valid + ".gz"
            if not self.test.endswith(".gz"):
                self.test = self.test + ".gz"

        self.language = language

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_json(self):
        return json.dumps(self.__dict__)

    def save_to_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.to_json())

    def init(self, in_path, out_path):
        out_path = os.path.join(out_path, self.name)
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        config_path = os.path.join(out_path, "config.json")
        if os.path.exists(config_path):
            existing_config = Config.load_from_file(config_path)
            if existing_config is not None and existing_config == self:
                Logger.debug("Dataset already preprocessed.")
                return
            else:
                Logger.debug("Configs do not match. Overwriting existing dataset.")

        DATA_LOADERS[self.loader].preprocess_dataset(in_path, out_path, self)

        self.save_to_file(config_path)

    @staticmethod
    def load_from_file(filename):
        assert os.path.exists(filename)
        with open(filename, "r") as f:
            return Config.from_json(f.read().strip())

    @staticmethod
    def from_json(data):
        try:
            if isinstance(data, str):
                return Config(**json.loads(data))
            elif isinstance(data, dict):
                return Config(**data)
            else:
                assert False
        except:  # noqa: E722
            return None


def split_dataset(filenames, config):
    test_size = (
        int(len(filenames) * config.test_ratio)
        if config.test_ratio < 1.0
        else config.test_ratio
    )
    valid_size = (
        int(len(filenames) * config.valid_ratio)
        if config.valid_ratio < 1.0
        else config.valid_ratio
    )

    if config.eval_first:
        # test files are taken from the beginning to allow increasing training data while keeping the test set unchanged
        test_filenames = filenames[:test_size]
        valid_filenames = filenames[test_size : test_size + valid_size]
        train_filenames = filenames[test_size + valid_size :]
    else:
        test_filenames = filenames[-test_size:]
        valid_filenames = filenames[-(test_size + valid_size) : -test_size]
        train_filenames = filenames[: -(test_size + valid_size)]
    assert len(train_filenames) + len(valid_filenames) + len(test_filenames) == len(
        filenames
    ), "train: {}, valid: {}, test: {} != total {}".format(
        len(train_filenames), len(valid_filenames), len(test_filenames), len(filenames)
    )

    return {"train": train_filenames, "valid": valid_filenames, "test": test_filenames}


class DatasetLoader(ABC):
    @abstractmethod
    def preprocess_dataset(self, in_path: str, out_path: str, config: Config):
        """
        Takes a raw dataset (stored in folder `in_path`) and produces
        a preprocessed dataset. The preprocessed dataset is stored in
        folder `out_path` in three files:
          - os.path.join(out_path, confing.train)
          - os.path.join(out_path, confing.valid)
          - os.path.join(out_path, confing.test)

        Each file contains one sample per line represented as json. For example:

        >>> head -n 1 <out_path>/train.json
        >>> {'id': 'test.java', 'target':'5', 'types': ['A', 'B', 'C'], 'values': ['1', '2', '3']}

        where the actual fields (e.g., 'target', 'types', 'values') depend on the given task

        Args:
            in_path: path to the raw dataset (e.g., GitHub repositories)
            out_path: path where to store the processed dataset
            config: configuration describing the dataset

        """

        pass


class DeepTyperAstLoader(DatasetLoader):
    @staticmethod
    def process_ast(tree, idx=0, analyze=False, max_size=3000):
        excludes = ["<null>"]
        fields = ["target", "gold"]

        tree = AstTree.fromJson(
            tree,
            analyzer=TypeScriptGraphAnalyzer() if analyze else None,
            field_names=fields,
        )
        if max_size > 0 and len(tree.nodes) > max_size:
            return None, None
        types = [node.type for node in tree.nodes]
        values = [str(node.value) if node.value else "<null>" for node in tree.nodes]

        target = [node.fields.get("target", "<null>") for node in tree.nodes]
        mask_valid = [1 if t not in excludes else 0 for t in target]
        mask_gold = [1 if "gold" in node.fields else 0 for node in tree.nodes]
        gold_type = [node.fields.get("gold", "<null>") for node in tree.nodes]
        depth = [node.depth() for node in tree.nodes]
        pos = [min(16, node.pos_in_parent()) for node in tree.nodes]

        data = {
            "id": idx,
            "ast_values": values,
            "ast_types": types,
            "target_full": target,
            "mask_valid_full": mask_valid,
            "mask_gold": mask_gold,
            "gold_type": gold_type,
            "pos": pos,
            # debugging
            "depth": depth,
            # used by type inference
            "dependencies": [],
        }
        if analyze:
            per_type_edges = tree.compute_all_edges()
            for edge_type, values in per_type_edges.items():
                data[edge_type + "_src"] = [v[0] for v in values]
                data[edge_type + "_tgt"] = [v[1] for v in values]
        return data, tree

    @staticmethod
    def analyze_tree(tree, filename, dependencies, analyze=False, strict=False):
        if isinstance(tree, AstTree):
            tree = tree.root
        code = AstPrinter.toTS(tree, include_types=True)
        if code is None:
            return None, None
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", dir=os.path.dirname(filename), delete=True
        ) as f:
            # Convert AST to code and write to a file
            f.write(code)
            f.flush()

            ast_json = parser.parse_file_server(
                f.name,
                data={
                    "remove_types": True,
                    # remove self dependencies
                    "deps": [d for d in dependencies if d != filename],
                },
            )
        if isinstance(ast_json, dict) and "ast" in ast_json:
            if strict and ast_json["syntactic_errors"] > 0:
                return None, None
            ast_json = ast_json["ast"]
        return DeepTyperAstLoader.process_ast(ast_json, analyze=analyze)

    @staticmethod
    def process_line(line):
        entry = json.loads(line)
        data, tree = DeepTyperAstLoader.process_ast(
            entry["ast"], idx=entry["filename"], analyze=True
        )
        if data is None:
            # num_skipped += 1
            return None

        """
         Parse serialized file again to check that the results are the same.
         This is important in case the file is analyzed repeatedly 
         while generating adversarial examples.
         
         For TypeScript/JavaScript there are some corner cases where all types do not match.
         One of them is that some imported types depend on the filename which changes since
         the modifier tree is save to a temporary file.
        """
        # _, ref_tree = DeepTyperAstLoader.analyze_tree(tree, data['id'], entry['dependencies'], strict=True)
        # if ref_tree is None:
        #     # num_parse_errors += 1
        #     return None
        # if not tree.root.tree_equal(ref_tree.root):
        #     # num_tree_diffs += 1
        #     return None

        data["dependencies"] = entry["dependencies"]
        data = json.dumps(data) + "\n"
        return data

    def __process_lines(self, lines, pool, config, f_out):
        num_skipped = 0
        # uncomment for sequential version
        # for data in [DeepTyperAstLoader.process_line(line) for line in lines]:
        for data in pool.imap(DeepTyperAstLoader.process_line, lines):
            if data is None:
                num_skipped += 1
                continue
            if config.compressed:
                f_out.write(data.encode("utf-8"))
            else:
                f_out.write(data)
        return num_skipped

    def preprocess_dataset(self, in_path, out_path, config):
        if sys.getrecursionlimit() < 10000:
            # default recursion limit 1000 is too small for large trees
            sys.setrecursionlimit(10000)

        for split_name in ["valid", "train", "test"]:
            dataset_path = os.path.join(out_path, getattr(config, split_name))

            num_skipped = 0
            num_total = 0
            num_tree_diffs = 0
            num_parse_errors = 0
            f_out = (
                open(dataset_path, "w")
                if not config.compressed
                else gzip.open(dataset_path, "wb")
            )
            with multiprocessing.Pool(8) as pool:
                with gzip.open(
                    os.path.join(in_path, "out", split_name + ".json.gz"), "rb"
                ) as f:
                    lines = []
                    for idx, line in enumerate(f):
                        sys.stderr.write("\r{}".format(idx))
                        lines.append(line)
                        num_total += 1

                        if (
                            config.num_samples is not None
                            and num_total > config.num_samples
                        ):
                            break

                        if len(lines) >= 1000:
                            num_skipped += self.__process_lines(
                                lines, pool, config, f_out
                            )
                            lines = []

                    num_skipped += self.__process_lines(lines, pool, config, f_out)

            f_out.close()

            sys.stderr.write("\n")
            print(
                "{}: Skipped {}/{} files with empty target, {}/{} files with non-matching trees, {}/{} parse errors".format(
                    split_name,
                    num_skipped,
                    num_total,
                    num_tree_diffs,
                    num_total,
                    num_parse_errors,
                    num_total,
                )
            )


DATA_LOADERS = {
    "deeptyperast": DeepTyperAstLoader(),
}


"""
Extends torchtext to support reading compressed files
"""


class TabularCompressedDataset(torchtext.data.Dataset):
    """Defines a Dataset of columns stored in CSV, TSV, or JSON format."""

    def add_field(self, name, field):
        self.fields[name] = field
        for sample in self:
            field.extend_sample(sample)

    def __init__(
        self, path, format, fields, skip_header=False, csv_reader_params={}, **kwargs
    ):
        """Create a TabularDataset given a path, file format, and field list.

        Arguments:
            path (str): Path to the data file.
            format (str): The format of the data file. One of "CSV", "TSV", or
                "JSON" (case-insensitive).
            fields (list(tuple(str, Field)) or dict[str: tuple(str, Field)]:
                If using a list, the format must be CSV or TSV, and the values of the list
                should be tuples of (name, field).
                The fields should be in the same order as the columns in the CSV or TSV
                file, while tuples of (name, None) represent columns that will be ignored.

                If using a dict, the keys should be a subset of the JSON keys or CSV/TSV
                columns, and the values should be tuples of (name, field).
                Keys not present in the input dictionary are ignored.
                This allows the user to rename columns from their JSON/CSV/TSV key names
                and also enables selecting a subset of columns to load.
            skip_header (bool): Whether to skip the first line of the input file.
            csv_reader_params(dict): Parameters to pass to the csv reader.
                Only relevant when format is csv or tsv.
                See
                https://docs.python.org/3/library/csv.html#csv.reader
                for more details.
        """
        format = format.lower()
        make_example = {
            "json": torchtext.data.example.Example.fromJSON,
            "dict": torchtext.data.example.Example.fromdict,
            "tsv": torchtext.data.example.Example.fromCSV,
            "csv": torchtext.data.example.Example.fromCSV,
        }[format]

        with gzip.open(os.path.expanduser(path), "rb") as f:
            if format == "csv":
                reader = torchtext.utils.unicode_csv_reader(f, **csv_reader_params)
            elif format == "tsv":
                reader = torchtext.utils.unicode_csv_reader(
                    f, delimiter="\t", **csv_reader_params
                )
            else:
                reader = f

            if format in ["csv", "tsv"] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError(
                        "When using a dict to specify fields with a {} file,"
                        "skip_header must be False and"
                        "the file must have a header.".format(format)
                    )
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = torchtext.data.dataset.partial(
                    make_example, field_to_index=field_to_index
                )

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularCompressedDataset, self).__init__(examples, fields, **kwargs)
