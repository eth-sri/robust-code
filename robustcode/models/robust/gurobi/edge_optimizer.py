import collections
import itertools
from functools import partial
from typing import Iterable
from typing import Tuple

import dgl
import numpy as np
import torch

from robustcode.models.modules.dgl.iterators import GraphBatchIterator
from robustcode.models.modules.iterators import MiniBatch
from robustcode.util.misc import Logger
from robustcode.util.misc import Timer


class EdgeFilter:
    def __init__(self, valid_features: Iterable[Tuple[str, float, int]]):
        if valid_features is None:
            return
        self.valid_features, self.costs, self.counts = zip(*valid_features)
        sorted_ids = np.argsort(self.costs).tolist()
        sorted_ids.reverse()

        "remove edges seen less than 10 times"
        sorted_ids = [i for i in sorted_ids if self.counts[i] >= 20]
        # sum_costs = sum(self.costs)
        # sorted_ids = [i for i in sorted_ids if self.costs[i] > 0.005 * sum_costs]

        self.valid_features = [self.valid_features[i] for i in sorted_ids]
        self.costs = [self.costs[i] for i in sorted_ids]
        self.counts = [self.counts[i] for i in sorted_ids]
        self.seen_counts = collections.Counter()
        self.init_cache()

    def init_cache(self):
        node_types_u = []
        node_types_v = []
        edge_types = []
        for feature in self.valid_features:
            node_type_u, node_type_v, edge_type = list(map(int, feature.split("_")))
            node_types_u.append(node_type_u)
            node_types_v.append(node_type_v)
            edge_types.append(edge_type)

        # print(len(node_types_u), len(node_types_v), len(edge_types))
        self.node_types_u = torch.tensor(node_types_u).unsqueeze(dim=1).cuda()
        self.node_types_v = torch.tensor(node_types_v).unsqueeze(dim=1).cuda()
        self.edge_types = torch.tensor(edge_types).unsqueeze(dim=1).cuda()

    def clear_cache(self):
        self.node_types_u = None
        self.node_types_v = None
        self.edge_types = None

    def __len__(self):
        return len(self.valid_features)

    def print(self, dataset=None, edge_gen=None):
        Logger.debug("EdgeFilter")
        cumsum = 0
        cumsum_seen = 0
        total_seen = sum(self.seen_counts.values())
        for feature, cost, count in zip(self.valid_features, self.costs, self.counts):
            cumsum += count
            cumsum_seen += self.seen_counts[feature]

            if dataset is not None or edge_gen is not None:
                node_type_u, node_type_v, edge_type = feature.split("_")
                node_type_u = (
                    dataset.TYPES.vocab.itos[int(node_type_u)]
                    if dataset is not None
                    else node_type_u
                )
                node_type_v = (
                    dataset.TYPES.vocab.itos[int(node_type_v)]
                    if dataset is not None
                    else node_type_v
                )
                edge_type = (
                    edge_gen.id_to_edge_type[int(edge_type)]
                    if edge_gen is not None
                    else edge_type
                )

                feature = "{}_{}_{}".format(node_type_u, node_type_v, edge_type)

            Logger.debug(
                "\t{:>40s} cost: {:10.0f} ({:5.2f}%), count: {:10d} ({:5.2f}%), cumsum: {:6.2f}%, seen: {:6.2f}%".format(
                    feature,
                    cost,
                    cost * 100.0 / sum(self.costs),
                    count,
                    count * 100.0 / sum(self.counts),
                    cumsum * 100.0 / sum(self.counts),
                    (cumsum_seen * 100.0 / total_seen) if total_seen != 0 else 0,
                )
            )

    def filter_self_loops(self, u: torch.Tensor, v: torch.Tensor, eid: torch.Tensor):
        not_self_loops = u != v
        return u[not_self_loops], v[not_self_loops], eid[not_self_loops]

    def edges_to_remove(
        self, u: Iterable[int], v: Iterable[int], eid: Iterable[int], g: dgl.DGLGraph
    ):
        u, v, eid = self.filter_self_loops(u, v, eid)
        if u.numel() == 0:
            return []

        valid_features = torch.sum(
            (g.nodes[u].data["types"] == self.node_types_u)
            & (g.nodes[v].data["types"] == self.node_types_v)
            & (g.edges[eid].data["type"] == self.edge_types),
            dim=0,
        )
        invalid_features = valid_features == 0
        to_remove_edges_t = eid[invalid_features].tolist()

        return to_remove_edges_t

    @staticmethod
    def edge_features(
        edges: Iterable[Tuple[int, int]], g: dgl.DGLGraph, debug_info=None
    ):
        return EdgeFilter.edge_features_uv(
            [i for i, _ in edges], [j for _, j in edges], g, debug_info=debug_info
        )

    @staticmethod
    def edge_features_uv(
        u: Iterable[int], v: Iterable[int], g: dgl.DGLGraph, debug_info=None
    ):
        if len(u) == 0:
            return []
        node_types_u = g.nodes[u].data["types"].cpu().numpy()
        node_types_v = g.nodes[v].data["types"].cpu().numpy()
        edge_types = g.edges[u, v].data["type"].cpu().numpy()

        if debug_info is not None:
            debug_info += "u: {}, v: {}\n".format(u, v)
            debug_info += "node_types: {}, edge_types: {}\n".format(
                node_types_v, edge_types
            )

        # convert to string without spaces that can be used in the MIP solver
        return [
            "{}_{}_{}".format(type_u, type_v, edge_type)
            for type_u, type_v, edge_type in zip(node_types_u, node_types_v, edge_types)
        ]


class FilteredGraphIterator:
    def __init__(self, it: GraphBatchIterator, edge_filter: EdgeFilter):
        assert edge_filter is not None
        self.it = it
        self.edge_filter = edge_filter

    @staticmethod
    def from_iter(it, edge_filter):
        if edge_filter is None:
            return it
        return FilteredGraphIterator(it, edge_filter)

    @staticmethod
    def filter_batch(batch: MiniBatch, edge_filter: EdgeFilter):
        g = batch.X
        trees = dgl.unbatch(g)
        for tree in trees:
            u, v, eid = tree.all_edges(form="all")
            to_remove = edge_filter.edges_to_remove(u, v, eid, tree)
            tree.remove_edges(to_remove)
        trees = dgl.batch(trees)

        return MiniBatch(
            trees,
            batch.Y,
            batch.lengths,
            {mask: trees.ndata[mask] for mask in batch.masks.keys()},
            batch.P,
            batch.data,
            batch.ids,
        )

    def __iter__(self):
        for batch in self.it:
            yield self.filter_batch(batch, self.edge_filter)

    def init_epoch(self):
        self.it.init_epoch()

    def __len__(self):
        return len(self.it)


class EdgeOptimizer:
    Sample = collections.namedtuple("Sample", ["nodes", "edges", "inflow"])

    def __init__(self):
        self.samples = []
        self.edge_types = None
        self.id_to_edge_type = None

    def add_sample(self, nodes, edges, inflow):
        assert len(edges) >= 0
        self.samples.append(EdgeOptimizer.Sample(nodes, edges, inflow))

    def build_edge_types(self, samples):
        self.edge_types = collections.defaultdict(partial(next, itertools.count()))
        for sample in samples:
            for edge, edge_type in sample.edges.items():
                self.edge_types[edge_type]

        self.id_to_edge_type = {value: key for key, value in self.edge_types.items()}
        # print(self.edge_types)

    def solve(self, debug_info=None):
        import gurobipy as gb

        verbose = len(self.samples) > 1
        if verbose:
            Logger.debug("Number of samples: #{}".format(len(self.samples)))
        self.build_edge_types(self.samples)
        # Create optimization model
        m = gb.Model("netflow")

        timers = collections.defaultdict(Timer)
        if verbose:
            Logger.start_scope("Encoding Solver Model")
        cost = m.addVars(
            range(len(self.edge_types.values())),
            obj=1.0,
            name="cost",
            vtype=gb.GRB.INTEGER,
        )
        flows = []
        for idx, sample in enumerate(self.samples):
            timers["flow"].start()
            flow = m.addVars(
                sample.edges.keys(), name="flow_{}".format(idx), vtype=gb.GRB.INTEGER
            )
            timers["flow"].stop()
            flows.append(flow)

            # Arc-capacity constraints
            timers["cap"].start()
            m.addConstrs(
                (
                    flow[i, j] <= cost[self.edge_types[e_type]]
                    for (i, j), e_type in sample.edges.items()
                ),
                "cap_{}".format(idx),
            )
            timers["cap"].stop()

            # Flow-conservation constraints
            timers["node"].start()
            m.addConstrs(
                (
                    flow.sum("*", j) + sample.inflow.get(j, 0) == flow.sum(j, "*")
                    for j in sample.nodes
                ),
                "node_{}".format(idx),
            )
            timers["node"].stop()

        if verbose:
            for key, timer in timers.items():
                Logger.debug("{} {}".format(key, timer))
            Logger.end_scope()

            Logger.start_scope("Optimizing")
        m.write("file.lp")
        # disable logging
        m.Params.OutputFlag = 0
        m.optimize()
        if verbose:
            Logger.end_scope()

        # Print solution
        if m.status == gb.GRB.Status.OPTIMAL:
            edge_costs = collections.Counter()
            edge_counts = collections.Counter()
            for flow, sample in zip(flows, self.samples):
                solution = m.getAttr("x", flow)
                # print('\nOptimal flows:')
                for (i, j), e_type in sample.edges.items():
                    if solution[i, j] > 0:
                        # print('%s -> %s: %g' % (i, j, solution[i, j]))
                        edge_costs[e_type] += solution[i, j]
                        edge_counts[e_type] += 1

            valid_features = []
            solution = m.getAttr("x", cost)
            # print('Costs')
            for idx, c in enumerate(solution):
                # print('\t{} {} -> {} {:.2f} ({:.2f}%)'.format(idx, c, solution[c],
                #                                   edge_costs[self.id_to_edge_type[c]],
                #                                   edge_costs[self.id_to_edge_type[c]] * 100.0 / sum(edge_costs.values()))
                #       )
                if solution[c] > 0:
                    edge_type = self.id_to_edge_type[c]
                    valid_features.append(
                        (edge_type, edge_costs[edge_type], edge_counts[edge_type])
                    )
            if not valid_features:
                print("valid_features", valid_features)
                print(debug_info)
                exit(0)

            return EdgeFilter(valid_features)
        else:
            print(debug_info)
            print(m.status)
            print("The model is infeasible; computing IIS")

            for sample in self.samples[:5]:
                print(sample.inflow)
                print(sample.edges)
                print(sample.nodes)

            m.computeIIS()
            if m.IISMinimal:
                print("IIS is minimal\n")
            else:
                print("IIS is not minimal\n")
            print("\nThe following constraint(s) cannot be satisfied:")
            for c in m.getConstrs():
                if c.IISConstr:
                    print("%s" % c.constrName)
            exit(0)
