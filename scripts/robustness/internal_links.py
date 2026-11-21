"""Date-aware family membership and any-shared-assignee comparisons."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def component_at(tree, patent_ids, days):
    ids = tree['ids']
    position = np.searchsorted(ids, np.asarray(patent_ids, dtype=np.int64))
    valid = position < len(ids)
    valid[valid] &= ids[position[valid]] == np.asarray(patent_ids, dtype=np.int64)[valid]
    node = np.zeros(len(position), dtype=np.int32)
    node[valid] = position[valid]
    for ancestor in tree['ancestor'][::-1]:
        parent = ancestor[node]
        selected = np.flatnonzero(valid & (parent >= 0))
        selected = selected[tree['day'][parent[selected]] <= days[selected]]
        node[selected] = parent[selected]
    node[~valid] = -1
    return node


class FamilyIndex:
    """Euler leaf intervals let each query use its own observation date."""
    def __init__(self, tree):
        self.tree = tree
        n, parent = len(tree['ids']), tree['ancestor'][0]
        children = np.full((len(parent), 2), -1, np.int32)
        edges = np.flatnonzero(parent >= 0)
        edges = edges[np.argsort(parent[edges], kind='stable')]
        if len(edges) % 2 or not np.array_equal(parent[edges[::2]], parent[edges[1::2]]):
            raise ValueError('Family cache must be a binary temporal union tree')
        children[parent[edges[::2]], 0] = edges[::2]
        children[parent[edges[1::2]], 1] = edges[1::2]
        rank = np.full(n, -1, np.int32)
        stack, counter = np.flatnonzero(parent < 0).tolist(), 0
        while stack:
            node = stack.pop()
            if node < n:
                rank[node] = counter
                counter += 1
            else:
                stack.extend([int(children[node, 1]), int(children[node, 0])])
        if counter != n or (rank < 0).any():
            raise ValueError('Invalid family-tree leaf coverage')
        low, high = np.empty(len(parent), np.int32), np.empty(len(parent), np.int32)
        low[:n] = high[:n] = rank
        for node in range(n, len(parent)):
            a, b = children[node]
            if a < 0 or b < 0 or max(a, b) >= node:
                raise ValueError('Family merges must follow their children')
            low[node], high[node] = min(low[a], low[b]), max(high[a], high[b])
        self.rank, self.low, self.high = rank, low, high

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls({name: archive[name] for name in ['ids', 'ancestor', 'day']})

    def enrich(self, frame):
        result = frame.copy()
        ids = result.patent_id.to_numpy(np.int64)
        dates = pd.to_datetime(result.grant_date, errors='raise')
        if dates.isna().any():
            raise ValueError('Family comparison dates must be nonmissing')
        days = dates.to_numpy('datetime64[D]').astype(np.int32)
        node = component_at(self.tree, ids, days)
        leaf = np.searchsorted(self.tree['ids'], ids)
        valid = leaf < len(self.tree['ids'])
        valid[valid] &= self.tree['ids'][leaf[valid]] == ids[valid]
        ranks = np.full(len(ids), -1, np.int32)
        ranks[valid] = self.rank[leaf[valid]]
        result['family_leaf_rank'] = ranks
        result['family_low_at_grant'] = np.where(node >= 0, self.low[np.maximum(node, 0)], 1)
        result['family_high_at_grant'] = np.where(node >= 0, self.high[np.maximum(node, 0)], 0)
        return result


class OwnerIndex:
    """Keep all observed owners, with unknown ownership separate from no overlap."""
    def __init__(self, links):
        column = 'entity_id' if 'entity_id' in links else 'assignee_id'
        if 'patent_id' not in links or column not in links:
            raise ValueError('Assignee links require patent_id and entity_id or assignee_id')
        links = links[['patent_id', column]].dropna().copy()
        links['patent_id'] = links.patent_id.astype(str)
        links[column] = links[column].astype(str)
        links = links.drop_duplicates().sort_values(['patent_id', column])
        codes, _ = pd.factorize(links[column], sort=True)
        sizes = links.groupby('patent_id', sort=True).size()
        self.ids = sizes.index.to_numpy(str)
        self.offsets = np.r_[0, np.cumsum(sizes.to_numpy(np.int64))]
        self.entities = codes.astype(np.int64)

    def rows(self, ids):
        ids = np.asarray(ids, dtype=str)
        rows = np.searchsorted(self.ids, ids)
        valid = rows < len(self.ids)
        valid[valid] &= self.ids[rows[valid]] == ids[valid]
        rows[~valid] = -1
        return rows

    def padded(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        sizes = np.zeros(len(rows), np.int64)
        known = rows >= 0
        sizes[known] = self.offsets[rows[known] + 1] - self.offsets[rows[known]]
        width = int(sizes.max()) if len(sizes) else 0
        result = np.full((len(rows), max(1, width)), -1, np.int64)
        for index in np.flatnonzero(known):
            result[index, :sizes[index]] = self.entities[self.offsets[rows[index]]:self.offsets[rows[index] + 1]]
        return result

    def overlap(self, query_rows, history_rows):
        q, h = self.padded(query_rows), self.padded(history_rows)
        result = np.zeros((len(q), len(h)), bool)
        for column in range(q.shape[1]):
            result |= (q[:, column, None] >= 0) & (q[:, column, None, None] == h[None, :, :]).any(axis=2)
        return result


def historical_family_mask(query, history):
    leaves = history.family_leaf_rank.to_numpy()
    return ((leaves[None, :] >= 0) &
        (leaves[None, :] >= query.family_low_at_grant.to_numpy()[:, None]) &
        (leaves[None, :] <= query.family_high_at_grant.to_numpy()[:, None]))


def future_family_mask(query, history):
    leaves = query.family_leaf_rank.to_numpy()
    return ((leaves[:, None] >= 0) &
        (leaves[:, None] >= history.family_low_at_grant.to_numpy()[None, :]) &
        (leaves[:, None] <= history.family_high_at_grant.to_numpy()[None, :]))
