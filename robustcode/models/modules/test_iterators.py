import torch

from robustcode.models.modules.iterators import BPTTIterator


def test_BPTTIterator():
    x = torch.rand(100, 40)
    chunks = []
    for chunk in BPTTIterator(x, bptt_len=10, batch_first=False):
        assert chunk.size() == (10, 40)
        chunks.append(chunk)
    xs = torch.cat(chunks)
    assert torch.all(x == xs)


def test_BPTTIterator_large_step():
    x = torch.rand(100, 40)
    chunks = []
    for chunk in BPTTIterator(x, bptt_len=200, batch_first=False):
        assert chunk.size() == (100, 40)
        chunks.append(chunk)
    xs = torch.cat(chunks)
    assert torch.all(x == xs)


def test_BPTTIterator_batch_first():
    x = torch.rand(40, 100)
    chunks = []
    for chunk in BPTTIterator(x, bptt_len=10, batch_first=True):
        assert chunk.size() == (40, 10)
        chunks.append(chunk)
    xs = torch.cat(chunks, dim=1)
    assert torch.all(x == xs)


def test_BPTTIterator_large_step():
    x = torch.rand(40, 100)
    chunks = []
    for chunk in BPTTIterator(x, bptt_len=200, batch_first=True):
        assert chunk.size() == (40, 100)
        chunks.append(chunk)
    xs = torch.cat(chunks)
    xs = torch.cat(chunks, dim=1)
    assert torch.all(x == xs)
