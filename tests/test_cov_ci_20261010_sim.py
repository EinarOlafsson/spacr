"""Queue-budget validation and writer cleanup in ``run_multiple_simulations``."""
from __future__ import annotations

import pytest

from spacr import sim as S
import spacr.database_concurrency as database_concurrency


class _Manager:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def list(self):
        return []


class _Writer:
    instances = []

    def __init__(self, path, commit, report, ram_gib):
        self.path = path
        self.ram_gib = ram_gib
        self.endpoint = object()
        self._thread = None
        self.events = []
        _Writer.instances.append(self)

    def start(self):
        self._thread = object()
        self.events.append('start')

    def cancel(self):
        self.events.append('cancel')

    def finish(self):
        self.events.append('finish')
        raise OSError('writer could not flush')


class _Pool:
    def __init__(self, workers, initializer=None, initargs=()):
        self.workers = workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starmap_async(self, function, arguments):
        list(arguments)
        raise KeyError('pool lost')


def _settings(tmp_path, **overrides):
    base = dict(
        replicates=1, avg_genes_per_well=[4], avg_cells_per_well=[20],
        classifier_accuracy=[0.9], avg_reads_per_gene=[100],
        sequencing_error=[0.01], well_ineq_coeff=[1.2], gene_ineq_coeff=[1.2],
        nr_plates=[1], number_of_genes=[30], number_of_active_genes=[10],
        number_of_control_genes=5, max_workers=1, src=str(tmp_path),
        plot=False, name='sweep', variable='classifier_accuracy',
        database_write_queue_gib=1)
    base.update(overrides)
    return base


@pytest.fixture
def fakes(monkeypatch):
    _Writer.instances = []
    monkeypatch.setattr(S, 'Manager', _Manager)
    monkeypatch.setattr(database_concurrency, '_DatabaseWriteQueue', _Writer)
    return _Writer


@pytest.mark.parametrize('budget', [-0.5, 64.5])
def test_queue_budget_outside_zero_to_64_gib_is_refused(tmp_path, fakes, budget):
    with pytest.raises(ValueError, match='between 0 and 64 GiB'):
        S.run_multiple_simulations(_settings(tmp_path, database_write_queue_gib=budget))
    assert fakes.instances == []


def test_pool_start_failure_skips_writer_cleanup_when_writer_never_started(
        tmp_path, fakes, monkeypatch):
    def refuse(*args, **kwargs):
        raise MemoryError('no pool')

    monkeypatch.setattr(S, 'Pool', refuse)
    with pytest.raises(MemoryError, match='no pool'):
        S.run_multiple_simulations(_settings(tmp_path))
    writer, = fakes.instances
    assert writer._thread is None
    assert writer.events == []
    assert writer.ram_gib == 1.0


def test_started_writer_is_cancelled_and_its_finish_error_does_not_mask_the_cause(
        tmp_path, fakes, monkeypatch):
    monkeypatch.setattr(S, 'Pool', _Pool)
    with pytest.raises(KeyError, match='pool lost'):
        S.run_multiple_simulations(_settings(tmp_path))
    writer, = fakes.instances
    assert writer.events == ['start', 'cancel', 'finish']
    assert writer.path.endswith('.simulation_write_queue')
