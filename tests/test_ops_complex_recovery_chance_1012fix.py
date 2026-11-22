"""Chance-corrected complex recovery for four- and eight-member report controls."""
import numpy as np
import pytest

from spacr import ops_clusters as C


def _recovery(labels,members):
    """Call the sole production recovery owner with a full scored background."""
    names=[str(i) for i in range(len(labels))]
    group=[str(i) for i in members]
    return C.complex_metrics(names,np.asarray(labels),{'complex':group},{'complex':1})


def test_existing_fisher_tail_remains_two_sided_by_default():
    """The original two-sided table probability remains distinct from enrichment-only recovery."""
    assert C._fisher_p(2,3,3,n_background=7)==pytest.approx(17/35)
    assert C._fisher_p(2,3,3,n_background=7,alternative='greater')==pytest.approx(13/35)


@pytest.mark.parametrize('size',[4,8])
def test_two_members_recover_four_and_eight_member_sets_when_chance_is_small(size):
    """An informative shared pair can recover either set size at many small clusters."""
    labels=np.arange(6400)//10
    members=[0,1]+[10*i for i in range(2,size)]
    result=_recovery(labels,members)
    assert result['REPORT_complexes']==1
    assert result['complexes_recovered_pct']==100.0
    assert result['CORUM_partner_in_cluster_pct']==pytest.approx(200/size)


def test_large_clusters_and_single_cluster_partitions_do_not_supply_recovery_evidence():
    """Coincidental overlap in a large background cluster cannot satisfy the new recovery rule."""
    labels=np.r_[np.zeros(3200,int),np.arange(1,3201)]
    assert _recovery(labels,[0,1,3200,3201,3202,3203,3204,3205])['complexes_recovered_pct']==0
    assert _recovery(np.zeros(6400,int),range(8))['complexes_recovered_pct']==0


def test_searching_more_clusters_changes_the_family_correction():
    """Identical local overlap is assessed against the actual number of searched clusters."""
    small_family=np.r_[np.zeros(10,int),np.ones(6390,int)]
    large_family=np.r_[np.zeros(10,int),np.arange(1,6391)]
    members=[0,1,10,11,12,13,14,15]
    assert _recovery(small_family,members)['complexes_recovered_pct']==100
    assert _recovery(large_family,members)['complexes_recovered_pct']==0


def test_repeated_members_do_not_inflate_set_size_or_support():
    """Membership is a gene set, so repeated annotations retain the same statistic."""
    labels=np.arange(6400)//10
    assert _recovery(labels,[0,1,20,30])==_recovery(labels,[0,0,1,1,20,30,30])


@pytest.mark.parametrize('size,n,clusters',[(4,10000,10),(8,6400,640)])
def test_shuffled_sets_have_recovery_near_the_nominal_five_percent(size,n,clusters):
    """Fixed balanced null partitions give the predicted conservative rate near five percent."""
    rng=np.random.default_rng(C.SWEEP_CONTROL_SEED)
    names=[str(i) for i in range(n)]
    labels=np.arange(n)//(n//clusters)
    groups={str(i):[names[j] for j in rng.choice(n,size,replace=False)] for i in range(20000)}
    result=C.complex_metrics(names,labels,groups,dict.fromkeys(groups,1))
    assert result['REPORT_complexes']==20000
    assert 3.0<=result['complexes_recovered_pct']<=7.0
