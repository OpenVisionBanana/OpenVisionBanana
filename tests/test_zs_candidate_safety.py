import numpy as np
from scripts.analysis.audit_zs_candidate_safety import shape_stats, stable_random, confirmed_duplicate


def test_holes_are_not_confused_with_edges():
    mask=np.ones((20,20),bool);mask[0,:]=False;mask[10,10]=False
    report=shape_stats(mask)
    assert report['components']==2 and report['edge_components']==1
    assert report['interior_components']==1 and report['invalid_pixels']==21


def test_no_textureless_wall_appearance_rejection():
    a=dict(rgb=np.zeros((32,32,3)),rgb_std=0,depth=np.ones((32,32)),
        mask=np.ones((32,32),bool),normal=np.tile([0,0,1],(32,32,1)))
    assert not confirmed_duplicate(a,a)
    a['rgb_std']=10
    assert confirmed_duplicate(a,a)
    b={**a,'depth':a['depth']*1.2}
    assert not confirmed_duplicate(a,b)
    assert stable_random(['c','b','a'],2)==stable_random(['a','c','b'],2)
