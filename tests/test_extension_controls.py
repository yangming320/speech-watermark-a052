import numpy as np
import pytest
from speechwm.extension_controls import common_indices,speaker_bootstrap

def test_common_mask_uses_all_rates_and_both_labels_on_identical_indices():
    masks=[np.array([1,1,1,0],bool),np.array([1,0,1,1],bool),np.array([0,1,1,1],bool)]
    np.testing.assert_array_equal(common_indices(masks),[2])
    with pytest.raises(ValueError):common_indices([np.ones(2),np.ones(3)])

def test_cluster_bootstrap_resamples_whole_speaker_values_reproducibly():
    first,draws=speaker_bootstrap([-.1,.1,.2,-.2]);second,again=speaker_bootstrap([-.1,.1,.2,-.2])
    assert first==second and first['speakers']==4 and first['mean_delta_error']==0
    np.testing.assert_array_equal(draws,again)
    constant,_=speaker_bootstrap([-.1]*40);assert constant['ci_low']==pytest.approx(-.1) and constant['ci_high']==pytest.approx(-.1)
    with pytest.raises(ValueError):speaker_bootstrap([np.nan,1])
