import numpy as np


def assert_model(true_model, created_model, rtol=1.e-4, atol=1.e-6):
    assert len(true_model['model']) == len(created_model['model'])
    for i in range(len(true_model['model'])):
        true_tree = true_model['model'][i]
        created_tree = created_model['model'][i]
        assert np.allclose(
            true_tree['weights'], created_tree['weights'], rtol=rtol, atol=atol), "weights for tree{0}".format(i)
        assert true_tree['nodes'] == created_tree['nodes'], "nodes for tree {0}".format(
            i)
