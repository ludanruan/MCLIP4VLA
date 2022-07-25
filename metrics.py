from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import pdb
import torch.nn.functional as F
from sklearn import metrics

def t2v_metrics(sims, query_masks=None, query_masks_class_t2v=None):
    """Compute retrieval metrics from a similiarity matrix.

    Args:
        sims (th.Tensor): N x M matrix of similarities between embeddings, where
             x_{i,j} = <text_embd[i], vid_embed[j]>
        query_masks (th.Tensor): mask any missing queries from the dataset (two videos
             in MSRVTT only have 19, rather than 20 captions)

    Returns:
        (dict[str:float]): retrieval metrics
    """
    assert sims.ndim == 2, "expected a matrix"

    num_queries, num_vids = sims.shape
    dists = -sims
    sorted_dists = np.sort(dists, axis=1)

    # The indices are computed such that they slice out the ground truth distances
    # from the psuedo-rectangular dist matrix
    queries_per_video = num_queries // num_vids
    gt_idx = [[np.ravel_multi_index([ii, jj], (num_queries, num_vids))
              for ii in range(jj * queries_per_video, (jj + 1) * queries_per_video)]
              for jj in range(num_vids)]
    gt_idx = np.array(gt_idx)
    gt_dists = dists.reshape(-1)[gt_idx.reshape(-1)]
    gt_dists = gt_dists[:, np.newaxis]
    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # --------------------------------
    # NOTE: Breaking ties
    # --------------------------------
    # We sometimes need to break ties (in general, these should occur extremely rarely,
    # but there are pathological cases when they can distort the scores, such as when
    # the similarity matrix is all zeros). Previous implementations (e.g. the t2i
    # evaluation function used
    # here: https://github.com/niluthpol/multimodal_vtt/blob/master/evaluation.py and
    # here: https://github.com/linxd5/VSE_Pytorch/blob/master/evaluation.py#L87) generally
    # break ties "optimistically".  However, if the similarity matrix is constant this
    # can evaluate to a perfect ranking. A principled option is to average over all
    # possible partial orderings implied by the ties. See # this paper for a discussion:
    #    McSherry, Frank, and Marc Najork,
    #    "Computing information retrieval performance measures efficiently in the presence
    #    of tied scores." European conference on information retrieval. Springer, Berlin, 
    #    Heidelberg, 2008.
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.8892&rep=rep1&type=pdf

    # break_ties = "optimistically"
    break_ties = "averaging"

    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            _, idx = np.unique(rows, return_index=True)
            cols = cols[idx]
        elif break_ties == "averaging":
            # fast implementation, based on this code:
            # https://stackoverflow.com/a/49239335
            locs = np.argwhere((sorted_dists - gt_dists) == 0)

            # Find the split indices
            steps = np.diff(locs[:, 0])
            splits = np.nonzero(steps)[0] + 1
            splits = np.insert(splits, 0, 0)

            # Compute the result columns
            summed_cols = np.add.reduceat(locs[:, 1], splits)
            counts = np.diff(np.append(splits, locs.shape[0]))
            avg_cols = summed_cols / counts
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    

    if query_masks is not None:
        # remove invalid queries
        assert query_masks.size == num_queries, "invalid query mask shape"
        cols = cols[query_masks.reshape(-1).astype(np.bool)]
        assert cols.size == query_masks.sum(), "masking was not applied correctly"
        # update number of queries to account for those that were missing
        num_queries = query_masks.sum()

    return cols2metrics(cols, num_queries, query_masks_class_t2v)

def cols2metrics(cols, num_queries, query_masks_class=None):
    if query_masks_class is not None:
        new_query_number = int(num_queries) - np.count_nonzero(query_masks_class==0)
        new_cols = np.zeros(new_query_number)
        counter = 0
        query_masks_class = query_masks_class.reshape(cols.shape)
        for loc, query_mask in enumerate(query_masks_class):
            if query_mask == 1:
                new_cols[counter] = cols[loc]
                counter += 1
        cols = new_cols
        num_queries = new_query_number
    metrics = {}
    metrics["R1"] =  float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = float(np.sum(cols < 5)) / num_queries
    metrics["R10"] =  float(np.sum(cols < 10)) / num_queries
    metrics["MR"] = np.median(cols) + 1
    #metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
   
    return metrics


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

if __name__ == '__main__':
    original_sim = np.random.rand(1,5)
    adjust_sim =np.random.rand(1,5)#
    pdb.set_trace()
    rerank_sim = sim_rerank(original_sim, adjust_sim,3)
    
    pdb.set_trace()
    print(rerank_sim)
    