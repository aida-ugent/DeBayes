from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict

import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm


class ConditionalNetworkEmbedding:
    def __init__(self,
                 prior,
                 d=8,
                 s1=1,
                 s2=2,
                 nb_epochs=1,
                 learning_rate=1e-3,
                 k_subsample=None,
                 sampling_correction=True):
        self.prior = prior
        self.d = d
        self.s1 = s1
        self.s2 = s2
        self.nb_epochs = nb_epochs
        self.learning_rate = learning_rate
        self.k_subsample = k_subsample
        self.sampling_correction = sampling_correction

        self.__embedding = None

    def fit(self, A):
        self.__presample_ids(A)

        n = A.shape[0]

        print("Fitting CNE embeddings.")
        init_X = np.random.randn(n, self.d)
        embedding = self.__optimizer_adam(init_X, A, num_epochs=self.nb_epochs, alpha=self.learning_rate)

        self.__embedding = embedding

    def get_embeddings(self):
        return self.__embedding

    def set_embeddings(self, embedding):
        self.__embedding = embedding

    def __presample_ids(self, A):
        print("Starting neighbourhood sampling.")

        self.__sample_ids = []
        self.__sample_masks = []
        self.__sample_corrections = []
        n = A.shape[0]

        for i in range(n):
            # Get possible positive neighbours.
            pos_ids = sparse.find(A[i, :])[1]

            # If the node is completely unconnected, then our prior will be zero anyway. Therefore, it is not useful to
            # evaluate anything for this node at all.
            if pos_ids.shape[0] == 0:
                self.__sample_ids.append(None)
                self.__sample_masks.append(None)
                self.__sample_corrections.append(None)
                continue

            # Get possible negative neighbours.
            neighbour_row = A[i, :].A.squeeze()
            preliminary_neg_mask = neighbour_row == 0
            preliminary_neg_mask[i] = False

            # Look for edges where the prior is approximately zero and make sure that we don't sample them.
            preliminary_neg_ids = np.arange(n)[preliminary_neg_mask]
            priors = self.prior.get_row_probability(i, preliminary_neg_ids)
            zero_prior_indices = preliminary_neg_ids[np.where(priors < 1e-8)[0]]
            preliminary_neg_mask[zero_prior_indices] = False

            # Get the final possible negative neighbours.
            neg_ids = np.arange(n)[preliminary_neg_mask]

            if self.k_subsample and pos_ids.shape[0] > self.k_subsample:
                pos_samples = np.random.choice(pos_ids, self.k_subsample, replace=False)
            else:
                pos_samples = pos_ids
            pos_sample_correction = pos_ids.shape[0] / pos_samples.shape[0]

            if self.k_subsample and neg_ids.shape[0] > 2 * self.k_subsample - pos_samples.shape[0]:
                neg_samples = np.random.choice(neg_ids, 2 * self.k_subsample - pos_samples.shape[0], replace=False)
            else:
                neg_samples = neg_ids

                # Check if there are any negative nodes at all. Note that this could not be the case if either the other
                # priors are all too low, or if the node is connected to everything.
                if neg_samples.shape[0] == 0:
                    raise ValueError("Node {} has no negative neighbours!".format(i))
            neg_sample_correction = neg_ids.shape[0] / neg_samples.shape[0]

            self.__sample_ids.append(np.hstack((
                pos_samples,
                neg_samples)))
            self.__sample_masks.append(np.hstack((
                np.ones_like(pos_samples, dtype=np.bool),
                np.zeros_like(neg_samples, dtype=np.bool))))
            self.__sample_corrections.append(np.hstack((
                np.ones_like(pos_samples, dtype=np.float)*pos_sample_correction,
                np.ones_like(neg_samples, dtype=np.float)*neg_sample_correction)))

    def __optimizer_adam(self, X, A, num_epochs=100, alpha=0.001, beta_1=0.9, beta_2=0.9999, eps=1e-8):
        m_prev = np.zeros_like(X)
        v_prev = np.zeros_like(X)

        s_diff = (1/self.s1**2 - 1/self.s2**2)
        s_div = (self.s1 / self.s2)**self.d

        for epoch in tqdm(range(num_epochs), position=0, leave=True):
            obj, grad = self.__eval_obj_grad(X, A, s_diff, s_div)
            obj = -obj
            grad = -grad

            # Adam optimizer
            m = beta_1*m_prev + (1-beta_1)*grad
            v = beta_2*v_prev + (1-beta_2)*grad**2

            m_prev = m.copy()
            v_prev = v.copy()

            m = m/(1-beta_1**(epoch+1))
            v = v/(1-beta_2**(epoch+1))
            X -= alpha*m/(v**.5 + eps)

            grad_norm = np.sum(grad**2)**.5
            if num_epochs >= 50 and epoch % int(num_epochs / 50) == 0:
                tqdm.write('epoch: {:d}/{:d}, gradient norm: {:.4f}, obj: {:.4f}'
                           .format(epoch, num_epochs, grad_norm, obj))
        return X

    def __eval_obj_grad(self, X, A, s_diff, s_div):
        obj = 0.
        n = A.shape[0]
        grad = np.zeros_like(X)
        for i in range(n):
            sample_ids = self.__sample_ids[i]
            if sample_ids is None:
                continue
            sample_masks = self.__sample_masks[i]
            sample_corrections = self.__sample_corrections[i]

            P_aij_X = self.__compute_posterior(i, sample_ids, s_diff, s_div, X=X)
            grad_i = 2*s_diff*((P_aij_X - sample_masks)*(X[i, :] - X[sample_ids, :]).T).T
            if self.sampling_correction:
                grad_i *= sample_corrections[:, np.newaxis]
            grad[i, :] += np.sum(grad_i, axis=0)
            grad[sample_ids, :] -= grad_i

            obj += np.sum(np.log(P_aij_X[sample_masks])) + np.sum(np.log(1-P_aij_X[~sample_masks]))

        return obj, grad

    def __compute_posterior(self, row_id, col_ids, s_diff, s_div, X=None):
        if X is None:
            X = self.__embedding
        prior = self.prior.get_row_probability(row_id, col_ids)
        dist = self.__compute_squared_dist(X, row_id, col_ids)

        # See CNE paper supplement 2: Deriving the log probability of posterior P(G|X).
        likelihood_div = s_div * np.exp(s_diff * dist / 2)
        prior_div = (1-prior) / prior

        posterior = 1. / (1 + likelihood_div * prior_div)
        return posterior

    @staticmethod
    def __compute_squared_dist(X, target_ids, sample_ids):
        # return np.sum((X[target_id, :] - X[sample_ids, :]) ** 2, axis=1).T
        dist = X[target_ids] - X[sample_ids]
        return np.einsum('ij,ij->i', dist, dist)

    def predict(self, edges):
        edge_dict = defaultdict(list)
        ids_dict = defaultdict(list)
        for i, edge in enumerate(edges):
            edge_dict[edge[0]].append(edge[1])
            ids_dict[edge[0]].append(i)

        predictions = []
        ids = []
        for u in edge_dict.keys():
            neighbours = np.array(edge_dict[u])
            u_predictions = np.empty(neighbours.shape[0], dtype=np.float)

            tiny_priors = np.where(self.prior.get_row_probability(u, edge_dict[u]) < 1e-10)[0]
            u_predictions[tiny_priors] = 0.0

            not_tiny = np.ones_like(u_predictions, dtype=np.bool)
            not_tiny[tiny_priors] = False

            s_diff = (1 / self.s1 ** 2 - 1 / self.s2 ** 2)
            s_div = (self.s1 / self.s2)**self.d
            u_predictions[not_tiny] = self.__compute_posterior(u, neighbours[not_tiny], s_diff, s_div)

            predictions.extend(u_predictions)
            ids.extend(ids_dict[u])

        return [p for _, p in sorted(zip(ids, predictions))]
