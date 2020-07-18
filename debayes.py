# Author: Maarten Buyl
# Contact: maarten.buyl@ugent.be
# Date: 17/07/2020


import networkx as nx
import numpy as np
import pickle
from natsort import natsorted
from os.path import join, abspath, dirname

from CNE_modified import ConditionalNetworkEmbedding
from bg_dist import BgDistBuilder


class DeBayes:
    def __init__(self,
                 dimension=8,
                 s1=None,
                 s2=None,
                 subsample=100,
                 learning_rate=1e-1,
                 nb_epochs=1000,
                 training_prior_type='biased_degree',
                 eval_prior_type='degree',
                 sensitive_attributes=None,
                 dump_pickles=False,
                 load_pickles=False,
                 dataset_name=""
                 ):
        """
        The class for the DeBayes algorithm. DeBayes works in two phases. First: the embeddings are trained using the
        Conditional Network Embedding (CNE) method, with a biased prior distribution. Second: the prior distribution
        is swapped for an oblivious prior for evaluation. Both priors can be trained separately, and this is done before
        the two phases.
        :param dimension: the dimensionality of the embeddings.
        :param s1: a factor that controls how close embeddings of connected nodes should be. Since it merely fixes the
        scale of the embedding space, this factor can be left at value 1 permanently.
        :param s2: a factor that controls how close embeddings of disconnected nodes should be. This should be tuned
        per dataset. Good values range from 2 to 64, so trying various powers of 2 is a good idea. S2 should be higher
        than s1.
        :param subsample: if not None then the neighbourhood of every node is subsampled in CNE, according to this
        value. For example, if subsample = k then for every node, only k negative neighbours are sampled.
        :param learning_rate: the learning rate of the CNE algorithm.
        :param nb_epochs: the number of epochs that the CNE algorithm may run.
        :param training_prior_type: a string that designates the type of prior to use during training. The options are:
        'density', 'degree' or 'biased_degree'. For debiasing embeddings, 'biased_degree' should be used.
        :param eval_prior_type: a string that designates the type of prior to use during training. The options are:
        'density', 'degree' or 'biased_degree'. For debiasing predictions, 'biased_degree' should NOT be used.
        :param sensitive_attributes:
        :param dump_pickles: if True, write embeddings and the prior distribution to a pickle file. They will be written
        to the "cne_dump" folder.
        :param load_pickles: if True, load embeddings and the prior distribution from a pickle file.
        :param dataset_name: some identifier string for the dataset. This identifier is used to specify the pickle
        files.
        """

        super().__init__()

        self.dimension = dimension
        self.s1 = s1
        self.s2 = s2
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.sensitive_attributes = sensitive_attributes
        self.training_prior_type = training_prior_type
        self.eval_prior_type = eval_prior_type
        self.dump_pickles = dump_pickles
        self.load_pickles = load_pickles
        self.dataset_name = dataset_name

        self.__G = None
        self.__partition_mask = None
        self.__label_to_index = None
        self.__index_to_label = None

        self.__training_bg_dist = None
        self.__eval_bg_dist = None
        self.__cne = None

    def fit(self, train_data, attributes, random_seed=None):
        """
        Fit the DeBayes algorithm using the current parameters.
        :param train_data: the data that may be used for fitting.
        :param attributes: a dict that maps nodes to their 'attributes', which are also organized as a dict. For example
        for a user in the Movielens dataset, you would have
        'attributes[user_id] = {gender: "M", age: 18, occupation: artist, partition: 0}'. The 'partition' attribute
        is especially important, since this can be used to specify types of nodes in the graph. In Movielens there are
        users and movies, so there you would also have 'attributes[movie_id] = {partition: 1}'. By specifying these
        partitions, the prior can be made much more efficient and informative.
        :param random_seed: an integer that specifies the random seed that was used to generate the training data.
        :return:
        """

        if (self.dump_pickles or self.load_pickles) and random_seed is None:
            raise ValueError("To load or dump pickles, a random seed should be given!")

        # Build graph from training edges.
        self.__G = nx.Graph()
        for row in train_data:
            self.__G.add_edge(row[0], row[1])

        # Get all known entities and add their ids as disconnected nodes to the graph.
        self.__G.add_nodes_from(attributes.keys())

        # Give each label an index and keep track of the old_label -> new_index mapping.
        new_idx = np.arange(len(self.__G.nodes()))
        self.__label_to_index = dict(list(zip(natsorted(self.__G.nodes()), new_idx)))
        self.__index_to_label = dict(list(zip(new_idx, natsorted(self.__G.nodes()))))
        prepped_graph = nx.relabel_nodes(self.__G, mapping=self.__label_to_index)

        # Using the integer-indexed graph, build an adjacency matrix with the same indices.
        A = nx.adjacency_matrix(prepped_graph, nodelist=sorted(prepped_graph.nodes()), weight=None)

        # Extend adjacency matrix with partition mask, which indicates which partition a node belongs to.
        self.__partition_mask = np.zeros(A.shape[0], dtype=np.int)
        for node in self.__G.nodes():
            if node not in attributes:
                raise ValueError("Node " + node + " has no attributes! It should at least be given an empty dict!")
            if 'partition' in attributes[node]:
                self.__partition_mask[self.__label_to_index[node]] = attributes[node]['partition']

        # For everything that we dump to a pickle, generate the base path.
        pickle_path = join(dirname(abspath(__file__)), "cne_dump", "{}_seed_{}".format(
            self.dataset_name, random_seed))

        # Fit the training and evaluation prior.
        self.__training_bg_dist = self.__fit_prior(A, attributes, self.training_prior_type, pickle_path)

        # training_P = self.__training_bg_dist.get_full_P_matrix()
        self.__eval_bg_dist = self.__fit_prior(A, attributes, self.eval_prior_type, pickle_path)

        # CNE arguments:
        self.__cne = ConditionalNetworkEmbedding(prior=self.__training_bg_dist, d=self.dimension, s1=self.s1,
                                                 s2=self.s2, nb_epochs=self.nb_epochs, learning_rate=self.learning_rate,
                                                 k_subsample=self.subsample, sampling_correction=False)

        # Attempt to load matching embeddings from previous runs.
        embeddings_path = pickle_path + "_dim_" + str(self.dimension) + \
                          "_trp_" + self.__training_bg_dist.string_code() + ".emb"
        embeddings = None
        if self.load_pickles:
            try:
                embeddings = pickle.load(open(embeddings_path, "rb"))
            except FileNotFoundError:
                embeddings = None

        # If embeddings could be loaded, use those. Else, fit CNE!
        if embeddings is not None:
            self.__cne.set_embeddings(embeddings)
        else:
            self.__cne.fit(A)
            if self.dump_pickles:
                try:
                    pickle.dump(self.__cne.get_embeddings(), open(embeddings_path, "wb"))
                except FileNotFoundError:
                    print("Tried to dump embeddings to " + embeddings_path + " but the folder does not exist. "
                                                                             "Therefore, not dumping!")

        # Switch the prior for CNE to the evaluation prior.
        self.__cne.prior = self.__eval_bg_dist

    def __fit_prior(self, A, attributes, prior_type, pickle_path=None):
        bg_dist = BgDistBuilder.build(prior_type)
        prior_path = pickle_path + "_" + bg_dist.string_code() + ".prior"

        if self.load_pickles:
            try:
                bg_dist = pickle.load(open(prior_path, "rb"))
                return bg_dist
            except FileNotFoundError:
                pass

        print("Computing a background distribution of type: " + prior_type + ".")
        if 'biased' in prior_type:
            attributes_formatted = self.__format_attribute_values(attributes)
            bg_dist.fit(A, block_mask=self.__partition_mask, attributes=attributes_formatted)
        else:
            bg_dist.fit(A, block_mask=self.__partition_mask)

        if self.dump_pickles:
            pickle.dump(bg_dist, open(prior_path, "wb"))
        return bg_dist

    def get_embeddings(self, ids):
        all_embeddings = self.__cne.get_embeddings()
        requested_embeddings = []
        for node_id in ids:
            requested_embeddings.append(all_embeddings[self.__label_to_index[node_id]])
        return np.array(requested_embeddings)

    def predict(self, edges):
        edges = self.__edges_to_adjacency_index(edges)
        scores = self.__cne.predict(edges)
        return scores

    def __edges_to_adjacency_index(self, edge_list):
        edge_array = np.empty((len(edge_list), 2), dtype=np.int)
        for i, edge in enumerate(edge_list):
            edge_start = self.__label_to_index[edge[0]]
            edge_end = self.__label_to_index[edge[1]]
            edge_array[i, :] = [edge_start, edge_end]
        return edge_array

    def __format_attribute_values(self, attributes):
        # Generate a dict of attribute value arrays. "N/A" signifies that this attribute was not given for this
        # node. Each element in the dict corresponds with a type of sensitive attribute.
        attribute_values = {}
        for sensitive_attribute in self.sensitive_attributes:
            attribute_values[sensitive_attribute] = ["N/A" for _ in range(len(self.__G))]

        # Iterate over all nodes and fill in the relevant attribute value.
        for node in self.__G.nodes():
            node_data = attributes[node]
            for sensitive_attribute in self.sensitive_attributes:
                try:
                    attribute_val = node_data[sensitive_attribute]
                except KeyError:
                    continue
                attribute_values[sensitive_attribute][self.__label_to_index[node]] = attribute_val

        # Convert the attribute value lists to arrays.
        for sensitive_attribute, attribute_array in attribute_values.items():
            attribute_values[sensitive_attribute] = np.array(attribute_array)
        return attribute_values

    def filename(self):
        return self.__class__.__name__ + "_trp_" + self.__training_bg_dist.string_code() + "_evp_" + \
               self.__eval_bg_dist.string_code()
