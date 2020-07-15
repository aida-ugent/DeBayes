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
                 dump_priors=False,
                 load_pickles=False,
                 dataset_name=""
                 ):
        super(DeBayes, self).__init__()

        self.dimension = dimension
        self.s1 = s1
        self.s2 = s2
        self.subsample = subsample
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.sensitive_attributes = sensitive_attributes
        self.training_prior_type = training_prior_type
        self.eval_prior_type = eval_prior_type
        self.dump_priors = dump_priors
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
        self.__partition_mask = np.empty(A.shape[0], dtype=np.int)
        for node in self.__G.nodes():
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
            pickle.dump(self.__cne.get_embeddings(), open(embeddings_path, "wb"))

        # Switch the prior for CNE to the evaluation prior.
        self.__cne.prior = self.__eval_bg_dist

    def __fit_prior(self, A, attributes, prior_type, pickle_path=None):
        bg_dist = BgDistBuilder.build(prior_type)
        prior_path = pickle_path + "_" + bg_dist.string_code() + ".prior"

        if self.load_pickles:
            try:
                bg_dist.load(prior_path)
                return bg_dist
            except FileNotFoundError:
                pass

        print("Computing a background distribution of type: " + prior_type + ".")
        if 'biased' in prior_type:
            attributes_formatted = self.__format_attribute_values(attributes)
            bg_dist.fit(A, block_mask=self.__partition_mask, attributes=attributes_formatted)
        else:
            bg_dist.fit(A, block_mask=self.__partition_mask)

        if self.dump_priors:
            bg_dist.save(prior_path)
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
