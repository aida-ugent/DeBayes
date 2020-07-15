from os.path import join, dirname, abspath
import numpy as np
import sklearn as sk

from debayes import DeBayes


def movielens_example():
    # Load the movielens 100k ratings.
    dataset_path = join(dirname(abspath(__file__)), 'movielens100k_data')
    with open(join(dataset_path, 'u.data'), mode='r', encoding='ISO-8859-1') as file_name:
        ratings = np.genfromtxt(file_name, delimiter='\t', autostrip=True, dtype=str)

    # Keep only the actual rating scores and timestamps.
    edges = ratings[:, [0, 1]]

    # Distinguish between users and items, since their numeric ID's overlap.
    edges = np.array([["user_" + str(edge[0]), "movie_" + str(edge[1])] for edge in edges])

    # Split the data into train and test edges.
    np.random.shuffle(edges)
    train_edges = edges[:int(0.8 * edges.shape[0])]
    test_edges_pos = edges[int(0.8 * edges.shape[0]):]

    # Generate some test edges that do not exist in the data. They are 'negative' edges.
    test_edges_neg = generate_negative_edges(edges)
    del edges

    # Parse the user file to get the user attributes.
    attributes = {}
    users = []
    age_brackets = [1, 18, 25, 35, 45, 50, 56, 1000]
    with open(join(dataset_path, 'u.user'), mode='r', encoding='ISO-8859-1') as file_name:
        users_data = np.genfromtxt(file_name, delimiter='|', autostrip=True, dtype=str)
    for user_data_row in users_data:
        user = "user_" + user_data_row[0]
        users.append(user)

        age = None
        for i in range(1, len(age_brackets)):
            if int(user_data_row[1]) < age_brackets[i]:
                age = age_brackets[i-1]
        attributes[user] = {
            'partition': 0,
            'gender': user_data_row[2],
            'age': age,
            'occupation': user_data_row[3]
        }

    # Also parse the movies file: just so that we get all movie ids. Just like the users, we assign a partition number
    # to the movies.
    with open(join(dataset_path, 'u.item'), mode='r', encoding='ISO-8859-1') as file_name:
        movies_data = np.genfromtxt(file_name, delimiter='|', autostrip=True, dtype=str)
    for movie_data_row in movies_data:
        movie = "movie_" + movie_data_row[0]
        attributes[movie] = {
            'partition': 1
        }

    # Create the DeBayes class.
    debayes = DeBayes(
        dimension=8,
        s1=0.7,
        s2=1,
        subsample=None,
        learning_rate=1e-1,
        nb_epochs=1000,
        training_prior_type='biased_degree',
        eval_prior_type='degree',
        sensitive_attributes=['gender', 'age', 'occupation'],
        dump_priors=False,
        load_pickles=False,
        dataset_name="ml-100k"
    )

    # Fit on the training data and all attributes (including partition numbers for every node)
    debayes.fit(train_data=train_edges, attributes=attributes, random_seed=None)

    ######
    # Section: measure AUC.
    test_labels_pos = np.ones(test_edges_pos.shape[0], np.bool)
    test_labels_neg = np.zeros(test_edges_neg.shape[0], np.bool)
    test_data = np.concatenate((test_edges_pos, test_edges_neg))
    test_labels = np.concatenate((test_labels_pos, test_labels_neg))

    debayes.predict(test_data)
    test_score = sk.metrics.roc_auc_score(test_labels, test_data)
    print("The TEST AUC score is {}.".format(test_score))

    ######
    # Section: measure bias in embeddings.

    # Gather the user embeddings.
    X = debayes.get_embeddings(ids=users)
    Y = np.array([attributes[user]['gender'] for user in users])

    # Split the embeddings into train and test examples.
    X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y)

    logi_reg = sk.model_selection.GridSearchCV(sk.pipeline.Pipeline([
        ('scaler', sk.preprocessing.StandardScaler()),
        ('logi', sk.linear_model.LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000))
    ]), param_grid={'logi__C': 100. ** np.arange(-2, 3), 'logi__penalty': ['l1', 'l2']}, cv=3,
        scoring='roc_auc')

    logi_reg.fit(X_train, X_test)
    clf = logi_reg.best_estimator_

    clf.fit(X_train, Y_train)
    test_score = sk.metrics.get_scorer('roc_auc')(clf, X_test, Y_test)
    print("The TEST AUC for predicting gender is {}.".format(test_score))


def generate_negative_edges(data):
    lhs_ids = np.unique(data[:, 0])
    rhs_ids = np.unique(data[:, 1])
    nb_lhs_ids = lhs_ids.shape[0]
    nb_rhs_ids = rhs_ids.shape[0]
    lhs_id_to_idx = dict(zip(lhs_ids, np.arange(nb_lhs_ids)))
    rhs_id_to_idx = dict(zip(rhs_ids, np.arange(nb_rhs_ids)))
    nb_negative_samples = data.shape[0]

    # Find linear indexes for the data in a simplified, 0-indexed matrix.
    simplified_data = np.array([(lhs_id_to_idx[edge[0]], rhs_id_to_idx[edge[1]]) for edge in data])
    data_lin_idx = np.ravel_multi_index((simplified_data[:, 0], simplified_data[:, 1]), dims=(nb_lhs_ids, nb_rhs_ids))

    negative_samples = []
    current_nb_negative_samples = 0
    while current_nb_negative_samples < nb_negative_samples:
        # Sample a bunch of edges.
        nb_left_to_sample = nb_negative_samples - current_nb_negative_samples
        lhs_samples = np.random.randint(low=0, high=nb_lhs_ids, size=nb_left_to_sample)
        rhs_samples = np.random.randint(low=0, high=nb_rhs_ids, size=nb_left_to_sample)

        # Check if they are negative by comparing their linear indices.
        candidate_lin_idx = np.ravel_multi_index((lhs_samples, rhs_samples), dims=(nb_lhs_ids, nb_rhs_ids))
        actual_negative_lin_idx = np.setdiff1d(candidate_lin_idx, data_lin_idx)

        # Keep the actually negative samples.
        actual_negative_samples = np.unravel_index(actual_negative_lin_idx, dims=(nb_lhs_ids, nb_rhs_ids))

        # Before storing them, convert the indices to ids.
        sampled_lhs_ids = lhs_ids[actual_negative_samples[0]]
        sampled_rhs_ids = rhs_ids[actual_negative_samples[1]]
        actual_negative_samples = np.vstack((sampled_lhs_ids, sampled_rhs_ids)).T
        negative_samples.append(actual_negative_samples)
        current_nb_negative_samples += actual_negative_samples.shape[0]

    return np.vstack(negative_samples)


if __name__ == '__main__':
    movielens_example()
