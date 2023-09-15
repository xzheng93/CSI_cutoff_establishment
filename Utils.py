import os
from scipy.stats import stats
from scipy.stats import mannwhitneyu
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn import metrics
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.neighbors import kneighbors_graph
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import numpy as np
from minisom import MiniSom
from numpy.random import seed
seed(1)
from tensorflow import random
from sklearn.metrics import roc_auc_score
random.set_seed(1)
from sklearn.metrics import confusion_matrix
'''
Provide all the clustering methods and tool functions
'''

def replace_discrete(data):
    """
    replace the discrete variable to number
    :param data: dataframe
    :return: new dataframe
    """
    data = data.replace('Woman', 0)
    data = data.replace('Man', 1)
    data = data.replace('woman', 0)
    data = data.replace('man', 1)
    data = data.replace('Female', 0)
    data = data.replace('Male', 1)
    data = data.replace('Primary education', 0)
    data = data.replace('Secondary education', 0.5)
    data = data.replace('Higher education', 1)
    data = data.replace('Sedentary', 0)
    data = data.replace('Light', 0.5)
    data = data.replace('Medium', 1)
    data = data.replace('Heavy', 1.5)
    data = data.fillna(0)
    return data



def plot_dendrogram(model, **kwargs):
    """
    plot dendrogram for hierarchical clustering
    :param model: AgglomerativeClustering
    :param kwargs:
    :return: plot
    """
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def print_evaluation_table(data, label):
    """
    evaluation of the clustering, Silhouette Coefficient <1, the bigger the better
    Calinski-Harabasz, the bigger the better, Davies-Bouldin >0, the smaller the better
    :param data: input data of clustering method
    :param label: label predicted by cluster method
    :return: table of evaluation indicators
    """
    t = PrettyTable(['Indicator', 'Value'])
    evaluate_method = (('Silhouette', metrics.silhouette_score), ('Calinski-Harabasz', metrics.calinski_harabasz_score),
              ('Davies-Bouldin', metrics.davies_bouldin_score))
    for name, method in evaluate_method:

        try:
            score = method(data, label)
        except:
                score = 'NULL'
        t.add_row([name, score])
    return t


def compare_evaluation_table(data, df_labels):
    """
    evaluation of the clustering, Silhouette Coefficient <1, the bigger the better
    Calinski-Harabasz, the bigger the better, Davies-Bouldin >0, the lower the better
    :param data: input data of clustering method
    :param label_df: dataframe of label predicted by cluster methods
    :return: table of evaluation indicators
    """
    t = PrettyTable(['Indicator']+df_labels.columns.to_list())
    evaluate_method = (('Silhouette', metrics.silhouette_score), ('Calinski-Harabasz', metrics.calinski_harabasz_score),
              ('Davies-Bouldin', metrics.davies_bouldin_score))
    for name, method in evaluate_method:
        score = []
        for i in range(0, df_labels.shape[1]):
            try:
                score.append(method(data, list(df_labels.iloc[:, i])))
            except:
                score.append('NULL')
        t.add_row([name]+score)
    return t

def plot_box_dot(lists, groups, title):
    """
    boxplot and dots plot for the clustering results
    :param lists: a list of k lists of feature value
    :return: plot
    """
    # sort clusters by mean value of CSI scores
    # lists.sort(key=lambda x: np.mean(x))
    plt.figure(figsize=(8, 4))
    plt.boxplot(lists, vert=False, showmeans=True)
    plt.yticks([1, 2, 3], ['A', 'B', 'C'], fontsize=14)
    median_line = plt.Line2D([], [], color='darkorange', label='Median')
    mean_marker = plt.Line2D([], [], color='green', marker='^', linestyle='', label='Mean')
    group1_marker = plt.Line2D([], [], color='blue', marker='o', linestyle='', label='CLBP')
    group2_marker = plt.Line2D([], [], color='red', marker='o', linestyle='', label='HC')

    # groups = np.array(groups, dtype=object)
    # lists = np.array(lists, dtype=object)
    m_color = ['r', 'b', 'b']

    u_label = [0, 11, 222]
    for i in range(0, len(groups)):
        for j in range(0, 3):
            x = lists[i][np.where(groups[i] == u_label[j])]
            y = np.random.normal(1 + i, 0.04, size=len(x))
            plt.plot(x, y, '.', color=m_color[j], alpha=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel('CSI scores', fontsize=14)
    plt.ylabel('Clusters', fontsize=14)
    plt.legend(handles=[median_line, mean_marker, group1_marker, group2_marker], loc='upper left')
    plt.grid(axis='x', linestyle='-.', linewidth=0.5)
    plt.tight_layout()

def get_evaluation_and_boxplot(labels, data, csi_list, groups, title):
    """
    :param title:
    :param labels: clustering results
    :param data: dataframe
    :param csi_list: csi data
    :return:
    """

    # evaluate the performance of the clustering method
    e_table = print_evaluation_table(data, labels)

    # prepare data for box plot
    box_list = []
    h_label = []  # healthy subjects
    csi_table = PrettyTable(['group', 'mean', 'median', 'proportion'])
    for i in np.unique(labels):
        box_list.append(csi_list[np.where(labels == i)])
        h_label.append(groups[np.where(labels == i)])
        p = len(np.where(h_label[i] == 0)[0])
        csi_table.add_row(
            [i, np.mean(box_list[i]), np.median(box_list[i]), str(p)+'/'+str(len(box_list[i]))])

    # plot box plot
    plot_box_dot(box_list, h_label, title)
    return e_table, csi_table


def clustering_hierarchical(df, n_cluster=-1, type='ward'):
    connectivity = kneighbors_graph(
        df, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    if n_cluster > 0 and type == 'ward':
        model = AgglomerativeClustering(n_clusters=n_cluster, linkage='ward')
    elif n_cluster > 0 and type == 'average':
        model = AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=n_cluster)
    else:
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(df)
    labels = model.labels_
    print("hierarchical clustering")
    return labels, model


def clustering_kmeans(df, n_cluster):
    n_clusters = n_cluster
    kmeans = KMeans(n_clusters, init='random', n_init=200, max_iter=1000, algorithm='auto', verbose=0,
                    random_state=10)
    kmeans.fit(df)
    labels = kmeans.labels_
    return labels, kmeans


def clustering_dbscan(df, m_eps=0.7, m_s=10):
    dbscan = DBSCAN(eps=m_eps, min_samples=m_s)
    dbscan.fit(df)
    labels = dbscan.labels_
    return labels

def clustering_OPTICS(df, m_s=1):
    optics = OPTICS(min_samples=m_s)
    optics.fit(df)
    labels = optics.labels_
    return labels


def clustering_som(data, K, max_iter=1500, plot_flag=False):
    som_shape = (K, 1)
    som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=1, learning_rate=.08,
                  neighborhood_function='gaussian', random_seed=10)
    som.train_batch(data, max_iter, verbose=True)
    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    som_labels = np.ravel_multi_index(winner_coordinates, data.shape)
    if plot_flag:
        plot_som_qerror(max_iter, som, data)
        plt.show()
        plot_som_result(data, som_labels, som)
        plt.show()
    return som_labels


def autoencoder(df, encoding_dim=3, draw=1, verbose=2, lm_flag=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.path.exists('./model/%dencoder_moder.h5' % encoding_dim) and lm_flag:
        encoder = load_model('./model/%dencoder_moder.h5' % encoding_dim)
    else:
        n_col = df.shape[1]
        n_row = df.shape[0]
        X_train, X_test = df[:int(0.7*n_row)], df[int(0.3*n_row):]

        input_dim = Input(shape=(n_col, ))

        encoded = Dense(502, activation='tanh')(input_dim)
        code = Dense(encoding_dim, activation='tanh')(encoded)
        decoded = Dense(502, activation='tanh')(code)
        output_dim = Dense(n_col, activation='tanh')(decoded)

        autoencoder = Model(inputs=input_dim, outputs=output_dim)
        encoder = Model(inputs=input_dim, outputs=code)

        autoencoder.compile(optimizer='adamax',
                            loss='mean_squared_error',
                            metrics=['mae'])
        checkpointer = ModelCheckpoint(filepath='./model/%dmodel_check_point.h5' % encoding_dim,
                                       verbose=0,
                                       save_best_only=True)
        history = autoencoder.fit(X_train, X_train,
                                  epochs=150,
                                  batch_size=1,
                                  shuffle=True,
                                  validation_data=(X_test, X_test),
                                  verbose=verbose,
                                  callbacks=[checkpointer]).history
        encoder.save('./model/%dencoder_moder.h5' % encoding_dim)

        if draw == 1:
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.plot(history['loss'], c='dodgerblue', lw=3)
            plt.plot(history['val_loss'], c='coral', lw=3)
            plt.title('model loss')
            plt.ylabel('mse')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right')
            plt.subplot(122)
            plt.plot(history['mae'], c='dodgerblue', lw=3)
            plt.plot(history['val_mae'], c='coral', lw=3)
            plt.title('model mae')
            plt.ylabel('mae')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper right');
            plt.show()

    return encoder


def diff_between_var(labels, df, feature_name):
    # compute the statistical significant difference between 3 group
    t = PrettyTable(['Feature', 'group 0', 'group 1', 'P-Value', 'signal'])
    clusters = np.unique(labels)
    for i in range(0, df.shape[1]):
        data1 = df[np.where(labels == clusters[0]), i].flatten()
        data2 = df[np.where(labels == clusters[1]), i].flatten()
        t_score, p = mannwhitneyu(data1, data2)
        if p < 0.05:
            signal = '*'
        else:
            signal = '--'
        if i == 1:
            t.add_row([feature_name[i], f'{len(np.where(data1 == 0)[0])}/{len(np.where(data1 == 1)[0])}', f'{len(np.where(data2 == 0)[0])}/{len(np.where(data2 == 1)[0])}', f'{p:.3f}', signal])
        else:
            t.add_row([feature_name[i], f'{np.mean(data1):.1f}\u00B1{np.std(data1):.1f}', f'{np.mean(data2):.1f}\u00B1{np.std(data2):.1f}', f'{p:.3f}', signal])
    print(t)


def get_sen_spe(t_label, csi_score_list, cf_score):
    """
    compute the sensitivity and specificity
    :param t_label:
    :param csi_score_list:
    :param cf_score:
    :return:
    """

    p_label = csi_score_list.copy()
    p_label[np.where(p_label < cf_score)] = 0
    p_label[np.where(p_label >= cf_score)] = 1

    tn, fp, fn, tp = confusion_matrix(t_label, p_label).ravel()

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    auc = roc_auc_score(t_label, p_label)
    f1 = tp/(tp+0.5*(fp+fn))
    youden_index = tp / (tp + fn) - fp / (fp + tn)
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    PLR = sensitivity/(1-specificity)
    NLR = (1-sensitivity)/specificity

    return round(sensitivity, 2), round(specificity, 2), round(auc, 2), round(youden_index, 2), round(PPV,2), round(NPV,2), round(PLR,2), round(NLR,2)


def naive_auc(labels, preds):
    """
    compute the auc scores
    :param labels:
    :param preds:
    :return:
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg

    labels_preds = zip(labels, preds)
    labels_preds = sorted(labels_preds, key=lambda x: x[1])
    accumulated_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += accumulated_neg
        else:
            accumulated_neg += 1

    return satisfied_pair / float(total_pair)


def plot_som_result(data, labels, som_model):
    """
    plot the clustering result (first 2 dimension of data point) of som
    :param data:
    :param labels:
    :param som_model:
    :return:
    """
    # plotting the clusters using the first 2 dimentions of the data
    for c in np.unique(labels):
        plt.scatter(data[labels == c, 0],
                    data[labels == c, 1], label='cluster=' + str(c), alpha=.7)

    # plotting centroids
    for centroid in som_model.get_weights():
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                    s=80, color='k', label='centroid')
    plt.legend()


def plot_som_qerror(max_iter, som_model, data):
    """
    plot the quantitative error of som training
    :param max_iter:
    :param som_model:
    :param data:
    :return:
    """
    q_error = []
    for i in range(max_iter):
        rand_i = np.random.randint(len(data))
        som_model.update(data[rand_i], som_model.winner(data[rand_i]), i, max_iter)
        q_error.append(som_model.quantization_error(data))
    plt.plot(np.arange(max_iter), q_error, label='quantization error')
    plt.ylabel('quantization error')
    plt.xlabel('iteration index')
    plt.legend()


def remark_cluster(labels, csi_list):
    """
    remark the labels list, make lower clusters represent lower CSI scores (e.g. cluster 0 has lower CSI
    than cluster 1)
    :param labels:
    :param csi_list:
    :return:
    """
    clustering_labels = labels.copy()
    clusters = np.unique(clustering_labels)
    rank = []
    for i in clusters:
        rank.append(np.median(csi_list[np.where(clustering_labels == i)]))
    rank_index = np.argsort(rank)
    cluster_order = clusters[rank_index]
    for i in range(len(cluster_order)):
        clustering_labels[np.where(clustering_labels == cluster_order[i])] = (i+1)*100
    for i in range(len(cluster_order)):
        clustering_labels[np.where(clustering_labels == (i+1)*100)] = i
    return clustering_labels
