import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import shap
from Utils import replace_discrete, autoencoder, clustering_hierarchical, get_evaluation_and_boxplot, \
    clustering_kmeans, clustering_dbscan, diff_between_var, plot_dendrogram, get_sen_spe, \
    clustering_som, remark_cluster, compare_evaluation_table, plot_dendrogram
import scipy.stats as stats

'''
this script do the clustering
'''

if __name__ == "__main__":
    K = 3  # 3 clusters, determined by the dendrogram plot
    cluster_flag = 2   # 1 deep clustering, 2 pca+clustering, 3 no dimensional reduction,
    if cluster_flag == 1:
        save_name = 'deep'
    elif cluster_flag == 2:
        save_name = 'pca'
    else:
        save_name = 'raw'
    save_path = f'./results/{save_name}'
    # import the dataset
    dataset = pd.read_csv('./data/DemographicInfo.csv')
    id_list = dataset.iloc[:, [0]].values.flatten()
    csi_list = dataset.iloc[:, [-3]].values.flatten()
    groups = dataset.iloc[:, [-1]].values.flatten()
    # replace the discrete value
    dataset = replace_discrete(dataset)

    X = dataset.values
    feature_name = dataset.columns.tolist()
    del_list = [0, 1, 3, 4, 5, 14, 15]  # index of unused features

    X = np.delete(X, del_list, axis=1)
    feature_length = X.shape[1]
    feature_name = np.delete(feature_name, del_list, axis=0)
    X = np.array(X, dtype=np.float64)

    if cluster_flag == 1:
        nor_data = StandardScaler().fit_transform(X)
        encoding_dim = 3
        model = autoencoder(nor_data, encoding_dim)
        data = model.predict(nor_data)
        # explainer = shap.DeepExplainer(model, nor_data)
        # shap_values = explainer.shap_values(nor_data)
        # shap.initjs()
        # shap.summary_plot(shap_values, nor_data)
        # global_shap_values = []
        # for n in range(0, encoding_dim):
        #     global_shap_values.append(np.abs(shap_values[n]).mean(0))

    if cluster_flag == 2:
        data = stats.zscore(X)
        pca = PCA(n_components=0.8)
        data = pca.fit(data).transform(X)

        # Calculate the absolute values of the loadings (correlations between original features and principal components)
        loadings = np.abs(pca.components_)
        # Calculate the sum of the loadings for each feature
        feature_importance = np.sum(loadings, axis=0)
        # Normalize the feature importance values
        normalized_importance = feature_importance / np.sum(feature_importance)

        import matplotlib.pyplot as plt
        import seaborn as sns
        sorted_indices = np.argsort(normalized_importance)[::-1]
        fea = np.array(['Sex', 'VAS', 'PDI', 'WAS', 'Rand36-PF', 'PCS', 'IEQ', 'BSI', 'CSI'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x=fea[sorted_indices], y=normalized_importance[sorted_indices],
                    palette='Blues_r')
        plt.xlabel('Features', fontsize=16)
        plt.ylabel('Importance', fontsize=16)
        plt.title('Feature Importance', fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='-.', linewidth=0.5)
        plt.savefig(f'{save_path}/feature importance.png')
        plt.show()

    if cluster_flag == 3:
        # data = X
        data = StandardScaler().fit_transform(X)
        # X = np.array(X, dtype=np.float64)
        # data = stats.zscore(X)

    # plot dendrogram
    hc_labels, model = clustering_hierarchical(data, -1)
    labels = np.array(groups.copy())

    plt.figure(figsize=(15, 8))
    plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
    plot_dendrogram(model, labels=labels)
    plt.xlabel("Subject groups", fontsize=16)
    plt.ylabel("Distance", fontsize=16)
    group1_marker = plt.Line2D([], [], color='blue', marker='s', linestyle='', label='CLBP')
    group2_marker = plt.Line2D([], [], color='red', marker='s', linestyle='', label='HC')
    plt.legend(handles=[group1_marker, group2_marker], fontsize=16, loc='upper right')
    plt.savefig(f'{save_path}/dendrogram.pdf')
    plt.show()

    # hierarchical clustering
    print('hierarchical')
    hc_labels, model = clustering_hierarchical(data, K, 'ward')
    hc_labels = remark_cluster(hc_labels, csi_list)
    a, b = get_evaluation_and_boxplot(hc_labels, data, csi_list, groups, 'Hierarchical Clustering')
    print(b)
    plt.savefig(f'{save_path}/hierarchical.png')
    plt.show()

    # self-organizing maps clustering
    print('som')
    som_labels = clustering_som(data, K, max_iter=1500, plot_flag=False)
    som_labels = remark_cluster(som_labels, csi_list)
    a, b = get_evaluation_and_boxplot(som_labels, data, csi_list, groups, 'Self Organizing Map')
    print(b)
    plt.savefig(f'{save_path}/SOM.png')
    plt.show()
    # diff_between_var(som_labels, dataset.values, dataset.columns.tolist())

    # k-means clustering
    print('km')
    km_labels, model = clustering_kmeans(data, K)
    km_labels = remark_cluster(km_labels, csi_list)
    a, b = get_evaluation_and_boxplot(km_labels, data, csi_list, groups, 'K-means')
    print(b)
    plt.savefig(f'{save_path}/Kmeans.png')
    plt.show()

    # dbscan clustering
    print('dbscn')

    db_labels = clustering_dbscan(data, 15, 15)
    db_labels = remark_cluster(db_labels, csi_list)
    a, b = get_evaluation_and_boxplot(db_labels, data, csi_list, groups, 'DBSCAN')
    print(b)
    plt.savefig(f'{save_path}/dbscn.png')
    plt.show()

    df_labels = pd.DataFrame({'Hierarchical': hc_labels, 'KMeans': km_labels, 'DBSCAN': db_labels,
                              'SOM': som_labels})

    t = compare_evaluation_table(data, df_labels)
    print(t)

    best_labels = hc_labels
    best = best_labels.copy()
    print(f'cluster 0 = {np.mean(csi_list[np.where(best == 0)[0]])}')
    print(f'cluster 1 = {np.mean(csi_list[np.where(best == 1)[0]])}')
    print(f'cluster 2 = {np.mean(csi_list[np.where(best == 2)[0]])}')

    # merge o and 1 to a new cluster
    best[np.where(best_labels == 1)] = 0
    # make the cluster only has 0 and 1
    best[np.where(best_labels == 2)] = 1

    # cut off
    print('cut off for overall group')
    for n in range(20, 60):
        sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR = get_sen_spe(best, csi_list,n)
        print(n, sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR)
    # women
    print('cut off for women')
    w_index = np.where(dataset[' Seks '] == 0)[0]
    for n in range(20, 60):
        sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR = get_sen_spe(best[w_index], csi_list[w_index], n)
        print(n, sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR)

    # men
    print('cut off for men')
    m_index = np.where(dataset[' Seks '] == 1)[0]
    for n in range(20, 60):
        sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR = get_sen_spe(best[m_index], csi_list[m_index], n)
        print(n, sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR)

########### print the demography for results###########
    print('low cs vs high cs')
    X = dataset.values
    feature_name = dataset.columns.tolist()
    del_list = [0, 14, 15]
    X = np.delete(X, del_list, axis=1)
    feature_name = np.delete(feature_name, del_list, axis=0)
    X = np.array(X, dtype=np.float64)
    diff_between_var(best, X, feature_name)

    print('HC vs cluster B')
    best = best_labels.copy()
    X = dataset.values
    feature_name = dataset.columns.tolist()
    del_list = [0, 14, 15]
    X = np.delete(X, del_list, axis=1)
    feature_name = np.delete(feature_name, del_list, axis=0)
    X = np.array(X, dtype=np.float64)

    cluster_c = np.where(best_labels == 2)
    best = np.delete(best, cluster_c)
    X = np.delete(X, cluster_c, axis=0)
    diff_between_var(best, X, feature_name)

    print('HC vs cluster C')
    best = best_labels.copy()
    X = dataset.values
    feature_name = dataset.columns.tolist()
    del_list = [0, 14, 15]
    X = np.delete(X, del_list, axis=1)
    feature_name = np.delete(feature_name, del_list, axis=0)
    X = np.array(X, dtype=np.float64)

    cluster_b = np.where(best_labels == 1)
    best = np.delete(best, cluster_b)
    X = np.delete(X, cluster_b, axis=0)
    diff_between_var(best, X, feature_name)

    print('cluster B vs cluster C')
    best = best_labels.copy()
    X = dataset.values
    feature_name = dataset.columns.tolist()
    del_list = [0, 14, 15]
    X = np.delete(X, del_list, axis=1)
    feature_name = np.delete(feature_name, del_list, axis=0)
    X = np.array(X, dtype=np.float64)

    cluster_a = np.where(best_labels == 0)
    best = np.delete(best, cluster_a)
    X = np.delete(X, cluster_a, axis=0)
    diff_between_var(best, X, feature_name)


    # compare cluster C with healthy controls
    # t = compare_evaluation_table(data, df_labels)
    # print(t)
    #
    # best_labels = hc_labels
    # best = best_labels.copy()
    #
    # # keep all HC in cluster 0
    # HC_index = np.where(groups == 0)[0]
    # best[HC_index] = 0
    # # remove low cs cluster
    # low_cs_index = np.where(best == 1)[0]
    #
    # new_best = np.delete(best, low_cs_index, axis=0)
    # new_csi = np.delete(csi_list, low_cs_index, axis=0)
    # new_best[np.where(new_best == 2)] = 1
    #
    # # cut off
    # print('cut off for overall group')
    # for n in range(20, 60):
    #     sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR = get_sen_spe(new_best, new_csi,n)
    #     print(n, sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR)
    #
    # # women
    # print('cut off for women')
    # sex_list = dataset[' Seks '].values.flatten()
    # new_sex = np.delete(sex_list, low_cs_index, axis=0)
    # w_index = np.where(new_sex == 0)[0]
    # for n in range(20, 60):
    #     sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR = get_sen_spe(new_best[w_index], new_csi[w_index], n)
    #     print(n, sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR)
    #
    # # men
    # print('cut off for men')
    # m_index = np.where(new_sex == 1)[0]
    # for n in range(20, 60):
    #     sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR = get_sen_spe(new_best[m_index], new_csi[m_index], n)
    #     print(n, sensitivity, specificity, auc, youden_index, PPV, NPV, PLR, NLR)
    #
