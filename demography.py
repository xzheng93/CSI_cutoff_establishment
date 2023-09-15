import pandas as pd
import numpy as np
from Utils import replace_discrete
import scipy.stats as stats

# import the dataset
dataset = pd.read_csv('./data/DemographicInfo.csv')
id_list = dataset.iloc[:, [0]].values.flatten()
csi_list = dataset.iloc[:, [-3]].values.flatten()
groups = dataset.iloc[:, [-1]].values.flatten()
# replace the discrete value
dataset = replace_discrete(dataset)
feature_name = dataset.columns.tolist()
X = dataset.values

hc_index = np.where(X[:,-1] == 0)[0]
clbp_index = np.where(X[:,-1] != 0)[0]
hc_data = X[hc_index]
clbp_data = X[clbp_index]

w_index = np.where(X[:,2] == 0)[0]
m_index = np.where(X[:,2] == 1)[0]
w_data = X[w_index]
m_data = X[m_index]


index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
feature_name = np.array(feature_name)
feature_name = feature_name[index]

hc_data = hc_data[:, index]
clbp_data = clbp_data[:, index]
w_data = w_data[:, index]
m_data = m_data[:, index]
# HC
for i in range(0, len(feature_name)):
    mean = round(np.mean(hc_data[:,i]), 1)
    std = round(np.std(hc_data[:,i]), 1)
    if i == 1:
        print(feature_name[i], f'{len(np.where(hc_data[:,i] == 0)[0])}/{len(np.where(hc_data[:,i] == 1)[0])}')
    else:
        print(f'{feature_name[i], mean}' u" \u00B1" f' {std}')
# CLBP
for i in range(0, len(feature_name)):
    mean = round(np.mean(clbp_data[:,i]), 1)
    std = round(np.std(clbp_data[:,i]), 1)
    if i == 1:
        print(feature_name[i], f'{len(np.where(clbp_data[:,i] == 0)[0])}/{len(np.where(clbp_data[:,i] == 1)[0])}')
    else:
        print(f'{feature_name[i], mean}' u" \u00B1" f' {std}')
# P test
for i in range(0, len(feature_name)):
    t_statistic, p_value = stats.mannwhitneyu(list(clbp_data[:,i]), list(hc_data[:,i]))
    print(round(p_value, 3))

# female
for i in range(0, len(feature_name)):
    mean = round(np.mean(w_data[:,i]), 1)
    std = round(np.std(w_data[:,i]), 1)
    if i == 1:
        print(feature_name[i], f'{len(np.where(w_data[:,i] == 0)[0])}/{len(np.where(w_data[:,i] == 1)[0])}')
    else:
        print(f'{feature_name[i], mean}' u" \u00B1" f' {std}')
# male
for i in range(0, len(feature_name)):
    mean = round(np.mean(m_data[:,i]), 1)
    std = round(np.std(m_data[:,i]), 1)
    if i == 1:
        print(feature_name[i], f'{len(np.where(m_data[:,i] == 0)[0])}/{len(np.where(m_data[:,i] == 1)[0])}')
    else:
        print(f'{feature_name[i], mean}' u" \u00B1" f' {std}')

# P test
for i in range(0, len(feature_name)):
    t_statistic, p_value = stats.mannwhitneyu(list(w_data[:,i]), list(m_data[:,i]))
    print(round(p_value, 3))
print()