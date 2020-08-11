import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PATH_SRC = '17_to_19_standardization_v3_cluster.csv'
COLNAMES_X = ['TM', 'TMA', 'AverageTip', 'RST', 'RSI',
              'RSE', 'Time', 'CorrectSteps', 'TriedSteps']
COLNAME_Y = 'Risk'
DROP_PROPERTIES = ['macro avg', 'weighted avg']
N_SPLITS = 5

df_src = pd.read_csv(PATH_SRC)
df_src['Risk'] = np.where(df_src['Score'] >= 80, 0, 1)

kf = KFold(n_splits=N_SPLITS)


def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_kf_split_data(n, c):
    cluster = df_src.loc[df_src[f'Cluster{n}'] == c]
    cluster_x = cluster.loc[:, COLNAMES_X].to_numpy()
    cluster_y = cluster[COLNAME_Y].to_numpy()
    kf_cluster = []

    for train_index, test_index in kf.split(cluster_x):
        x_train = cluster_x[train_index]
        x_test = cluster_x[test_index]
        y_train = cluster_y[train_index]
        y_test = cluster_y[test_index]

        kf_cluster.append({
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        })

    return kf_cluster


def merge_crs(crs):
    merged_cr = {}
    for cr in crs:
        for key in cr:
            if not key in merged_cr:
                merged_cr[key] = []
            merged_cr[key].append(cr[key])

    out_cr = {}
    for key in merged_cr:
        for obj in merged_cr[key]:
            if type(obj) is dict:
                if key not in out_cr:
                    out_cr[key] = {}
                for key1 in obj:
                    if not key1 in out_cr[key]:
                        out_cr[key][key1] = obj[key1]
                    else:
                        out_cr[key][key1] += obj[key1]
            else:
                if key not in out_cr:
                    out_cr[key] = 0
                out_cr[key] += obj

    for key in out_cr:
        if type(out_cr[key]) is dict:
            for key1 in out_cr[key]:
                out_cr[key][key1] /= N_SPLITS
        else:
            out_cr[key] /= N_SPLITS

    return out_cr


# get random forest classifier classification report
def get_rfc_cr(x_train, x_test, y_train, y_test):
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    rfc_cr = classification_report(
        y_test, y_pred, zero_division=1, output_dict=True)
    for p in DROP_PROPERTIES:
        del rfc_cr[p]

    return rfc_cr


# k = 3
n3_c1 = get_kf_split_data(3, 1)
n3_c2 = get_kf_split_data(3, 2)
n3_c3 = get_kf_split_data(3, 3)

n3_c1_crs = []
n3_c2_crs = []
n3_c3_crs = []

n3_c1_g_crs = []
n3_c2_g_crs = []
n3_c3_g_crs = []


# k = 5
n5_c1 = get_kf_split_data(5, 1)
n5_c2 = get_kf_split_data(5, 2)
n5_c3 = get_kf_split_data(5, 3)
n5_c4 = get_kf_split_data(5, 4)
n5_c5 = get_kf_split_data(5, 5)

n5_c1_crs = []
n5_c2_crs = []
n5_c3_crs = []
n5_c4_crs = []
n5_c5_crs = []

n5_c1_g_crs = []
n5_c2_g_crs = []
n5_c3_g_crs = []
n5_c4_g_crs = []
n5_c5_g_crs = []

for i in range(0, N_SPLITS):
    # k = 3, general, train data
    # n3_g_x_train = np.concatenate(
    #     (n3_c1[i]['x_train'], n3_c2[i]['x_train'], n3_c3[i]['x_train']))
    # n3_g_y_train = np.concatenate(
    #     (n3_c1[i]['y_train'], n3_c2[i]['y_train'], n3_c3[i]['y_train']))

    # k = 3, cluster1
    n3_c1_cr = get_rfc_cr(
        n3_c1[i]['x_train'], n3_c1[i]['x_test'], n3_c1[i]['y_train'], n3_c1[i]['y_test'])
    n3_c1_crs.append(n3_c1_cr)
    print(n3_c1_cr, '\n')
    # print(f'{pd.DataFrame(n3_c1_cr).transpose()}\n')

    # # k = 3, cluster2
    # n3_c2_cr = get_rfc_cr(
    #     n3_c2[i]['x_train'], n3_c2[i]['x_test'], n3_c2[i]['y_train'], n3_c2[i]['y_test'])
    # n3_c2_crs.append(n3_c2_cr)

    # # k = 3, cluster3
    # n3_c3_cr = get_rfc_cr(
    #     n3_c3[i]['x_train'], n3_c3[i]['x_test'], n3_c3[i]['y_train'], n3_c3[i]['y_test'])
    # n3_c3_crs.append(n3_c3_cr)

    # n3_c3_g_cr = get_rfc_cr(
    #     n3_g_x_train, n3_c3[i]['x_test'], n3_g_y_train, n3_c3[i]['y_test'])
    # n3_c3_g_crs.append(n3_c3_g_cr)


n3_c1_crs_mean = merge_crs(n3_c1_crs)
print(pd.DataFrame(n3_c1_crs_mean).transpose())
