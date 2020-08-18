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
OUTPUT_PROPERTIES = ['accuracy', 'precision', 'recall', 'f1-score', 'support']
CLUSTER_3_NAMES = ['Instrumental', 'Executive', 'Independent']
CLUSTER_5_NAMES = ['Executive', 'Instrumental Independent',
                   'Instrumental Executive', 'Instrumental', 'Indepnedent']
OUTPUT_CSV_PATH = 'out.csv'

N_SPLITS = 5

df_src = pd.read_csv(PATH_SRC)
df_src['Risk'] = np.where(df_src['Score'] >= 80, 0, 1)

kf = KFold(n_splits=N_SPLITS)


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

        # Balance sample, risk 0 1
        df_train = pd.DataFrame(np.array(x_train), columns=COLNAMES_X)
        df_train[COLNAME_Y] = y_train
        df_train_risk_0 = df_train[df_train[COLNAME_Y] == 0]
        df_train_risk_1 = df_train[df_train[COLNAME_Y] == 1]
        df_train_reshape = pd.DataFrame()

        df_test = pd.DataFrame(np.array(x_test), columns=COLNAMES_X)
        df_test[COLNAME_Y] = y_test
        df_test_risk_0 = df_test[df_test[COLNAME_Y] == 0]
        df_test_risk_1 = df_test[df_test[COLNAME_Y] == 1]
        df_test_reshape = pd.DataFrame()

        count_train_risk_0, count_train_risk_1 = df_train[COLNAME_Y].value_counts(
        )
        if count_train_risk_0 > count_train_risk_1:
            df_train_reshape = pd.concat(
                [df_train_risk_0, df_train_risk_1.sample(count_train_risk_0, replace=True)])
        else:
            df_train_reshape = pd.concat(
                [df_train_risk_1, df_train_risk_0.sample(count_train_risk_1, replace=True)])

        count_test_risk_0, count_test_risk_1 = df_test[COLNAME_Y].value_counts(
        )
        if count_test_risk_0 > count_test_risk_1:
            df_test_reshape = pd.concat(
                [df_test_risk_0, df_test_risk_1.sample(count_test_risk_0, replace=True)])
        else:
            df_train_reshape = pd.concat(
                [df_train_risk_1, df_train_risk_0.sample(count_test_risk_1, replace=True)])

        x_train = df_train_reshape.loc[:, COLNAMES_X].to_numpy()
        y_train = df_train_reshape[COLNAME_Y]
        x_test = df_test_reshape.loc[:, COLNAMES_X].to_numpy()
        y_test = df_test_reshape[COLNAME_Y]

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
    return rfc_cr


def appendCrToRow(cr, row, risk):
    row.append(cr['accuracy'])
    row.append(cr[f'{risk}']['precision'])
    row.append(cr[f'{risk}']['recall'])
    row.append(cr[f'{risk}']['f1-score'])
    row.append(cr[f'{risk}']['support'])


# k = 3
n3_c1 = get_kf_split_data(3, 1)
n3_c2 = get_kf_split_data(3, 2)
n3_c3 = get_kf_split_data(3, 3)

n3_cs = []
n3_c1_crs = []
n3_c2_crs = []
n3_c3_crs = []

n3_g_cs = []
n3_c1_g_crs = []
n3_c2_g_crs = []
n3_c3_g_crs = []

n3_crs_means = []
n3_g_crs_means = []


# k = 5
n5_c1 = get_kf_split_data(5, 1)
n5_c2 = get_kf_split_data(5, 2)
n5_c3 = get_kf_split_data(5, 3)
n5_c4 = get_kf_split_data(5, 4)
n5_c5 = get_kf_split_data(5, 5)

n5_cs = []
n5_c1_crs = []
n5_c2_crs = []
n5_c3_crs = []
n5_c4_crs = []
n5_c5_crs = []

n5_g_cs = []
n5_c1_g_crs = []
n5_c2_g_crs = []
n5_c3_g_crs = []
n5_c4_g_crs = []
n5_c5_g_crs = []

n5_crs_means = []
n5_g_crs_means = []

for i in range(0, N_SPLITS):
    # k = 3, general, train data
    n3_g_x_train = np.concatenate(
        (n3_c1[i]['x_train'], n3_c2[i]['x_train'], n3_c3[i]['x_train']))
    n3_g_y_train = np.concatenate(
        (n3_c1[i]['y_train'], n3_c2[i]['y_train'], n3_c3[i]['y_train']))

    # k = 3, cluster1
    n3_c1_cr = get_rfc_cr(
        n3_c1[i]['x_train'], n3_c1[i]['x_test'], n3_c1[i]['y_train'], n3_c1[i]['y_test'])
    n3_c1_crs.append(n3_c1_cr)

    n3_c1_g_cr = get_rfc_cr(
        n3_g_x_train, n3_c1[i]['x_test'], n3_g_y_train, n3_c1[i]['y_test'])
    n3_c1_g_crs.append(n3_c1_g_cr)

    # k = 3, cluster2
    n3_c2_cr = get_rfc_cr(
        n3_c2[i]['x_train'], n3_c2[i]['x_test'], n3_c2[i]['y_train'], n3_c2[i]['y_test'])
    n3_c2_crs.append(n3_c2_cr)

    n3_c2_g_cr = get_rfc_cr(
        n3_g_x_train, n3_c2[i]['x_test'], n3_g_y_train, n3_c2[i]['y_test'])
    n3_c2_g_crs.append(n3_c2_g_cr)

    # k = 3, cluster3
    n3_c3_cr = get_rfc_cr(
        n3_c3[i]['x_train'], n3_c3[i]['x_test'], n3_c3[i]['y_train'], n3_c3[i]['y_test'])
    n3_c3_crs.append(n3_c3_cr)

    n3_c3_g_cr = get_rfc_cr(
        n3_g_x_train, n3_c3[i]['x_test'], n3_g_y_train, n3_c3[i]['y_test'])
    n3_c3_g_crs.append(n3_c3_g_cr)

    # k = 5, general, train data
    n5_g_x_train = np.concatenate(
        (n5_c1[i]['x_train'], n5_c2[i]['x_train'], n5_c3[i]['x_train'], n5_c4[i]['x_train'], n5_c5[i]['x_train']))
    n5_g_y_train = np.concatenate(
        (n5_c1[i]['y_train'], n5_c2[i]['y_train'], n5_c3[i]['y_train'], n5_c4[i]['y_train'], n5_c5[i]['y_train']))

    # k = 5, cluster1
    n5_c1_cr = get_rfc_cr(
        n5_c1[i]['x_train'], n5_c1[i]['x_test'], n5_c1[i]['y_train'], n5_c1[i]['y_test'])
    n5_c1_crs.append(n5_c1_cr)

    n5_c1_g_cr = get_rfc_cr(
        n5_g_x_train, n5_c1[i]['x_test'], n5_g_y_train, n5_c1[i]['y_test'])
    n5_c1_g_crs.append(n5_c1_g_cr)

    # k = 5, cluster2
    n5_c2_cr = get_rfc_cr(
        n5_c2[i]['x_train'], n5_c2[i]['x_test'], n5_c2[i]['y_train'], n5_c2[i]['y_test'])
    n5_c2_crs.append(n5_c2_cr)

    n5_c2_g_cr = get_rfc_cr(
        n5_g_x_train, n5_c2[i]['x_test'], n5_g_y_train, n5_c2[i]['y_test'])
    n5_c2_g_crs.append(n5_c2_g_cr)

    # k = 5, cluster3
    n5_c3_cr = get_rfc_cr(
        n5_c3[i]['x_train'], n5_c3[i]['x_test'], n5_c3[i]['y_train'], n5_c3[i]['y_test'])
    n5_c3_crs.append(n5_c3_cr)

    n5_c3_g_cr = get_rfc_cr(
        n5_g_x_train, n5_c3[i]['x_test'], n5_g_y_train, n5_c3[i]['y_test'])
    n5_c3_g_crs.append(n5_c3_g_cr)

    # k = 5, cluster4
    n5_c4_cr = get_rfc_cr(
        n5_c4[i]['x_train'], n5_c4[i]['x_test'], n5_c4[i]['y_train'], n5_c4[i]['y_test'])
    n5_c4_crs.append(n5_c4_cr)

    n5_c4_g_cr = get_rfc_cr(
        n5_g_x_train, n5_c4[i]['x_test'], n5_g_y_train, n5_c4[i]['y_test'])
    n5_c4_g_crs.append(n5_c4_g_cr)

    # k = 5, cluster5
    n5_c5_cr = get_rfc_cr(
        n5_c5[i]['x_train'], n5_c5[i]['x_test'], n5_c5[i]['y_train'], n5_c5[i]['y_test'])
    n5_c5_crs.append(n5_c5_cr)

    n5_c5_g_cr = get_rfc_cr(
        n5_g_x_train, n5_c5[i]['x_test'], n5_g_y_train, n5_c5[i]['y_test'])
    n5_c5_g_crs.append(n5_c5_g_cr)


n3_cs.extend([n3_c1_crs, n3_c2_crs, n3_c3_crs])
n3_g_cs.extend([n3_c1_g_crs, n3_c2_g_crs, n3_c3_g_crs])

n3_c1_crs_mean = merge_crs(n3_c1_crs)
n3_c2_crs_mean = merge_crs(n3_c2_crs)
n3_c3_crs_mean = merge_crs(n3_c3_crs)
n3_crs_means.extend([n3_c1_crs_mean, n3_c2_crs_mean, n3_c3_crs_mean])

n3_c1_g_crs_mean = merge_crs(n3_c1_g_crs)
n3_c2_g_crs_mean = merge_crs(n3_c2_g_crs)
n3_c3_g_crs_mean = merge_crs(n3_c3_g_crs)
n3_g_crs_means.extend([n3_c1_g_crs_mean, n3_c2_g_crs_mean, n3_c3_g_crs_mean])

n5_cs.extend([n5_c1_crs, n5_c2_crs, n5_c3_crs, n5_c4_crs, n5_c5_crs])
n5_g_cs.extend([n5_c1_g_crs, n5_c2_g_crs,
                n5_c3_g_crs, n5_c4_g_crs, n5_c5_g_crs])

n5_c1_crs_mean = merge_crs(n5_c1_crs)
n5_c2_crs_mean = merge_crs(n5_c2_crs)
n5_c3_crs_mean = merge_crs(n5_c3_crs)
n5_c4_crs_mean = merge_crs(n5_c4_crs)
n5_c5_crs_mean = merge_crs(n5_c5_crs)
n5_crs_means.extend([n5_c1_crs_mean, n5_c2_crs_mean,
                     n5_c3_crs_mean, n5_c4_crs_mean, n5_c5_crs_mean])

n5_c1_g_crs_mean = merge_crs(n5_c1_g_crs)
n5_c2_g_crs_mean = merge_crs(n5_c2_g_crs)
n5_c3_g_crs_mean = merge_crs(n5_c3_g_crs)
n5_c4_g_crs_mean = merge_crs(n5_c4_g_crs)
n5_c5_g_crs_mean = merge_crs(n5_c5_g_crs)
n5_g_crs_means.extend([n5_c1_g_crs_mean, n5_c2_g_crs_mean,
                       n5_c3_g_crs_mean, n5_c4_g_crs_mean, n5_c5_g_crs_mean])

out_rows = []
for i in range(0, len(n3_cs)):
    risk_0_row = [3, CLUSTER_3_NAMES[i], f'cluster{i + 1}', 0]
    risk_1_row = [3, CLUSTER_3_NAMES[i], f'cluster{i + 1}', 1]
    risk_0_g_row = [3, CLUSTER_3_NAMES[i], f'cluster{i + 1}-general', 0]
    risk_1_g_row = [3, CLUSTER_3_NAMES[i], f'cluster{i + 1}-general', 1]

    for n3_c_cr in n3_cs[i]:
        appendCrToRow(n3_c_cr, risk_0_row, 0)
        appendCrToRow(n3_c_cr, risk_1_row, 1)

    for n3_g_c_cr in n3_g_cs[i]:
        appendCrToRow(n3_g_c_cr, risk_0_g_row, 0)
        appendCrToRow(n3_g_c_cr, risk_1_g_row, 1)

    appendCrToRow(n3_crs_means[i], risk_0_row, 0)
    appendCrToRow(n3_crs_means[i], risk_1_row, 1)
    appendCrToRow(n3_g_crs_means[i], risk_0_g_row, 0)
    appendCrToRow(n3_g_crs_means[i], risk_1_g_row, 1)

    out_rows.extend([risk_0_row, risk_1_row, risk_0_g_row, risk_1_g_row])

for i in range(0, len(n5_cs)):
    risk_0_row = [5, CLUSTER_5_NAMES[i], f'cluster{i + 1}', 0]
    risk_1_row = [5, CLUSTER_5_NAMES[i], f'cluster{i + 1}', 1]
    risk_0_g_row = [5, CLUSTER_5_NAMES[i], f'cluster{i + 1}-general', 0]
    risk_1_g_row = [5, CLUSTER_5_NAMES[i], f'cluster{i + 1}-general', 1]

    for n5_c_cr in n5_cs[i]:
        appendCrToRow(n5_c_cr, risk_0_row, 0)
        appendCrToRow(n5_c_cr, risk_1_row, 1)

    for n5_g_c_cr in n5_g_cs[i]:
        appendCrToRow(n5_g_c_cr, risk_0_g_row, 0)
        appendCrToRow(n5_g_c_cr, risk_1_g_row, 1)

    appendCrToRow(n5_crs_means[i], risk_0_row, 0)
    appendCrToRow(n5_crs_means[i], risk_1_row, 1)
    appendCrToRow(n5_g_crs_means[i], risk_0_g_row, 0)
    appendCrToRow(n5_g_crs_means[i], risk_1_g_row, 1)

    out_rows.extend([risk_0_row, risk_1_row, risk_0_g_row, risk_1_g_row])

out_colnames = ['k-means-n', 'name', 'model', 'risk']
for i in range(0, N_SPLITS):
    for p in OUTPUT_PROPERTIES:
        out_colnames.append(f'k-fold-{i + 1}-{p}')
for p in OUTPUT_PROPERTIES:
    out_colnames.append(f'mean-{p}')

out_df = pd.DataFrame(np.array(out_rows), columns=out_colnames)
out_df.to_csv(OUTPUT_CSV_PATH, index=False)
