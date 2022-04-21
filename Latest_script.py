import pynmrstar
import numpy as np
import time
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# ========== Automatic retrieval of data from the BMRB database ========== #

# List of BMRB identifiers containing NMR data of IDPs
scanlist = [6436, 6869, 11526, 15176, 15179, 15180, 15201, 15225, 15430, 15883,
            15884, 16296, 16445, 17290, 17483, 19258, 25118, 30205]

W = {}
for k, l in enumerate(scanlist):
    waiter = True

    while waiter:
        try:
            ent = pynmrstar.Entry.from_database(l)
            waiter = False
        except IOError:
            print('connection error on: ' + str(l))
            time.sleep(10)
            waiter = True
    spectral_peaks = ent.get_saveframes_by_category('assigned_chemical_shifts')

    if len(spectral_peaks) > 0:
        L = []
        for x in spectral_peaks[0]['_Atom_chem_shift']:
            L.append([x[5], x[6], x[7], x[10]])
        W[k] = L
wz = W[0][0][0]

Seq = {}
check = ['ID', 'numb', 'amino', 'protein', 'H', 'HB', 'HB1', 'HB2', 'HB3', 'CA', 'CB', 'C', 'CO', 'N']
indexTB = []

for key in W:
    wz = 'rrrrrrr'
    for k in W[key]:
        if k[0] != wz:
            idR = k[1] + k[0] + '_' + str(scanlist[key])
            indexTB.append(idR)
            wz = k[0]
TableB = pd.DataFrame(index=indexTB, columns=check)

for key in W:
    for k in W[key]:
        print(k, end='\r')
        idR = k[1] + k[0] + '_' + str(scanlist[key])
        TableB.at[idR, 'ID'] = k[1] + k[0]
        TableB.at[idR, 'numb'] = int(k[0])
        TableB.at[idR, 'amino'] = (k[1])
        TableB.at[idR, k[2]] = float(k[3])
        TableB.at[idR, 'protein'] = scanlist[key]

TableA = pd.DataFrame(index=indexTB, columns=['ID', 'HA', 'HB', 'CA', 'CB', 'C']).astype('float')
try:
    TableA['HB'] = TableB.loc[:, "HB":"HB3"].astype(float).mean(axis=1).astype(float)
except:
    TableA['HB'] = TableB.loc[:, "HB"].astype(float)
TableA['H'] = TableB.loc[:, "H"].astype(float)
TableA['CA'] = TableB.loc[:, "CA"].astype(float)
TableA['CB'] = TableB.loc[:, "CB"].astype(float)
TableA['N'] = TableB.loc[:, "N"].astype(float)
TableA['C'] = TableB.loc[:, "C"].astype(float)

try:
    TableA['HA'] = TableB.loc[:, ["HA", "HA2", "HA3"]].astype(float).mean(axis=1)
except:
    TableA['HA'] = TableB.loc[:, "HA"].astype(float)
TableA['amino'] = TableB['amino']
TableA['ID'] = TableB['ID']

# ============================ Pre-processing ============================ #

# Load test data
test_data = pd.read_excel('aSYN.xlsx', names=['SSN', 'HA', 'HB', 'CA', 'CB', 'H', 'N', 'C', 'amino'], header=None)
SSN = test_data.iloc[:len(test_data), 0]    # Spin Systems Numbering
test_data = test_data.drop(['SSN'], axis=1)

# Sort training data
train_data = pd.DataFrame(TableA, columns=['HA', 'HB', 'CA', 'CB', 'H', 'N', 'C', 'amino'])
train_data = train_data.sort_values(['amino'], ascending=True)

# Pareto scaling of train and test data simultaneously
train_data = pd.concat([train_data, test_data], ignore_index=True)
for k in train_data:
    if k == 'amino':
        break
    train_data[k] = (train_data[k] - np.mean(train_data[k])) / (np.sqrt(np.std(train_data[k])))

# ============================ Organize Data ============================= #

# Separate train and test sets
train_set = train_data.iloc[:len(TableA), :len(train_data.columns)-1]
train_classes = train_data.iloc[:len(TableA), len(train_data.columns)-1]
test_set = train_data.iloc[len(TableA):, :len(train_data.columns)-1]
test_classes = train_data.iloc[len(TableA):, len(train_data.columns)-1]
test_set = test_set.to_numpy()
x_labels = np.unique(train_classes)

# Separate Glycine and Proline residues in the training set
idxs_gly = (train_classes == 'GLY')
data_gly = train_set.loc[idxs_gly, ["HA", "CA", "H", "N", "C"]]
gly_classes = train_classes.loc[idxs_gly]
train_set = train_set.loc[np.invert(idxs_gly), :]
train_classes = train_classes.loc[np.invert(idxs_gly)]

idxs_pro = (train_classes == 'PRO')
data_pro = train_set.loc[idxs_pro, ["HA", "HB", "CA", "CB", "C"]]
pro_classes = train_classes.loc[idxs_pro]
train_set = train_set.loc[np.invert(idxs_pro), :]
train_classes = train_classes.loc[np.invert(idxs_pro)]

# Discard entries missing CSs in the training sets
idxs = np.invert(train_set.isnull().any(axis=1))
train_classes_all = train_classes.loc[idxs]
train_set_all = train_set.loc[idxs, :]
train_set_all = train_set_all.to_numpy()

idxs = np.invert(data_gly.isnull().any(axis=1))
gly_classes = gly_classes.loc[idxs]
train_gly = data_gly.loc[idxs, :]
train_gly = train_gly.to_numpy()

idxs = np.invert(data_pro.isnull().any(axis=1))
pro_classes = pro_classes.loc[idxs]
train_pro = data_pro.loc[idxs, :]
train_pro = train_pro.to_numpy()

# Separate test set entries with all CSs from those missing CSs
idxs_missing = np.isnan(test_set).any(axis=1)
test_set_all = test_set[~idxs_missing, :]
test_set_missing = test_set[idxs_missing, :]
test_classes_all = test_classes[~idxs_missing]
test_classes_missing = test_classes[idxs_missing]

# ==================== Classify test set with all CSs ==================== #

Labels = pd.DataFrame(np.zeros(len(test_set),))
Probabilities = np.ndarray(shape=(len(test_set), 18))

Mdl = LinearDiscriminantAnalysis()                              # Classification model
Mdl.fit(train_set_all, train_classes_all)                       # Train the model
Labels[~idxs_missing] = Mdl.predict(test_set_all)               # Predicted classes
Probabilities[~idxs_missing] = Mdl.predict_proba(test_set_all)  # Matrix of prediction probabilities
Probabilities = np.c_[Probabilities[:, :7], np.zeros((len(test_set), 1)), Probabilities[:, 7:13],
                      np.zeros((len(test_set), 1)), Probabilities[:, 13:]]

# ================== Classify test set with missing CSs ================== #

nans = np.isnan(test_set_missing)
combs = np.unique(nans, axis=0)

for i in range(0, len(test_set)):
    if idxs_missing[i]:
        comb = np.isnan(test_set[i, :])

        if np.logical_and(np.all(comb[[1, 3]]), np.any(~comb[[4, 5]])):
            comb_aux = comb[[0, 2, 4, 5, 6]]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux]))
            train_classes = np.concatenate((train_classes_all, gly_classes))
            #test_set = test_set_missing[idxs, :]
            #test_set = test_set[:, ~comb]

            Mdl_gly = LinearDiscriminantAnalysis()
            Mdl_gly.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Labels.iloc[i] = Mdl_gly.predict(observation)
            Probs_aux = Mdl_gly.predict_proba(observation)
            # Probabilities[i, :] = Mdl_gly.predict_proba(obs)
            Probs_aux = np.concatenate((Probs_aux[0, :13], np.array([0]), Probs_aux[0, 13:]))
            Probabilities[i, :] = Probs_aux

            #try:
            #    Labels_gly = np.concatenate((Labels_gly, Mdl_gly.predict(test_set)))
            #    Probabilities_gly = np.concatenate((Probabilities_gly, Mdl_gly.predict_proba(test_set)))
            #except NameError:
            #    Labels_gly = Mdl_gly.predict(test_set)
            #    Probabilities_gly = Mdl_gly.predict_proba(test_set)

        elif np.logical_and(np.all(comb[[4, 5]]), np.any(~comb[[1, 3]])):
            comb_aux = comb[[0, 1, 2, 3, 6]]
            train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux]))
            train_classes = np.concatenate((train_classes_all, pro_classes))

            Mdl_pro = LinearDiscriminantAnalysis()
            Mdl_pro.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Labels.iloc[i] = Mdl_pro.predict(observation)
            Probs_aux = Mdl_pro.predict_proba(observation)
            Probs_aux = np.concatenate((Probs_aux[0, :7], np.array([0]), Probs_aux[0, 7:]))
            Probabilities[i, :] = Probs_aux

        elif np.logical_and(np.all(comb[[1, 3]]), np.all(comb[[4, 5]])):
            comb_aux1 = comb[[0, 2, 4, 5, 6]]
            comb_aux2 = comb[[0, 1, 2, 3, 6]]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1], train_pro[:, ~comb_aux2]))
            train_classes = np.concatenate((train_classes_all, gly_classes, pro_classes))

            Mdl_both = LinearDiscriminantAnalysis()
            Mdl_both.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Labels.iloc[i] = Mdl_both.predict(observation)
            Probs_aux = Mdl_both.predict_proba(observation)
            Probabilities[i, :] = Probs_aux

        else:
            train_set = train_set_all[:, ~comb]

            Mdl_miss = LinearDiscriminantAnalysis()
            Mdl_miss.fit(train_set, train_classes_all)
            observation = test_set[i, ~comb].reshape(1, -1)
            Labels.iloc[i] = Mdl_miss.predict(observation)
            Probs_aux = Mdl_miss.predict_proba(observation)
            Probs_aux = np.concatenate((Probs_aux[0, :7], np.array([0]), Probs_aux[0, 7:13],
                                        np.array([0]), Probs_aux[0, 13:]))
            Probabilities[i, :] = Probs_aux

Probs = pd.DataFrame(Probabilities, index=SSN, columns=x_labels)
Probs.to_excel('Probabilities.xlsx')
# ================================= Plot ================================= #

counts = np.cumsum(Labels.value_counts(sort=False))

x_ticks = list(range(0, 20))
y_numbs = list(range(0, len(test_set)))
color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C9', 'C7', 'C8']

fig, ax = plt.subplots(figsize=(8,8))
k = 0

for i in y_numbs:
    for j in x_ticks:
        if Probabilities[i, j] != 0:
            if i == counts[k]:
                k = k + 1
            ax.scatter(x_ticks[j], y_numbs[i], s=600, c=color_list[k % 10], marker='_', alpha=Probabilities[i, j])

plt.ylabel("Spin systems", fontsize=20)
ax.set_yticks(y_numbs)
plt.yticks(y_numbs, SSN, fontsize=4)
ax.tick_params(axis='y', which='major', color='w')
plt.yticks(y_numbs, SSN, fontsize=6)

plt.xlabel("LDA classification", fontsize=20)
plt.xticks(x_ticks, x_labels, fontsize=14, rotation=45)

plt.grid(axis='x', color='k', linestyle='-', linewidth=0.2)
plt.grid(axis='y', color='k', linestyle=':', linewidth=0.2)
ax.set_aspect(0.2)
plt.savefig('Probabilities.pdf')
plt.show()

