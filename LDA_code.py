import sys
import time
import pynmrstar
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# ================ LDA method for residue mapping in IDPs ================ #

# Load test data
test_data = pd.read_excel(sys.argv[1])
fasta = open(sys.argv[2], 'r').read().rstrip('\n')

# ---------- Automatic retrieval of data from the BMRB database ---------- #

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
        except:
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

AAT_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY',
            'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'O': 'PYL',
            'S': 'SER', 'U': 'SEC', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

# Get Amino Acid Types in test protein
fasta = set(fasta)
AATs_fasta = [0] * len(fasta)
for i, k in enumerate(fasta):
    AATs_fasta[i] = AAT_dict[k]
AATs_fasta.sort()

# AATs not in test protein
AATs_missing = list(set(AAT_dict.values())-set(AATs_fasta))

# Test data
SSN = test_data.iloc[:len(test_data), 0]    # Spin Systems Numbers
test_data = test_data.drop(['SSN'], axis=1)
header = test_data.columns.tolist()

# Sort training data
train_data = pd.DataFrame(TableA, columns=header+['amino'])
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

# Separate Glycine and Proline residues in the training set
idxs_gly = (train_classes == 'GLY')
data_gly = train_set.loc[idxs_gly, list(set(header)-{'HB', 'CB'})]
gly_classes = train_classes.loc[idxs_gly]
train_set = train_set.loc[np.invert(idxs_gly), :]
train_classes = train_classes.loc[np.invert(idxs_gly)]

idxs_pro = (train_classes == 'PRO')
data_pro = train_set.loc[idxs_pro, list(set(header)-{'H', 'N'})]
pro_classes = train_classes.loc[idxs_pro]
train_set = train_set.loc[np.invert(idxs_pro), :]
train_classes = train_classes.loc[np.invert(idxs_pro)]

# Eliminate from training set residues of AATs not present in test protein
for AAT in AATs_missing:
    idxs = (train_classes == AAT)
    train_set = train_set.loc[np.invert(idxs), :]
    train_classes = train_classes.loc[np.invert(idxs)]

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
test_classes_all = test_classes[~idxs_missing]

# ==================== Classify test set with all CSs ==================== #

Labels = pd.DataFrame(np.zeros(len(test_set),))
miss_res = len(AATs_fasta) - len(set(AATs_fasta) - {'GLY', 'PRO'})
Probabilities = np.ndarray(shape=(len(test_set), len(fasta)-miss_res))

Mdl = LinearDiscriminantAnalysis()                              # Classification model
Mdl.fit(train_set_all, train_classes_all)                       # Train the model
Labels[~idxs_missing] = Mdl.predict(test_set_all)               # Predicted classes
Probabilities[~idxs_missing] = Mdl.predict_proba(test_set_all)  # Matrix of prediction probabilities

if miss_res == 1:
    if 'GLY' in AATs_fasta:
        Gidx = AATs_fasta.index('GLY')
        Probabilities = np.c_[Probabilities[:, :Gidx], np.zeros((len(test_set), 1)), Probabilities[:, Gidx:]]
    else:
        Pidx = AATs_fasta.index('PRO')
        Probabilities = np.c_[Probabilities[:, :Pidx], np.zeros((len(test_set), 1)), Probabilities[:, Pidx:]]
elif miss_res == 2:
    Gidx = AATs_fasta.index('GLY')
    Pidx = AATs_fasta.index('PRO')
    Probabilities = np.c_[Probabilities[:, :Gidx], np.zeros((len(test_set), 1)), Probabilities[:, Gidx:Pidx-1],
                          np.zeros((len(test_set), 1)), Probabilities[:, Pidx-1:]]

# ================== Classify test set with missing CSs ================== #

# Find index of HB and CB columns
ord_gly = []
if 'HB' in header:
    ord_gly.append(header.index('HB'))
if 'CB' in header:
    ord_gly.append(header.index('CB'))

# Find index of H and N columns
ord_pro = []
if 'H' in header:
    ord_pro.append(header.index('H'))

for i in range(0, len(test_set)):
    if idxs_missing[i]:
        comb = np.isnan(test_set[i, :])

        if np.logical_and(np.all(comb[ord_gly]), ~comb[ord_pro]):
            if 'GLY' in AATs_fasta:
                comb_aux = comb[list(set(list(range(0, len(header))))-set(ord_gly))]
                train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux]))
                train_classes = np.concatenate((train_classes_all, gly_classes))

                Mdl_gly = LinearDiscriminantAnalysis()
                Mdl_gly.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl_gly.predict(observation)
                Probs_aux = Mdl_gly.predict_proba(observation)
                if 'PRO' in AATs_fasta:
                    Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
                Probabilities[i, :] = Probs_aux

            else:
                train_set = train_set_all[:, ~comb]
                train_classes = train_classes_all

                Mdl = LinearDiscriminantAnalysis()
                Mdl.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl.predict(observation)
                Probs_aux = Mdl.predict_proba(observation)
                if 'PRO' in AATs_fasta:
                    Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
                Probabilities[i, :] = Probs_aux

        elif np.logical_and(comb[ord_pro], np.any(~comb[ord_gly])):
            if 'PRO' in AATs_fasta:
                comb_aux = comb[list(set(list(range(0, len(header))))-set(ord_pro))]
                train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux]))
                train_classes = np.concatenate((train_classes_all, pro_classes))

                Mdl_pro = LinearDiscriminantAnalysis()
                Mdl_pro.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl_pro.predict(observation)
                Probs_aux = Mdl_pro.predict_proba(observation)
                if 'GLY' in AATs_fasta:
                    Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
                Probabilities[i, :] = Probs_aux

            else:
                train_set = train_set_all[:, ~comb]
                train_classes = train_classes_all

                Mdl = LinearDiscriminantAnalysis()
                Mdl.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl.predict(observation)
                Probs_aux = Mdl.predict_proba(observation)
                if 'GLY' in AATs_fasta:
                    Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
                Probabilities[i, :] = Probs_aux

        elif np.logical_and(np.all(comb[ord_gly]), comb[ord_pro]):
            if all(x in AATs_fasta for x in {'GLY', 'PRO'}):
                comb_aux1 = comb[list(set(list(range(0, len(header)))) - set(ord_gly))]
                comb_aux2 = comb[list(set(list(range(0, len(header)))) - set(ord_pro))]
                train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1], train_pro[:, ~comb_aux2]))
                train_classes = np.concatenate((train_classes_all, gly_classes, pro_classes))

                Mdl_both = LinearDiscriminantAnalysis()
                Mdl_both.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl_both.predict(observation)
                Probs_aux = Mdl_both.predict_proba(observation)
                Probabilities[i, :] = Probs_aux

            elif 'GLY' in AATs_fasta:
                comb_aux1 = comb[list(set(list(range(0, len(header)))) - set(ord_gly))]
                train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1]))
                train_classes = np.concatenate((train_classes_all, gly_classes))

                Mdl = LinearDiscriminantAnalysis()
                Mdl.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl.predict(observation)
                Probs_aux = Mdl.predict_proba(observation)
                Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
                Probabilities[i, :] = Probs_aux

            elif 'PRO' in AATs_fasta:
                comb_aux2 = comb[list(set(list(range(0, len(header)))) - set(ord_pro))]
                train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux2]))
                train_classes = np.concatenate((train_classes_all, pro_classes))

                Mdl = LinearDiscriminantAnalysis()
                Mdl.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl.predict(observation)
                Probs_aux = Mdl.predict_proba(observation)
                Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
                Probabilities[i, :] = Probs_aux

            else:
                train_set = train_set_all[:, ~comb]
                train_classes = train_classes_all

                Mdl = LinearDiscriminantAnalysis()
                Mdl.fit(train_set, train_classes)
                observation = test_set[i, ~comb].reshape(1, -1)
                Labels.iloc[i] = Mdl.predict(observation)
                Probs_aux = Mdl.predict_proba(observation)
                Probabilities[i, :] = Probs_aux

        else:
            train_set = train_set_all[:, ~comb]

            Mdl_miss = LinearDiscriminantAnalysis()
            Mdl_miss.fit(train_set, train_classes_all)
            observation = test_set[i, ~comb].reshape(1, -1)
            Labels.iloc[i] = Mdl_miss.predict(observation)
            Probs_aux = Mdl_miss.predict_proba(observation)
            if miss_res == 1:
                if 'GLY' in AATs_fasta:
                    Probs_aux = np.concatenate([Probs_aux[:, :Gidx], np.array([0]), Probs_aux[:, Gidx:]])
                else:
                    Probs_aux = np.concatenate([Probs_aux[:, :Pidx], np.array([0]), Probs_aux[:, Pidx:]])
            elif miss_res == 2:
                Probs_aux = np.concatenate([Probs_aux[:, :Gidx], np.array([0]), Probs_aux[:, Gidx:Pidx-1],
                                            np.array([0]), Probs_aux[:, Pidx-1:]])
            Probabilities[i, :] = Probs_aux

# Write probabilities matrix to excel file
Probabilities[Probabilities < 0.1] = 0
Probs = pd.DataFrame(Probabilities, index=SSN, columns=AATs_fasta)
Probs.to_excel('Probabilities.xlsx')
# ================================= Plot ================================= #

x_labels = np.unique(AATs_fasta)
x_pos = list(range(0, len(x_labels)))
y_pos = list(range(0, len(test_set)))
x_vals = []
y_vals = []
p_vals = []

for i in y_pos:
    for j in x_pos:
        if Probabilities[i, j] != 0:
            x_vals.append(x_pos[j])
            y_vals.append(y_pos[i])
            p_vals.append(Probabilities[i, j])

plot_data = pd.DataFrame(columns=['label', 'residue', 'probability'])
plot_data['Label'] = x_vals
plot_data['Residue'] = y_vals
plot_data['Probability'] = p_vals

height = len(y_vals)/30

h = sns.relplot(x="Label", y="Residue", hue="Label", size="Probability", sizes=(40, 300), alpha=.5, palette="muted",
                height=height, data=plot_data)

h.set(aspect=0.5)
h._legend.remove()
h.ax.margins(x=0.05, y=0.02)
plt.ylabel("Spin system", size=20)
plt.yticks(y_pos, SSN, fontsize=8)
h.despine(top=False, right=False)
plt.xlabel("LDA classification", size=20)
plt.xticks(x_pos, x_labels, fontsize=10, rotation=60)
plt.grid(axis='x', color='k', linestyle='-', linewidth=0.2)
plt.grid(axis='y', color='k', linestyle=':', linewidth=0.2)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()
