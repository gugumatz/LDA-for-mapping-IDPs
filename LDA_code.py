import sys
import time
import pynmrstar
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import itertools
from itertools import compress
import re

# ================ LDA method for residue mapping in IDPs ================= #

# Read test data
try:
    test_data = pd.read_excel(sys.argv[1])
except 'FileNotFoundError':
    test_data = pd.read_excel(sys.argv[1], engine="odf")

# Read fasta file
with open(sys.argv[2], 'r') as f:
    fasta = [line.rstrip('\n') for line in f]
fasta = ''.join(fasta)

# Read BMRB entries for training set
with open(sys.argv[3], 'r') as f:
    line: str
    scanlist = [line.rstrip('\n') for line in f]
while scanlist[-1] == '':
    scanlist.pop(-1)

# Read connectivity chains
if sys.argv[-1] == 'chains.txt':
    with open(sys.argv[4], 'r') as f:
        chains = [x.rstrip('\n').split(" ") for x in f.readlines()]
    while chains[-1] == ['']:
        chains.pop(-1)

# ====== Automatic retrieval of training data from the BMRB database ====== #

W = {}
for k, l in enumerate(scanlist):
    waiter = True

    while waiter:
        try:
            ent = pynmrstar.Entry.from_database(l)
            waiter = False
        except (ConnectionError, OSError) as e:
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
except 'KeyError':
    TableA['HB'] = TableB.loc[:, "HB"].astype(float)
TableA['H'] = TableB.loc[:, "H"].astype(float)
TableA['CA'] = TableB.loc[:, "CA"].astype(float)
TableA['CB'] = TableB.loc[:, "CB"].astype(float)
TableA['N'] = TableB.loc[:, "N"].astype(float)
TableA['C'] = TableB.loc[:, "C"].astype(float)

try:
    TableA['HA'] = TableB.loc[:, ["HA", "HA2", "HA3"]].astype(float).mean(axis=1)
except 'KeyError':
    TableA['HA'] = TableB.loc[:, "HA"].astype(float)
TableA['amino'] = TableB['amino']
TableA['ID'] = TableB['ID']
TableA['protein'] = TableB['protein']

# ============================ Pre-processing ============================= #

AAT_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
            'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
            'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'O': 'PYL', 'S': 'SER', 'U': 'SEC',
            'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}

# Get Amino Acid Types in test protein from fasta file
fasta_set = set(fasta)
AATs_fasta = [AAT_dict[x] for x in fasta_set]
AATs_fasta.sort()

# Get column numbers of GLY and PRO
if 'GLY' in AATs_fasta:
    Gidx = AATs_fasta.index('GLY')
if 'PRO' in AATs_fasta:
    Pidx = AATs_fasta.index('PRO')

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

# ============================= Organize Data ============================= #

# Separate train and test sets
train_set = train_data.iloc[:len(TableA), :len(train_data.columns)-1]
train_classes = train_data.iloc[:len(TableA), len(train_data.columns)-1]
test_set = train_data.iloc[len(TableA):, :len(train_data.columns)-1]
test_classes = train_data.iloc[len(TableA):, len(train_data.columns)-1]
test_set = test_set.to_numpy()
miss_res = 0

# Eliminate from training set residues of AATs not present in test protein
for AAT in AATs_missing:
    idxs = (train_classes == AAT)
    train_set = train_set.loc[np.invert(idxs), :]
    train_classes = train_classes.loc[np.invert(idxs)]

# Separate GLY residues in the training set, if there are GLY in the test protein and HB or CB are input CSs
if np.logical_and('GLY' in AATs_fasta, np.logical_or('HB' in header, 'CB' in header)):
    idxs_gly = (train_classes == 'GLY')
    data_gly = train_set.loc[idxs_gly, [x for x in header if x not in ['HB', 'CB']]]
    gly_classes = train_classes.loc[idxs_gly]
    train_set = train_set.loc[np.invert(idxs_gly), :]
    train_classes = train_classes.loc[np.invert(idxs_gly)]

    # Discard entries missing CSs
    idxs = np.invert(data_gly.isnull().any(axis=1))
    gly_classes = gly_classes.loc[idxs]
    train_gly = data_gly.loc[idxs, :]
    train_gly = train_gly.to_numpy()
    miss_res = miss_res + 1

# Separate PRO residues in the training set, if there are PRO in the test protein and H or N are input CSs
if np.logical_and('PRO' in AATs_fasta, np.logical_or('H' in header, 'N' in header)):
    idxs_pro = (train_classes == 'PRO')
    data_pro = train_set.loc[idxs_pro, [x for x in header if x not in ['H', 'N']]]
    pro_classes = train_classes.loc[idxs_pro]
    train_set = train_set.loc[np.invert(idxs_pro), :]
    train_classes = train_classes.loc[np.invert(idxs_pro)]

    # Discard entries missing CSs
    idxs = np.invert(data_pro.isnull().any(axis=1))
    pro_classes = pro_classes.loc[idxs]
    train_pro = data_pro.loc[idxs, :]
    train_pro = train_pro.to_numpy()
    miss_res = miss_res + 1

# Discard entries missing CSs in the main training set
idxs = np.invert(train_set.isnull().any(axis=1))
train_classes_all = train_classes.loc[idxs]
train_set_all = train_set.loc[idxs, :]
train_set_all = train_set_all.to_numpy()

# Separate test set entries with all CSs from those missing CSs
idxs_missing = np.isnan(test_set).any(axis=1)
test_set_all = test_set[~idxs_missing, :]
test_classes_all = test_classes[~idxs_missing]

# ==================== Classify test set with all CSs ===================== #

Probabilities = np.ndarray(shape=(len(test_set), len(fasta_set)-miss_res))  # Matrix of AAT probabilities
Mdl = LinearDiscriminantAnalysis()                                          # Classification model
Mdl.fit(train_set_all, train_classes_all)                                   # Train the model
Probabilities[~idxs_missing] = Mdl.predict_proba(test_set_all)

if miss_res == 2:
    Probabilities = np.c_[Probabilities[:, :Gidx], np.zeros((len(test_set), 1)), Probabilities[:, Gidx:Pidx-1],
                          np.zeros((len(test_set), 1)), Probabilities[:, Pidx-1:]]
elif miss_res == 1:
    if 'GLY' in AATs_fasta:
        Probabilities = np.c_[Probabilities[:, :Gidx], np.zeros((len(test_set), 1)), Probabilities[:, Gidx:]]
    elif 'PRO' in AATs_fasta:
        Probabilities = np.c_[Probabilities[:, :Pidx], np.zeros((len(test_set), 1)), Probabilities[:, Pidx:]]

# ================== Classify test set with missing CSs =================== #

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
if 'N' in header:
    ord_pro.append(header.index('N'))

HB_CB_set = {"HB", "CB"}
H_N_set = {"H", "N"}

header_cols = list(range(0, len(header)))
num_missing = [i for i, x in enumerate(idxs_missing) if x]

# Loop across residues with missing CHs
for i in num_missing:
    comb = np.isnan(test_set[i, :])
    CSs = set(compress(header, comb))   # CSs missing in the current residue

    # Classify empty entries
    if np.all(comb):
        Probabilities[i, :] = np.zeros(Probabilities.shape[1])
        continue

    # Classify residues missing all HB, CB, H and N
    elif (HB_CB_set | H_N_set).issubset(CSs):
        # If the test protein has both GLY and PRO in its primary sequence
        if all(x in AATs_fasta for x in {'GLY', 'PRO'}):
            cols1 = [x for x in header_cols if x not in ord_gly]
            cols2 = [x for x in header_cols if x not in ord_pro]
            comb_aux1 = comb[cols1]
            comb_aux2 = comb[cols2]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1], train_pro[:, ~comb_aux2]))
            train_classes = np.concatenate((train_classes_all, gly_classes, pro_classes))

            Mdl_both = LinearDiscriminantAnalysis()
            Mdl_both.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl_both.predict_proba(observation)
            Probabilities[i, :] = Probs_aux

        # If the test protein only has GLY in its primary sequence
        elif 'GLY' in AATs_fasta:
            cols1 = [x for x in header_cols if x not in ord_gly]
            comb_aux1 = comb[cols1]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux1]))
            train_classes = np.concatenate((train_classes_all, gly_classes))

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
            Probabilities[i, :] = Probs_aux

        # If the test protein only has PRO in its primary sequence
        elif 'PRO' in AATs_fasta:
            cols2 = [x for x in header_cols if x not in ord_pro]
            comb_aux2 = comb[cols2]
            train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux2]))
            train_classes = np.concatenate((train_classes_all, pro_classes))

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
            Probabilities[i, :] = Probs_aux

        # If the residue is missing all HB, CB, H and N, but the test protein doesn't have GLY nor PRO
        else:
            train_set = train_set_all[:, ~comb]
            train_classes = train_classes_all

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            Probabilities[i, :] = Probs_aux

    # Classify residues missing HB and CB
    elif HB_CB_set.issubset(CSs):
        # If the test protein has GLY in its primary sequence
        if 'GLY' in AATs_fasta:
            cols = [x for x in header_cols if x not in ord_gly]
            comb_aux = comb[cols]
            train_set = np.concatenate((train_set_all[:, ~comb], train_gly[:, ~comb_aux]))
            train_classes = np.concatenate((train_classes_all, gly_classes))

            Mdl_gly = LinearDiscriminantAnalysis()
            Mdl_gly.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl_gly.predict_proba(observation)
            # If the test protein also has PRO in its primary sequence
            if np.logical_and('PRO' in AATs_fasta, np.logical_or('H' in header, 'N' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
            Probabilities[i, :] = Probs_aux

        # If the residue is missing HB and CB, but the test protein doesn't have GLY in its primary sequence
        else:
            train_set = train_set_all[:, ~comb]
            train_classes = train_classes_all

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            # If the test protein has PRO in its primary sequence
            if np.logical_and('GLY' in AATs_fasta, np.logical_or('HB' in header, 'CB' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]))
            Probabilities[i, :] = Probs_aux

    # Classify residues missing H and N
    elif H_N_set.issubset(CSs):
        # If the test protein has PRO in its primary sequence
        if 'PRO' in AATs_fasta:
            cols = [x for x in header_cols if x not in ord_pro]
            comb_aux = comb[cols]
            train_set = np.concatenate((train_set_all[:, ~comb], train_pro[:, ~comb_aux]))
            train_classes = np.concatenate((train_classes_all, pro_classes))

            Mdl_pro = LinearDiscriminantAnalysis()
            Mdl_pro.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl_pro.predict_proba(observation)
            # If the test protein also has GLY in its primary sequence
            if np.logical_and('GLY' in AATs_fasta, np.logical_or('HB' in header, 'CB' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
            Probabilities[i, :] = Probs_aux

        # If the residue is missing H and N, but the test protein doesn't have PRO in its primary sequence
        else:
            train_set = train_set_all[:, ~comb]
            train_classes = train_classes_all

            Mdl = LinearDiscriminantAnalysis()
            Mdl.fit(train_set, train_classes)
            observation = test_set[i, ~comb].reshape(1, -1)
            Probs_aux = Mdl.predict_proba(observation)
            # If it has GLY in its primary sequence
            if np.logical_and('PRO' in AATs_fasta, np.logical_or('H' in header, 'N' in header)):
                Probs_aux = np.concatenate((Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]))
            Probabilities[i, :] = Probs_aux

    # Classify residues missing any other combination of chemical shifts
    else:
        train_set = train_set_all[:, ~comb]

        Mdl_miss = LinearDiscriminantAnalysis()
        Mdl_miss.fit(train_set, train_classes_all)
        observation = test_set[i, ~comb].reshape(1, -1)
        Probs_aux = Mdl_miss.predict_proba(observation)
        if miss_res == 1:
            if 'GLY' in AATs_fasta:
                Probs_aux = np.concatenate([Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:]])
            else:
                Probs_aux = np.concatenate([Probs_aux[0, :Pidx], np.array([0]), Probs_aux[0, Pidx:]])
        elif miss_res == 2:
            Probs_aux = np.concatenate([Probs_aux[0, :Gidx], np.array([0]), Probs_aux[0, Gidx:Pidx-1],
                                        np.array([0]), Probs_aux[0, Pidx-1:]])
        Probabilities[i, :] = Probs_aux


# Write probabilities matrix to excel file
Probabilities[Probabilities < 0.1] = 0
Probs = pd.DataFrame(Probabilities, index=SSN, columns=AATs_fasta)
Probs.to_excel('Probabilities.xlsx')

# ========= Discard combinations not present in protein sequence ========== #

if sys.argv[-1] == 'chains.txt':

    # Get indices corresponding to SSNs in each chain
    chains_SSN_indices = []
    for i in chains:
        aux = i
        aux2 = []
        for j in aux:
            aux2.append(SSN.loc[SSN == int(j)].index[0])
        chains_SSN_indices.append(aux2)

    # Boolean array from rows of Probabilities array: true = non-zero probability
    non_zero_probs = []
    for i in chains_SSN_indices:
        aux = Probabilities[i, :] != 0
        non_zero_probs.append(aux)

    # Get probabilities for all possible AATs of each residue in each chain
    possible_AATs = []      # Possible AATs
    possible_probs = []     # Probabilities of possible AATs
    for i in range(len(non_zero_probs)):
        aux = non_zero_probs[i]
        cSSNid = chains_SSN_indices[i]
        aux1 = []
        aux2 = []
        for j in range(aux.shape[0]):
            aux1.append(list(compress(AATs_fasta, aux[j, :])))
            aux2.append(list(compress(Probabilities[cSSNid[j], :], aux[j, :])))
        possible_AATs.append(aux1)
        possible_probs.append(aux2)

    AAT_dict_inv = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
                    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
                    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'PYL': 'O', 'SER': 'S', 'SEC': 'U',
                    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

    # Convert possible AATs to one-letter code
    possible_AATs_conv = []
    for i in possible_AATs:
        aux = i
        aux1 = []
        for j in aux:
            aux1.append([AAT_dict_inv[x] for x in j])
        possible_AATs_conv.append(aux1)

    # Count all possible combinations for each chain based on possible AATs for each residue in the chain
    combinations = []       # Combinations vector
    combs_probs = []        # Probabilities vector
    for i in range(len(possible_AATs_conv)):
        aux = list(itertools.product(*possible_AATs_conv[i]))
        aux1 = list(itertools.product(*possible_probs[i]))
        combinations.append(aux)
        combs_probs.append(aux1)

    # Form all possible words (sequences of AATs given by chains) and get their probabilities
    words = []
    words_probs = []
    for i in range(len(combinations)):
        aux_combs = combinations[i]
        aux_probs = combs_probs[i]
        aux1 = []
        aux2 = []
        for j in range(len(aux_combs)):
            aux1.append(''.join(aux_combs[j]))
            aux2.append(np.prod(aux_probs[j]))
        words.append(aux1)
        words_probs.append(aux2)

    # Find which words actually exist in the primary sequence of the test protein and which don't exist
    existing_words = []
    existing_probs = []
    absent_words = []
    for i in range(len(words)):
        aux = words[i]
        aux1 = words_probs[i]
        aux2 = []
        aux3 = []
        aux4 = []
        for j in range(len(aux)):
            if [m.start() for m in re.finditer(aux[j], fasta)]:
                aux2.append(aux[j])
                aux3.append(aux1[j])
            else:
                aux4.append(aux[j])
        existing_words.append(aux2)
        existing_probs.append(aux3)
        absent_words.append(aux4)

    # Create data frame to export excel file
    chains_probs = pd.DataFrame(columns=['Chain #', 'SSNs', 'Possibilities', 'Probabilities', 'Discarded'])
    for i in range(len(existing_words)):
        aux1 = chains_SSN_indices[i]
        aux2 = existing_words[i]
        aux3 = existing_probs[i]
        aux4 = absent_words[i]
        df1 = pd.DataFrame(columns=['Chain #', 'SSNs', 'Possibilities', 'Probabilities', 'Discarded'])
        df2 = pd.DataFrame(columns=['Chain #', 'SSNs', 'Possibilities', 'Probabilities', 'Discarded'])
        for j in range(len(aux2)):
            if j == 0:
                df1['Chain #'] = [i+1]
                df1['SSNs'] = [aux1]
                df1['Possibilities'] = aux2[j]
                df1['Probabilities'] = aux3[j]
                df1['Discarded'] = [aux4]
                chains_probs = pd.concat([chains_probs, df1])
                aux2.pop(0)
                aux3.pop(0)
            else:
                df2['Possibilities'] = aux2
                df2['Probabilities'] = aux3
                chains_probs = pd.concat([chains_probs, df2])

    # Formatting
    writer = pd.ExcelWriter('Chains_probabilities.xlsx', engine='xlsxwriter')
    chains_probs.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('D:D', None, format1)

    # Auto-adjust column widths in excel file
    for column in chains_probs:
        column_width = max(chains_probs[column].astype(str).map(len).max(), len(column))
        col_idx = chains_probs.columns.get_loc(column)
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_width)

    writer.save()

# ================================= Plot ================================== #

x_labels = list(np.unique(AATs_fasta))
x_pos = list(range(0, len(x_labels)))
y_pos = list(range(0, len(test_set)))
legend_labels = x_labels.copy()

for i in reversed(range(0, Probabilities.shape[1])):
    if (Probabilities[:, i] == 0).all():
        legend_labels.pop(i)

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

palette = sns.color_palette(cc.glasbey, n_colors=len(legend_labels))
h = sns.relplot(x="Label",
                y="Residue",
                hue="Label",
                size="Probability",
                sizes=(40, 300), alpha=.5, palette=palette, data=plot_data, aspect=0.4)

h.ax.margins(x=0.05, y=0.02)
h.despine(top=False, right=False)
plt.ylabel("Spin system", size=20)
plt.yticks(y_pos, SSN, fontsize=8)
plt.xlabel("LDA classification", size=20)
plt.xticks(x_pos, x_labels, fontsize=10, rotation=60)
plt.grid(axis='x', color='k', linestyle='-', linewidth=0.2)
plt.grid(axis='y', color='k', linestyle=':', linewidth=0.2)
sns.move_legend(h, "upper right", bbox_to_anchor=(0.72, 0.8))
for t, l in zip(h._legend.texts, ['Labels']+legend_labels):
    t.set_text(l)

h.fig.set_dpi(100)
h.fig.set_figheight(15)
h.fig.set_figwidth(25)
h.savefig('Probabilities.svg', dpi=100)
plt.show()
