import re
import sys
import pynmrstar
import numpy as np
import pandas as pd
from itertools import compress
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

# Read test data
test_data = pd.read_excel(sys.argv[1], engine='openpyxl')

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

# ======= Read NMR data of training IDPs from downloaded BMRB files ======= #

names = ['H', 'HB', 'HB1', 'HB2', 'HB3', 'CA', 'CB', 'C', 'CO', 'N', 'amino', 'protein']
TableB = pd.DataFrame(columns=names)

for k, l in enumerate(scanlist):
    waiter = True
    sys.stdout.write('\rScanning data of training proteins: %i/%i' % (k + 1, len(scanlist)))
    sys.stdout.flush()

    while waiter:
        try:
            ent = pynmrstar.Entry.from_database(l)
            waiter = False
        except (ConnectionError, OSError) as e:
            print('connection error on: ' + str(l))
            time.sleep(2)
            waiter = True
    spectral_peaks = ent.get_saveframes_by_category('assigned_chemical_shifts')

    if len(spectral_peaks) > 0:
        TableC = pd.DataFrame(columns=names)
        w = spectral_peaks[0]['_Atom_chem_shift.Seq_ID']
        x = spectral_peaks[0]['_Atom_chem_shift.Comp_ID']
        y = spectral_peaks[0]['_Atom_chem_shift.Atom_ID']
        z = spectral_peaks[0]['_Atom_chem_shift.Val']
        wp = w[0]
        n = 0

        for i, j in enumerate(w):
            if j != wp:
                n = n + 1
                wp = w[i]
            TableC.at[n, 'amino'] = x[i]
            TableC.at[n, 'protein'] = l
            TableC.at[n, y[i]] = float(z[i])
        TableB = pd.concat([TableB, TableC], ignore_index=True)

print('\n')
TableA = pd.DataFrame(columns=['HN', 'N', 'CO', 'HA', 'HB', 'CA', 'CB', 'amino', 'protein']).astype('float')
try:
    TableA['HB'] = TableB.loc[:, "HB":"HB3"].astype(float).mean(axis=1).astype(float)
except 'KeyError':
    TableA['HB'] = TableB.loc[:, "HB"].astype(float)
TableA['HN'] = TableB.loc[:, "H"].astype(float)
TableA['CA'] = TableB.loc[:, "CA"].astype(float)
TableA['CB'] = TableB.loc[:, "CB"].astype(float)
TableA['N'] = TableB.loc[:, "N"].astype(float)
TableA['CO'] = TableB.loc[:, "C"].astype(float)

try:
    TableA['HA'] = TableB.loc[:, ["HA", "HA2", "HA3"]].astype(float).mean(axis=1)
except 'KeyError':
    TableA['HA'] = TableB.loc[:, "HA"].astype(float)
TableA['amino'] = TableB['amino']
TableA['protein'] = TableB['protein']

# ============================ Pre-processing ============================= #

# Dictionary: from 1 char codes to 3 chars codes
AAT_dict = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN',
            'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS',
            'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP',
            'Y': 'TYR', 'V': 'VAL'}

# Get all Amino Acid Types in test protein from seq file
AATs_unique = list(set([AAT_dict[x] for x in set(fasta)]))
AATs_unique.sort()

# AATs not in test protein
AATs_missing = list(set(AAT_dict.values()) - set(AATs_unique))

# Test data
SSN = test_data.iloc[:len(test_data), 0]    # Spin Systems Numbers
test_set = test_data.drop(['SSN'], axis=1)
nuclei = test_set.columns.tolist()

# Sort training data
train_data = pd.DataFrame(TableA, columns=nuclei+['amino', 'protein'])
train_data = train_data.sort_values(['protein', 'amino'], ascending=True)

# Pareto scaling of train and test data simultaneously
for_scaling = pd.concat([test_set, train_data], ignore_index=True)
for k in for_scaling:
    if k == 'amino':
        break
    for_scaling[k] = (for_scaling[k] - np.mean(for_scaling[k])) / (np.sqrt(np.std(for_scaling[k])))

test_set = for_scaling.iloc[:len(test_data), :]
for_scaling = for_scaling.iloc[len(test_data):, :]

# Eliminate from training set residues of AATs not present in test protein
for AAT in AATs_missing:
    for_scaling = for_scaling.drop(for_scaling[for_scaling["amino"] == AAT].index)

classes = for_scaling["amino"].copy()
train_set = for_scaling.drop(["protein"], axis=1)

# ============================ Classification ============================= #

# Get column positions of GLY and PRO
if 'GLY' in AATs_unique:
    G_idx = AATs_unique.index('GLY')
if 'PRO' in AATs_unique:
    P_idx = AATs_unique.index('PRO')

# Initialization
probs = np.ndarray(shape=(len(test_set), len(AATs_unique)))
Mdl = LinearDiscriminantAnalysis()
test_nan = np.array(test_set.isnull())
combs_nan = np.unique(test_nan, axis=0)
nuclei_ix = list(range(0, len(nuclei)))

# Loop over combinations
for i in combs_nan:
    idxs_nan = np.all(test_nan == i, axis=1)
    nuclei_meas = list(compress(nuclei, ~i))
    test_missing_comb = test_set[idxs_nan][nuclei_meas]
    print('Classifying residues with CSs: ', nuclei_meas)

    # Classify residues with all NaN values
    if np.all(i):
        a = np.empty((idxs_nan.sum(), probs.shape[1]))
        a[:] = np.nan
        probs[idxs_nan, :] = a
        continue

    # Classify residues that could be GLY or PRO
    elif {"HB", "CB", "HN"}.isdisjoint(nuclei_meas) and all(x in AATs_unique for x in {'GLY', 'PRO'}):
        train_set_aux = train_set[nuclei_meas]
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = classes[~idxs]

        Mdl_clone = clone(Mdl)
        Mdl_clone.fit(train_set_aux, labels)
        probs[idxs_nan, :] = Mdl_clone.predict_proba(test_missing_comb)

    # Classify residues that could be GLY
    elif {"HB", "CB"}.isdisjoint(set(nuclei_meas)) and 'GLY' in AATs_unique:
        train_set_aux = train_set[train_set["amino"] != 'PRO']
        labels = classes[classes != 'PRO']
        train_set_aux = train_set_aux[nuclei_meas]
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = labels[~idxs]

        Mdl_clone = clone(Mdl)
        Mdl_clone.fit(train_set_aux, labels)
        probs_aux = Mdl_clone.predict_proba(test_missing_comb)
        if 'PRO' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]]
        else:
            probs[idxs_nan, :] = probs_aux

    # Classify residues that could be PRO
    elif {"HN"}.isdisjoint(set(nuclei_meas)) and 'PRO' in AATs_unique:
        train_set_aux = train_set[train_set["amino"] != 'GLY']
        labels = classes[classes != 'GLY']
        train_set_aux = train_set_aux[nuclei_meas]
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = labels[~idxs]

        Mdl_clone = clone(Mdl)
        Mdl_clone.fit(train_set_aux, labels)
        probs_aux = Mdl_clone.predict_proba(test_missing_comb)
        if 'GLY' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]]
        else:
            probs[idxs_nan, :] = probs_aux

    # Classify other residues
    else:
        train_set_aux = train_set[np.logical_or(train_set["amino"] != 'PRO', train_set["amino"] != 'GLY')]
        labels = classes[np.logical_or(classes != 'PRO', classes != 'GLY')]
        train_set_aux = train_set_aux[nuclei_meas]
        idxs = train_set_aux.isnull().any(axis=1)
        train_set_aux = train_set_aux[~idxs]
        labels = labels[~idxs]

        Mdl_clone = clone(Mdl)
        Mdl_clone.fit(train_set_aux, labels)
        probs_aux = Mdl_clone.predict_proba(test_missing_comb)
        if all(x in AATs_unique for x in {'GLY', 'PRO'}):
            probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)),
            probs_aux[:, G_idx:P_idx - 1], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx - 1:]]
        elif 'PRO' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :P_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, P_idx:]]
        elif 'GLY' in AATs_unique:
            probs[idxs_nan, :] = np.c_[probs_aux[:, :G_idx], np.zeros((idxs_nan.sum(), 1)), probs_aux[:, G_idx:]]
        else:
            probs[idxs_nan, :] = probs_aux

# Set threshold of probabilities (for more clear results)
for i in combs_nan:
    idxs_nan = np.all(test_nan == i, axis=1)
    nuclei_meas = list(compress(nuclei, ~i))
    test_missing_comb = test_set[idxs_nan][nuclei_meas]
    if {'CB'}.issubset(test_missing_comb):
        probs_aux = probs[idxs_nan, :]
        probs_aux[probs_aux < 0.05] = 0
        probs[idxs_nan, :] = probs_aux
    else:
        probs_aux = probs[idxs_nan, :]
        probs_aux[probs_aux < 0.1] = 0
        probs[idxs_nan, :] = probs_aux

# Write probabilities matrix to excel file
df = pd.DataFrame(probs, index=SSN, columns=AATs_unique)
df.to_excel('Probabilities.xlsx')

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
        aux = probs[i, :] != 0
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
            aux1.append(list(compress(AATs_unique, aux[j, :])))
            aux2.append(list(compress(probs[cSSNid[j], :], aux[j, :])))
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
    words_probs_mean = []
    words_probs_lowest = []
    for i in range(len(combinations)):
        aux_combs = combinations[i]
        aux_probs = combs_probs[i]
        aux1 = []
        aux2 = []
        aux3 = []
        for j in range(len(aux_combs)):
            aux1.append(''.join(aux_combs[j]))
            aux2.append(np.mean(aux_probs[j]))
            aux3.append(np.amin(aux_probs[j]))
        words.append(aux1)
        words_probs_mean.append(aux2)
        words_probs_lowest.append(aux3)

    # Find which words actually exist in the primary sequence of the test protein and which don't exist
    existing_words = []
    existing_probs_mean = []
    existing_probs_lowest = []
    absent_words = []
    for i in range(len(words)):
        aux = words[i]
        aux1 = words_probs_mean[i]
        aux2 = words_probs_lowest[i]
        aux3 = []
        aux4 = []
        aux5 = []
        aux6 = []
        for j in range(len(aux)):
            if [m.start() for m in re.finditer(aux[j], fasta)]:
                aux3.append(aux[j])
                aux4.append(aux1[j])
                aux6.append(aux2[j])
            else:
                aux5.append(aux[j])
        existing_words.append(aux3)
        existing_probs_mean.append(aux4)
        existing_probs_lowest.append(aux6)
        absent_words.append(aux5)

    # Create data frame to export excel file
    chains_probs = pd.DataFrame(columns=['Chain #', 'SSNs', 'Possibilities', 'Mean prob', 'Lowest prob', 'Discarded'])
    for i in range(len(existing_words)):
        aux1 = SSN.iloc[chains_SSN_indices[i]].values.tolist()
        aux2 = existing_words[i]
        aux3 = existing_probs_mean[i]
        aux4 = existing_probs_lowest[i]
        aux5 = absent_words[i]
        df1 = pd.DataFrame(columns=['Chain #', 'SSNs', 'Possibilities', 'Mean prob', 'Lowest prob', 'Discarded'])
        df2 = pd.DataFrame(columns=['Chain #', 'SSNs', 'Possibilities', 'Mean prob', 'Lowest prob', 'Discarded'])
        for j in range(len(aux2)):
            if j == 0:
                df1['Chain #'] = [i+1]
                df1['SSNs'] = [aux1]
                df1['Possibilities'] = aux2[j]
                df1['Mean prob'] = aux3[j]
                df1['Lowest prob'] = aux4[j]
                df1['Discarded'] = [aux5]
                chains_probs = pd.concat([chains_probs, df1])
                aux2.pop(0)
                aux3.pop(0)
                aux4.pop(0)
            else:
                df2['Possibilities'] = aux2
                df2['Mean prob'] = aux3
                df2['Lowest prob'] = aux4
                chains_probs = pd.concat([chains_probs, df2])

    # Formatting
    writer = pd.ExcelWriter('Chains_probabilities.xlsx', engine='xlsxwriter')
    chains_probs.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('D:E', None, format1)

    # Auto-adjust column widths in excel file
    for column in chains_probs:
        column_width = max(chains_probs[column].astype(str).map(len).max(), len(column))
        col_idx = chains_probs.columns.get_loc(column)
        writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_width)
    writer.save()

# ================================= Plot ================================== #

# Legends and labels
x_labels = list(np.unique(AATs_unique))
x_pos = list(range(0, len(x_labels)))
y_pos = list(range(0, len(test_set)))
legend_labels = x_labels.copy()

# Remove extra legends
for i in reversed(range(0, probs.shape[1])):
    if (probs[:, i] == 0).all():
        legend_labels.pop(i)

# Create lists for plotting
x_vals = []
y_vals = []
p_vals = []
for i in y_pos:
    for j in x_pos:
        if probs[i, j] != 0:
            x_vals.append(x_pos[j])
            y_vals.append(y_pos[i])
            p_vals.append(probs[i, j])

# Create Data Frame using lists
plot_data = pd.DataFrame(columns=['label', 'residue', 'probability'])
plot_data['Label'] = x_vals
plot_data['Residue'] = y_vals
plot_data['Probability'] = p_vals

# Create stable color palette (always same colors for each AAT)
dict_col = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,
            'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13,
            'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
palette = sns.color_palette(cc.glasbey, n_colors=20)
col_nums = [dict_col[x] for x in legend_labels]
col_list = []
for i in col_nums:
    col_list.append(palette[i])

# Create figure
h = sns.relplot(x="Label", y="Residue", hue="Label", size="Probability",
                sizes=(40, 300), alpha=.5, palette=col_list, data=plot_data, aspect=0.3)

# Formatting and save
h.ax.margins(x=0.05, y=0.02)
h.despine(top=False, right=False)
plt.ylabel("Spin system", size=20)
plt.yticks(y_pos, SSN, fontsize=8)
plt.xlabel("LDA classification", size=20)
plt.xticks(x_pos, x_labels, fontsize=10, rotation=60)
plt.grid(axis='x', color='k', linestyle='-', linewidth=0.2)
plt.grid(axis='y', color='k', linestyle=':', linewidth=0.2)
sns.move_legend(h, "upper right", bbox_to_anchor=(0.7, 0.8))
for t, l in zip(h._legend.texts, ['Labels']+legend_labels):
    t.set_text(l)
h.fig.set_dpi(100)
h.fig.set_figheight(15)
h.fig.set_figwidth(25)
h.savefig('Probabilities.svg', dpi=100)
plt.show()

