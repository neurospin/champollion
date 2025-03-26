import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import auc, roc_curve, roc_auc_score, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

save_dir = '~/Documents/LinearModels/'

## orbital

crops_dir = '/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/ORBITAL/mask'
label='Left_OFC'
side = 'L'
labels = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/hcp/hcp_OFC_labels_from_0.csv', usecols=['Subject', label])
splits_dir = '/neurospin/dico/data/deep_folding/current/datasets/orbital_patterns/Troiani/Left'

##

skels = np.load(os.path.join(crops_dir, f'{side}skeleton.npy'))
skel_subs = pd.read_csv(os.path.join(crops_dir, f'{side}skeleton_subject.csv'))
skels = skels.reshape(skels.shape[0], np.prod(skels.shape[1:]))
skels = skels.astype(bool)

train_subs = pd.read_csv(os.path.join(splits_dir, 'train_split.csv'), names=['Subject'])
val_subs = pd.read_csv(os.path.join(splits_dir, 'val_split.csv'), names=['Subject'])
test_subs = pd.read_csv(os.path.join(splits_dir, 'test_split.csv'), names=['Subject'])

proportions = np.unique(labels[label], return_counts=True)
proportions = proportions[1] / np.sum(proportions[1])

train = skel_subs.loc[skel_subs['Subject'].isin(train_subs['Subject'])]
idxs_train = train.index.tolist()
Y_train = pd.merge(train, labels)[label].to_numpy().reshape(-1,1)
train_skels= skels[idxs_train]

val = skel_subs.loc[skel_subs['Subject'].isin(val_subs['Subject'])]
idxs_val = val.index.tolist()
Y_val = pd.merge(val, labels)[label].to_numpy().reshape(-1,1)
val_skels= skels[idxs_val]

test = skel_subs.loc[skel_subs['Subject'].isin(test_subs['Subject'])]
idxs_test = test.index.tolist()
Y_test = pd.merge(test, labels)[label].to_numpy().reshape(-1,1)
test_skels= skels[idxs_test]

# define train val and their indexes for gridsearch cv
X_train_val = np.vstack((train_skels, val_skels))
Y_train_val = np.vstack((Y_train, Y_val))
split_index = np.full(len(X_train_val), -1)  # All initially train
split_index[len(train_skels):] = 0  # Mark validation samples
predefined_split = PredefinedSplit(test_fold=split_index)

# logistic
parameters={'l1_ratio': np.linspace(0,1,11), 'C': [10**k for k in range(-3,3)]}
model = LogisticRegression(solver='saga', penalty='elasticnet',
                           max_iter=1000000, random_state=0)

grid_search = GridSearchCV(model, parameters, cv=predefined_split, scoring='roc_auc_ovr_weighted', refit=True, n_jobs=-1)
grid_search.fit(X_train_val, Y_train_val)
# Print best parameters
print("Best parameters:", grid_search.best_params_)
# Evaluate on the test set
best_model = grid_search.best_estimator_

y_test_pred = best_model.predict_proba(test_skels)
score = roc_auc_score(Y_test, y_test_pred, multi_class='ovr', average='weighted') ## WON'T WORK WITH SINGLE CLASS
print(score)

with open(os.path.join(save_dir,"SOr.txt"),"w") as file:
      file.write(f'Score : {score}, Best_parameters : {grid_search.best_params_}')
