import numpy as sns
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from sklearn.manifold import Isomap
import umap
from denmarf import DensityEstimate
from sklearn.neighbors import KDTree



model_path = '/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-05-28/23-11-30/'
model_path = "/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-06-05/17-32-49/"
#Reconstruction error results
Train_subjects = pd.read_csv(model_path+'Train_subjects.csv').ID.values.tolist()
Test_subjects = pd.read_csv(model_path+'Test_subjects.csv').ID.values.tolist()
Interrupted_subjects = pd.read_csv(model_path+'Interrupted_CS_subjects.csv').ID.values.tolist()

Reconstruction_error = pd.read_csv(model_path+'Reconstruction_error.csv')
Reconstruction_error.columns = ['ID','Recon']

Reconstruction_train = Reconstruction_error[Reconstruction_error['ID'].isin(Train_subjects)]
Reconstruction_interrupted = Reconstruction_error[Reconstruction_error['ID'].isin(Interrupted_subjects)]
Reconstruction_test = Reconstruction_error[Reconstruction_error['ID'].isin(Test_subjects)]


print('Recon train',Reconstruction_train)
print('Recon interrupted',Reconstruction_test)
print('Recon test',Reconstruction_interrupted)

plt.figure()
sns.histplot(data = Reconstruction_train,x='Recon',element="step",label='Train',color='blue',stat='density',kde=True)
sns.histplot(data = Reconstruction_interrupted,x='Recon',element="step",label='Int. CS',color='red',stat='density',kde=True)
sns.histplot(data = Reconstruction_test,x='Recon',element="step",label='Test',color='orange',stat='density',kde=True,alpha=0.1)
plt.legend()
plt.savefig(model_path+'Reconstruction_error.png',dpi=300)

U1, p = mannwhitneyu(Reconstruction_train.Recon.values, Reconstruction_interrupted.Recon.values)
print(U1,p)

U1, p = mannwhitneyu(Reconstruction_test.Recon.values, Reconstruction_interrupted.Recon.values)
print(U1,p)

U1, p = mannwhitneyu(Reconstruction_test.Recon.values, Reconstruction_train.Recon.values)
print(U1,p)

print(Reconstruction_test[Reconstruction_test.Recon.gt(5000)])

mu = np.mean(Reconstruction_train.Recon.values)
std = np.std(Reconstruction_train.Recon.values)
Subject_high_recon = Reconstruction_train.loc[Reconstruction_train['Recon'] > mu + 2*std]

print(Subject_high_recon,'mu',mu,'std',std)
Subject_high_recon_= []
for sub in Subject_high_recon.ID.values:
    Subject_high_recon_.append('sub-'+str(sub))
print(Subject_high_recon_,len(Subject_high_recon_))

#SVM
print('Support vector machine')
Embeddings = pd.read_csv(model_path+'Embeddings.csv')
columns = ['ID']
for i in range(0,75):
    columns.append('dim'+str(i))
Embeddings.columns = columns
print(Embeddings)

Embeddings_interrupted = Embeddings[Embeddings['ID'].isin(Interrupted_subjects)]
Embeddings_test = Embeddings[Embeddings['ID'].isin(Test_subjects)]

labels_0 = [0]*207
Embeddings_test.insert(loc=76, column='labels', value=labels_0)
print('Embedding test',Embeddings_test)

labels_1 = [1]*207
Embeddings_interrupted.insert(loc=76, column='labels', value=labels_1)
print('Embedding interrupted',Embeddings_interrupted)

Full_dataset = pd.concat([Embeddings_test,Embeddings_interrupted], axis=0)
print('Full dataset',Full_dataset)

#ROC Curve
X = Full_dataset.drop(columns=['ID','labels']).to_numpy()
y = Full_dataset['labels'].to_numpy()

kf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure()
i=1
for train_index, test_index in kf.split(X, y):
    print('Fold')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('train and test indices',len(X_train),len(X_test),len(y_train),len(y_test))
    model = svm.SVC(probability=True, kernel='linear', random_state=42,C=0.01)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]
    print('yprob ytest shape',y_prob.shape,y_test.shape)
    fpr, tpr, threshold = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    aucs.append(roc_auc)
    print('score' , model.score(X_test,y_test))

    # Interpolación para tener todos los TPR en el mismo eje
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    #if i == 3:

    with open(model_path+'SVM_fold_'+str(i)+'.pkl','wb') as f:
        pickle.dump(model,f)

    plt.plot(fpr, tpr, alpha=0.5, label=f'ROC Fold {i} (AUC = {roc_auc:.3f})',lw=2)
    i+=1

# Curva ROC promedio
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
#print(mean_tpr,mean_fpr,len(mean_tpr),len(mean_fpr))
mean_auc = auc(np.asarray(mean_fpr), np.asarray(mean_tpr))

plt.plot(mean_fpr, mean_tpr, color='black',
         label=f'Mean ROC (AUC = {mean_auc:.2f})',lw=3)

# Sombra ± 1 desviación estándar
std_tpr = np.std(tprs, axis=0)
tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='black', alpha=0.2, label='± 1 std. dev.')

# Línea aleatoria
plt.plot([0, 1], [0, 1], 'r--', lw=1,label='Random')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve Stratified K-Cross Fold Validation')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(model_path+'SVM.png',dpi=300)

#Get Interrupted subjects based on SVM probability 
print('SVM with highest AUC is ',str(np.argmax(aucs)+1), aucs[np.argmax(aucs)])
with open(model_path+'SVM_fold_'+str(np.argmax(aucs)+1)+'.pkl', 'rb') as f:
    model = pickle.load(f)

X = Embeddings.drop(columns=['ID']).to_numpy()
print(X.shape)
y_prob = model.predict_proba(X)[:,1]
subjects = Embeddings.ID.values
subjects_str = ['sub-'+str(sub) for sub in subjects]
id_sub = {'ID':subjects_str,'prob':y_prob}
prob_df = pd.DataFrame(id_sub)
prob_df_sorted = prob_df.sort_values(by='prob')

#UMAP Unsupervised
reducer = umap.UMAP(random_state=42)
X = Embeddings.drop(columns=['ID']).to_numpy()
X_umap = reducer.fit_transform(X)

labels = []
All_subjects = Embeddings.ID.values
for sub in All_subjects:
    if sub in Interrupted_subjects:
        labels.append('Interrupted')
    elif sub in Test_subjects:
        labels.append('Test')
    else:
        labels.append('Train')

#Coloring each class
tmp = dict(zip(All_subjects, zip(X_umap[:, 0], X_umap[:, 1], labels)))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','Class']
print(tmp_df)
plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2',size = 'Class',sizes = [1,6,6] , hue = 'Class',s=1,palette = ['blue','red','orange'])
plt.savefig(model_path+'UMAP.png',dpi=300)

#Coloring according to SVM probability
tmp = dict(zip(All_subjects, zip(X_umap[:, 0], X_umap[:, 1], y_prob)))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','prob']
print(tmp_df)
plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2', palette='cool', hue = 'prob',s=2)
plt.savefig(model_path+'UMAP_prob.png',dpi=300)

#Coloring according to reconstruction error
tmp = dict(zip(All_subjects, zip(X_umap[:, 0], X_umap[:, 1], Reconstruction_error.Recon.values)))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','ReconError']
print(tmp_df)
tmp_df['ReconError_bin'] = pd.cut(tmp_df.ReconError.values, bins=[2000, 3000, 4000, 5000])
plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2', palette='cool', hue=tmp_df['ReconError_bin'],s=1)
plt.savefig(model_path+'UMAP_recon.png',dpi=300)


#UMAP Supervised
Embeddings = pd.read_csv(model_path+'Embeddings.csv')
columns = ['ID']
for i in range(0,75):
    columns.append('dim'+str(i))
Embeddings.columns = columns
print(Embeddings)

Embeddings_train = Embeddings[Embeddings['ID'].isin(Train_subjects)]
Embeddings_interrupted = Embeddings[Embeddings['ID'].isin(Interrupted_subjects)]
Embeddings_test = Embeddings[Embeddings['ID'].isin(Test_subjects)]

labels_0 = [0]*207
labels_0_str = ['Test']*207
Embeddings_test.insert(loc=76, column='labels', value=labels_0)
print('Embedding test',Embeddings_test)

labels_1 = [1]*207
labels_1_str = ['CS Int']*207
Embeddings_interrupted.insert(loc=76, column='labels', value=labels_1)
print('Embedding interrupted',Embeddings_interrupted)

Full_dataset = pd.concat([Embeddings_test,Embeddings_interrupted], axis=0)
print('Full_dataset UMAP Supervised',Full_dataset)

#UMAP based on test and CS Int labels
X = Full_dataset.drop(columns=['ID','labels']).to_numpy()
y = Full_dataset['labels'].to_numpy()
mapper = umap.UMAP(random_state=42).fit(X, y)

labels_neg_1 = [-1]*len(Train_subjects)
labels_2_str = ['Train']*len(Train_subjects)
Embeddings_train.insert(loc=76, column='labels', value=labels_neg_1)

#Embedding for all the subjects, ordered test - int - train
Embeddings_test_int_train = pd.concat([Embeddings_test,Embeddings_interrupted,Embeddings_train], axis=0)
print('Embeddings_test_int_train',Embeddings_test_int_train)

X = Embeddings_test_int_train.drop(columns=['ID','labels']).to_numpy()
embedding = mapper.transform(X)

#Coloring according class
tmp = dict(zip(Embeddings_test_int_train.ID.values.tolist(), zip(embedding[:, 0], embedding[:, 1], labels_0_str + labels_1_str + labels_2_str)))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','Class']
print(tmp_df)
plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2',size = 'Class',sizes = [6,6,1] , hue = 'Class',s=1,palette = ['orange','red','blue'])
plt.savefig(model_path+'UMAP_Semi_supervised.png',dpi=300)

#Coloring according probability
y_prob = model.predict_proba(X)[:,1]
tmp = dict(zip(Embeddings_test_int_train.ID.values.tolist(), zip(embedding[:, 0], embedding[:, 1], y_prob)))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','prob']
print(tmp_df)
plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2', palette='cool', hue = 'prob',s=2)
plt.savefig(model_path+'UMAP_Semi_supervised_prob.png',dpi=300)

#Coloring according reconstrcution error

# Sort using the custom order
Reconstruction_error['ID'] = pd.Categorical(Reconstruction_error['ID'], categories=Embeddings_test_int_train.ID.values.tolist(), ordered=True)
df_sorted = Reconstruction_error.sort_values('ID')
print(df_sorted)
tmp = dict(zip(Embeddings_test_int_train.ID.values.tolist(), zip(embedding[:, 0], embedding[:, 1], df_sorted.Recon.values.tolist())))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','Recon']
print(tmp_df)
tmp_df['ReconError_bin'] = pd.cut(df_sorted.Recon.values, bins=[2000, 3000, 4000, 5000])
print(tmp_df)
plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2', palette='cool', hue=tmp_df['ReconError_bin'],s=2)
plt.savefig(model_path+'UMAP_Semi_supervised_recon.png',dpi=300)

# Access nearest neighbor distances and indices
tree = KDTree(embedding)              
dist, ind = tree.query(embedding, k=10)
print(dist,ind)
knn_indices = ind[207:207*2]

all_subs_to_check = []
Interrupted_subjects = [str(sub) for sub in Interrupted_subjects]
print('int subject',Interrupted_subjects)
for sub in range(0,knn_indices.shape[0]):
    indices = knn_indices[sub,:]
    neighbors_subjects = np.asarray(Embeddings_test_int_train.ID.values)[indices].tolist()
    #print('-------------',sub)
    #print(len(neighbors_subjects))
    #print('neighbors',neighbors_subjects,'int subject',Interrupted_subjects)
    c= 0
    for n_sub in neighbors_subjects:
        n_sub = str(n_sub)
        if n_sub not in Interrupted_subjects:
            all_subs_to_check.append(n_sub)
            c+=1
    #print(c)



print('df before reset index',tmp_df)
tmp_df['ID'] = tmp_df.index
print('df after reset index',tmp_df)
subjects = tmp_df.ID.values.tolist()
neighbor = []

for sub in subjects:
    sub = str(sub)
    if sub in all_subs_to_check:
        neighbor.append(1)
    elif sub in Interrupted_subjects:
        neighbor.append(2)
    else:
        neighbor.append(0)

all_subs_to_check = ['sub-'+str(sub) for sub in all_subs_to_check]
#print('subs to check',all_subs_to_check)

tmp = dict(zip(subjects, zip(embedding[:, 0], embedding[:, 1], neighbor)))
tmp_df = pd.DataFrame.from_dict(tmp)
tmp_df = tmp_df.T
tmp_df.columns = ['DIM1','DIM2','Neighbor']

plt.figure()
sns.scatterplot(data = tmp_df,x='DIM1',y='DIM2', palette=['blue','purple','red'], hue='Neighbor',s=2)
plt.savefig(model_path+'UMAP_Semi_supervised_rneighbors.png',dpi=300)
