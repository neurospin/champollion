import os
import pandas as pd
from tqdm import tqdm
from beta_vae import *
from preprocess import SkeletonDataset


if torch.cuda.is_available():
    device = "cuda:0"
    print('GPU available')

output_dir = '/neurospin/dico/jlaval/Runs/02_STS_babies/Program/Output/BetaVAE/'

dimensions = 10 #10 or 75

#sulcus='S.T.s'
#side='R'
#model_name = '2023-12-11/14-10-52/'
#model_name= "2023-12-07/14-42-29/"
#model_name="2023-12-18/13-54-42/"
#model_name="2023-12-18/16-56-49/"

#side='L'
#model_name="2024-01-03/11-29-07/"
#model_name="/2024-01-03/15-24-21"

###############

sulcus='CINGULATE'
#side='L'
#model_name='/2024-01-04/14-24-19'
side='R'
model_name='/2024-01-04/14-00-48'

#crop_size=(1, 28, 56, 48)
#crop_size=(1, 28, 56, 56)
crop_size=(1, 28, 64, 56)

datasets = ['ukb', 'dHCP']

for dataset in datasets:

    model_dir = output_dir + model_name + '/checkpoint.pt'

    if dataset=='ukb':
        data_dir = f'/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/{sulcus}.baby/mask/{side}skeleton.pkl'
        subject_dir = f"/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/S.T.s.baby/mask/Rskeleton_subject_int.csv" #always the same

    elif dataset=='dHCP':
        data_dir = f'/neurospin/dico/data/deep_folding/current/datasets/dHCP_374_subjects/crops/2mm/{sulcus}.baby/mask/{side}skeleton.pkl'
        subject_dir = f'/neurospin/dico/data/deep_folding/current/datasets/dHCP_374_subjects/crops/2mm/S.T.s.baby/mask/Rskeleton_subject_no_header.csv' #always the same


    random_sampling=False # whether there is random sampling for the reconstruction

    #model_dir = '/neurospin/dico/lguillon/distmap/checkpoint.pt'
    model = VAE(crop_size, dimensions, depth=3)
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)

    train_list = pd.read_csv(subject_dir, header=None, usecols=[0],
                                names=['subjects'])
    train_list['subjects'] = train_list['subjects'].astype('str')

    tmp = pd.read_pickle(data_dir) #.T, but transposed manually for ukb and dHCP

    subjects = tmp.columns.tolist()
    if subjects[0][:4]=='sub-': # remove sub
        subjects = [subject[4:] for subject in subjects]
        tmp.columns = subjects
        tmp = tmp.T

    tmp.index.astype('str')
    #tmp['subjects'] = [re.search('(\d{6})', tmp.index[k]).group(0) for k in range(
    #                    len(tmp))]
    tmp['subjects'] = [tmp.index[k] for k in range(len(tmp))]
    tmp = tmp.merge(train_list, left_on = 'subjects', right_on='subjects', how='right')
    filenames = list(train_list['subjects'])

    #n=5000
    #tmp = tmp.iloc[:n]
    #filenames = filenames[:n]
    subset = SkeletonDataset(dataframe=tmp, filenames=filenames)


    loader =  torch.utils.data.DataLoader(
                            subset,
                            batch_size=1,
                            num_workers=1,
                            shuffle=False)

    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(class_weights, reduction='sum')
    tester = ModelTester(model=model, dico_set_loaders={'train': loader},
                        loss_func=criterion, kl_weight=2,
                        n_latent=dimensions, depth=3,
                        sampling=random_sampling)

    results, outputs = tester.test()
    encoded_all = {loader_name:[results[loader_name][k][1] for k in results[loader_name].keys()] for loader_name in {'train': loader}.keys()}
    losses_all = {loader_name:[int(results[loader_name][k][0].cpu().detach().numpy()) for k in results[loader_name].keys()] for loader_name in {'train': loader}.keys()}
    recon_all = {loader_name:[int(results[loader_name][k][2].cpu().detach().numpy()) for k in results[loader_name].keys()] for loader_name in {'train': loader}.keys()}
    var_all = {loader_name:[results[loader_name][k][3] for k in results[loader_name].keys()] for loader_name in {'train': loader}.keys()}
    #input_all = {loader_name:[results[loader_name][k][3].cpu().detach().numpy() for k in results[loader_name].keys()] for loader_name in {'train': loader}.keys()}

    df_encoded_all = pd.DataFrame()
    df_encoded_all['latent'] = encoded_all['train']
    df_encoded_all['loss'] = losses_all['train']
    df_encoded_all['recon'] = recon_all['train']
    df_encoded_all['var'] = var_all['train']
    #df_encoded_all['input'] = input_all['train']
    df_encoded_all['Group'] = ['train' for k in range(len(filenames))] 
    df_encoded_all['sub'] = list(filenames)

    if random_sampling:
        df_encoded_all.to_csv(output_dir + model_name + f'/{dataset}_embeddings.csv',
                        index=False)
        np.save(output_dir+model_name+f'/{dataset}_{side}skeleton_reconstructed.npy',
                outputs)
    else:
        df_encoded_all.to_csv(output_dir + model_name + f'/{dataset}_embeddings_no_sampling.csv',
                        index=False)
        np.save(output_dir+model_name+f'/{dataset}_{side}skeleton_reconstructed_no_sampling.npy',
                outputs)