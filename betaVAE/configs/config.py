from datetime import datetime
now = datetime.now()


class Config:

    def __init__(self):
        self.batch_size = 16
        self.nb_epoch = 150
        self.kl = 2
        self.n = 10 #75
        self.lr = 2e-4 #5e-4 ?

        sulcus="C.S"
        side='R'

        #self.in_shape = (1, 28, 56, 48) # input size with padding #22, 49, 46 without padding # 28 instead of 24 ?? Rsts
        #self.in_shape = (1, 28, 56, 56) # input size with padding #25, 55, 50 without padding # 28 instead of 24 ?? Lsts
        #self.in_shape = (1, 28, 64, 56) # input size with padding #21, 60, 51 without padding # 28 instead of 24 ?? Rcingulate
        #self.in_shape = (1, 28, 64, 56) # input size with padding #22, 62, 49 without padding # 28 instead of 24 ?? Lcingulate
        self.in_shape = (1, 48, 48, 48) #input size #43, 41, 44 without padding R CS
        #self.in_shape = (1, 48, 48, 48) #input size #43, 43, 44 without padding L CS

        #self.in_shape = (1, 20, 40, 40) # input size with padding #17, 40, 38 without padding

        #self.data_dir = "/path/to/data/directory"
        #self.subject_dir = "/path/to/list_of_subjects"
        #self.save_dir = "/path/to/saving/directory"
        self.save_dir = f"/neurospin/dico/jlaval/Runs/02_STS_babies/Program/Output/BetaVAE/{sulcus}/{side}/{now:%Y-%m-%d}/{now:%H-%M-%S}/"

        #self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/S.T.s.baby/mask/Rskeleton_str.pkl" #need to change R to L
        #self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE.baby/mask/Lskeleton_str.pkl" #need to change R to L
        self.data_dir = f"/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/babies/{sulcus}/mask/{side}skeleton_str.pkl" #need to change R to L

        #self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/CINGULATE/mask"

        #self.subject_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/S.T.s.baby/mask/Rskeleton_subject_int.csv"
        #self.subject_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/S.T.s.baby/mask/Lskeleton_subject_int.csv" #same as R file...
        #self.subject_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE.baby/mask/Lskeleton_subject_int.csv"
        self.subject_dir = f"/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/babies/{sulcus}/mask/{side}skeleton_subject_int.csv"
        #self.subject_dir = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE.baby/mask/Lskeleton_subject_int.csv"
        
        #self.subject_dir = "/neurospin/dico/data/deep_folding/papers/miccai2023/Input/datasets/hcp/crops/2mm/CINGULATE/mask/Rskeleton_subject_full.csv"
        # used for embeddings generation:
        #self.acc_subjects_dir = "/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/crops/2mm/CINGULATE/mask"
        #self.test_model_dir = "/neurospin/dico/agaudin/Runs/09_new_repo/Output/2023-04-05/11-53-51"


# class Config:

#     def __init__(self):
#         self.batch_size = 64
#         self.nb_epoch = 500 #300
#         self.kl = 2
#         self.n = 10
#         self.lr = 2e-4
#         self.in_shape = (1, 20, 40, 40) # input size with padding
#         self.save_dir = f"/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/n_{self.n}/"
#         self.data_dir = "/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/CINGULATE/mask/"
#         self.subject_dir = "/neurospin/dico/data/deep_folding/papers/midl2022/HCP_half_1bis.csv"
#         self.acc_subjects_dir = "/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/crops/2mm/CINGULATE/mask"
#         self.test_model_dir = f"/neurospin/dico/lguillon/collab_joel_aymeric_cingulate/n_{self.n}/#5"
