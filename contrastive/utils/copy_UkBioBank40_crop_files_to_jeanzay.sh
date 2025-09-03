input_folder="/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm"
ssh_connection="umy22uu@jean-zay.idris.fr"
output_folder="/lustre/fswork/projects/rech/tgu/umy22uu/Runs/70_self-supervised_two-regions/Input/UkBioBank40/crops/2mm"

# regions="CINGULATE. F.C.L.p.-S.GSM. F.C.L.p.-subsc.-F.C.L.a.-INSULA. "\
# "F.C.M.post.-S.p.C. F.Coll.-S.Rh. F.I.P. F.P.O.-S.Cu.-Sc.Cal. "\
# "fronto-parietal_medial_face. Lobule_parietal_sup. OCCIPITAL "\
# "ORBITAL Sc.Cal.-S.Li. S.C.-S.Pe.C. S.C.-S.Po.C. S.C.-sylv. "\
# "S.F.inf.-BROCA-S.Pe.C.inf. S.F.inter.-S.F.sup. S.F.int.-F.C.M.ant. "\
# "S.F.int.-S.R. S.F.marginal-S.F.inf.ant. S.F.median-S.F.pol.tr.-S.F.sup. "\
# "S.Or. S.Or.-S.Olf. S.Pe.C. S.Po.C. S.s.P.-S.Pa.int. S.T.i.-S.O.T.lat. "\
# "S.T.i.-S.T.s.-S.T.pol. S.T.s. S.T.s.br. S.T.s.-S.GSM."

regions="LARGE_CINGULATE."

echo "$regions"

for region in $regions;
do 
    input_region=$input_folder/$region/mask
    output_region="$output_folder/$region/mask"
    echo "$region:"
    ssh $ssh_connection "mkdir -p $output_region"
    rsync -aP $input_region/*.csv "$ssh_connection:$output_region/"
    rsync -aP $input_region/*.npy "$ssh_connection:$output_region/"
    ssh $ssh_connection "ls $output_region"
done
