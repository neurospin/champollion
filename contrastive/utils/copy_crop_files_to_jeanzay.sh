input_folder="/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm"
ssh_connection="umy22uu@jean-zay.idris.fr"
output_folder="/lustre/fswork/projects/rech/tgu/umy22uu/Runs/70_self-supervised_two-regions/Input/UkBioBank/crops/2mm"

regions="F.Coll.-S.Rh."

for region in $regions;
do 
    input_region=$input_folder/$region/mask
    output_region="$output_folder/$region/mask"
    echo "$region:"
    ssh $ssh_connection "mkdir -p $output_region"
    rsync -aP $input_region/*.csv "$ssh_connection:$output_region/"
    rsync -aP $input_region/*.npy "$ssh_connection:$output_region/"
    ls $output_region
done