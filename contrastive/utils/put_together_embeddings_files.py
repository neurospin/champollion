
import os
import shutil
import logging
import argparse

# path_champollion = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
# embeddings_subpath = "testxx_random_embeddings/full_embeddings.csv"
# output_path = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/embeddings/TESTXX_embeddings"


def is_it_a_file(sub_dir):
    if os.path.isdir(sub_dir):
        return False
    else:
        logging.debug(f"{sub_dir} is a file. Continue.")
        return True
    

def is_folder_a_model(sub_dir):
    if os.path.exists(sub_dir+'/.hydra/config.yaml'):
        return True
    else:
        logging.debug(f"\n{sub_dir} not associated to a model. Continue")
        return False


def get_model_paths(dir_path, result = None):
    """Recursively gets all models included in dir_path"""
    if result is None:  # create a new result if no intermediate was given
        result = [] 
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if is_it_a_file(sub_dir):
            pass
        elif not is_folder_a_model(sub_dir):
            result.extend(get_model_paths(sub_dir))
        else:
            result.append(sub_dir)
    return result


def put_together_embeddings_files(embeddings_subpath, output_path, path_champollion):

    model_paths = get_model_paths(path_champollion)

    model_paths

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for model_path in model_paths:
        file_input_name = f"{model_path}/{embeddings_subpath}"
        region = model_path.split('Champollion_V1_after_ablation/')[1].split('/')[0]
        model = model_path.split(region+'/')[1].replace("/", "--").replace("_", "--")
        file_output_name = f"{output_path}/{region}_{model}_embeddings.csv"
        try:
            shutil.copyfile(file_input_name, file_output_name)
        except OSError as e:
            msg = str(e)
            if "] " in msg:
                msg = msg.split("] ", 1)[1]
            print(f"The following warning can be normal if you have not generated this region in your dataset: {msg}")

# path_champollion = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation"
# embeddings_subpath = "testxx_random_embeddings/full_embeddings.csv"
# output_path = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/embeddings/TESTXX_embeddings"

def main():
    """Main function to put together the embeddings"""
    parser = argparse.ArgumentParser(
        description="Put together the embeddings (previously generated in folder models)"
    )

    # Required arguments
    parser.add_argument(
        "--embeddings_subpath",
        type=str,
        help="Subpath where to find ambeddings in each model folder."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for embeddings."
    )
    parser.add_argument(
        "--path_models",
        type=str,
        default="/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation",
        help="Path where all models lie."
    )

    args = parser.parse_args()
    print(args)

    put_together_embeddings_files(args.embeddings_subpath, args.output_path, args.path_models)

if __name__ == "__main__":
    main()



