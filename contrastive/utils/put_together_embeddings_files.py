
import os
import shutil
import logging
import argparse
import glob

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
    print(f"Found {len(model_paths)} model(s) in {path_champollion}")

    if not model_paths:
        print("WARNING: No models found. Check that path_champollion contains folders with .hydra/config.yaml")
        return

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Normalize path_champollion for consistent splitting
    path_champollion_normalized = path_champollion.rstrip('/')

    copied_count = 0
    for model_path in model_paths:
        file_input_name = f"{model_path}/{embeddings_subpath}"
        # Extract region and model from relative path within path_champollion
        relative_path = model_path.replace(path_champollion_normalized + '/', '')
        parts = relative_path.split('/')
        region = parts[0] if parts else 'unknown'
        model = '--'.join(parts[1:]).replace("_", "--") if len(parts) > 1 else 'model'
        file_output_name = f"{output_path}/{region}_{model}_embeddings.csv"

        if not os.path.exists(file_input_name):
            print(f"  Skipping (not found): {file_input_name}")
            continue

        try:
            shutil.copyfile(file_input_name, file_output_name)
            copied_count += 1
            print(f"  Copied: {region}_{model}_embeddings.csv")
        except OSError as e:
            print(f"  Error copying {file_input_name}: {e}")

    nb_csv = len(glob.glob(f"{output_path}/*.csv"))
    print(f"\n{copied_count} file(s) copied, {nb_csv} CSV(s) in {output_path}")
    if nb_csv > 0:
        print("DONE!")
    else:
        print(f"WARNING: no embeddings in output directory {output_path}")

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



