import argparse
import numpy as np
import pandas as pd
import torch
import glob
import omegaconf
import os
import ast
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from champollion.backbones.convnet import ConvNet
from champollion.backbones.resnet import ResNet, BasicBlock

### Small .py to generate embeddings using Champollionv1 ###

def load_model(model_path, in_shape):
    weights_path = glob.glob(model_path + '/logs/lightning_logs/version_0/checkpoints/*.ckpt')[0]
    config_path = model_path + '/.hydra/config.yaml'
    checkpoint = torch.load(weights_path, map_location='cpu')
    encoder_state_dict = {k.replace('backbones.0.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('backbones.0.encoder.')}
    config = omegaconf.OmegaConf.load(config_path)
    try:
        input_size = ast.literal_eval(config.data[0].input_size)
        assert in_shape == input_size
    except Exception as e:
        raise ValueError(
            f"Data shape {in_shape} is not equal to the model input shape {input_size}."
        )
    
    if config.backbone_name=='convnet':
        kwargs = {k: config[k] for k in ('encoder_depth', 'block_depth', 'filters', 'initial_kernel_size',
                                         'initial_stride', 'max_pool', 'backbone_output_size', 'adaptive_pooling', 'drop_rate')}
        kwargs['num_representation_features']=kwargs.pop('backbone_output_size')
        model = ConvNet(in_channels=1, in_shape=in_shape, **kwargs)
    elif config.backbone_name=='resnet':
        kwargs = {k: config[k] for k in ('layers', 'channels', 'backbone_output_size', 'zero_init_residual',
                                         'drop_rate', 'initial_kernel_size', 'initial_stride', 'adaptive_pooling')}
        kwargs['num_classes']=kwargs.pop('backbone_output_size')
        kwargs['dropout_rate']=kwargs.pop('drop_rate')
        model = ResNet(in_channels=1, block=BasicBlock, out_block=None, **kwargs)
    else:
        raise ValueError("The architecture is not handled.")
    
    try:
        model.load_state_dict(encoder_state_dict)
    except Exception as e:
        raise ValueError(
            f"Failed to load weights: {e}\n"
            "This likely means the model parameters (encoder_depth, block_depth, filters, etc.) "
            "do not match the saved weights. Please check your model configuration."
        )
    model.eval()
    print('Model weights loaded')
    return model


def load_data(skels_path, batch_size=32):
    arrays = np.load(skels_path) # (N, H, W, D, C)
    arrays = (arrays != 0).astype(np.float32)
    arrays = torch.tensor(arrays).permute(0, 4, 1, 2, 3).contiguous()  # (N, C, H, W, D)
    in_shape = tuple(arrays.shape[1:]) # (C, H, W, D)
    return DataLoader(TensorDataset(arrays), batch_size=batch_size, shuffle=False, pin_memory=True), in_shape


def load_subjects(subjects_path):
    """Read the subjects CSV and return the 'Subject' column as an array.

    Raises a ValueError naming the file and its actual columns if the
    expected 'Subject' column is absent, instead of a bare KeyError.
    """
    df = pd.read_csv(subjects_path)
    if 'Subject' not in df.columns:
        raise ValueError(
            f"Subjects file {subjects_path} has no 'Subject' column; "
            f"columns read: {list(df.columns)}"
        )
    return df['Subject'].values


def generate_embeddings(model, dataloader, device):
    embeddings = []
    total_samples = len(dataloader.dataset)
    with torch.no_grad(), tqdm(total=total_samples, desc='Generating embeddings', unit='sample') as pbar:
        for (batch,) in dataloader:
            batch = batch.to(device)
            out = model(batch).cpu().numpy()
            embeddings.append(out)
            pbar.update(out.shape[0])
    return np.concatenate(embeddings, axis=0)


def main(args):

    # ── 1. Loading Data  ───────────────────
    dataloader, in_shape = load_data(args.skels_path)
    subjects = load_subjects(args.subjects_path)

    n_samples = len(dataloader.dataset)
    if len(subjects) != n_samples:
        raise ValueError(
            f"Subject count mismatch: {len(subjects)} subjects in "
            f"{args.subjects_path} but {n_samples} samples on the first axis "
            f"of {args.skels_path}; refusing to label embeddings by position."
        )

    # ── 2. Loading Model ───────────────────
    model = load_model(args.model_path, in_shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ── 3. Computing embeddings  ───────────
    embeddings = generate_embeddings(model, dataloader, device)

    # ── 4. Saving results  ─────────────────
    df = pd.DataFrame(embeddings, columns=[f'dim{i}' for i in range(1,embeddings.shape[1]+1)])
    df.insert(0, 'ID', subjects)
    os.makedirs(os.path.dirname(args.saving_path), exist_ok=True)
    df.to_csv(args.saving_path, index=False)
    print(f'Saved embeddings to {args.saving_path} — shape: {df.shape}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python file to generate Champollion embeddings for a single region using only the model weights and the crops.')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the .pt file with the region-specific model weights of Champollion.')
    parser.add_argument('-sk', '--skels_path', type=str, required=True, help='Path to the skeletons/crops of one region (shape must be (N,H,W,D,C)).')
    parser.add_argument('-i', '--subjects_path', type=str, required=True, help='Path to the corresponding subjects (csv files with one column named "Subject").')
    parser.add_argument('-s', '--saving_path', type=str, required=True, help='Path to where the embeddings will be saved.')
    args = parser.parse_args()
    main(args)