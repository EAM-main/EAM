import argparse
import random
import numpy as np
import torch
from Run import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fakett', help='fakett/fakesv')
parser.add_argument('--mode', default='train', help='train/inference_test')
parser.add_argument('--epoches', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--gpu', default='0')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--inference_ckp', help='input path of inference checkpoint when mode is inference')
parser.add_argument('--path_ckp', default='./checkpoints/')
parser.add_argument('--path_tb', default='./tensorboard/')
args = parser.parse_args()

available_gpus = torch.cuda.device_count()
print(f'Available GPUs: {available_gpus}')
if available_gpus > 0:
    gpu_id = int(args.gpu) if int(args.gpu) < available_gpus else 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using device: {device}')

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print (args)
config = {
    'dataset': args.dataset,
    'mode': args.mode,
    'epoches': args.epoches,
    'batch_size': args.batch_size,
    'early_stop': args.early_stop,
    'device': device,
    'lr': args.lr,
    'inference_ckp': args.inference_ckp,
    'path_ckp': args.path_ckp,
    'path_tb': args.path_tb
}

if __name__ == '__main__':
    Run(config=config).main()
