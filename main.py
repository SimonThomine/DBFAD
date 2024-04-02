import argparse

from KD_ReverseResidual import KD_ReverseResidual

import torch


def get_args():
    parser = argparse.ArgumentParser(description='DBFAD')
    parser.add_argument('--phase', default='train')
    parser.add_argument("--model", type=str, default="ReverseResidual")
    parser.add_argument("--obj", type=str, default="carpet")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    print('Available devices ', torch.cuda.device_count())
    print('Current cuda device ', torch.cuda.current_device())
    args = get_args()
    obj = args.obj
    print(obj)

    if args.model == "ReverseResidual":
        DG=False
    else:
        DG=True

    print("ReverseResidual")
    kd_reverseResi = KD_ReverseResidual(obj=obj, vis=False, data_path="dataset/MVTEC",DG=DG)
    if args.phase == 'train':
        kd_reverseResi.train()
        kd_reverseResi.test()
    elif args.phase == 'test':
        kd_reverseResi.test()

