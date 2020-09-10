import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
# from scipy.misc import imsave
torch.multiprocessing.set_start_method("spawn", force=True)
from torch.utils import data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from networks.EAGR import EAGRNet
from collections import OrderedDict
from dataset.datasets import HelenDataSet
import os
import torchvision.transforms as transforms
from utils.miou import compute_mean_ioU
from copy import deepcopy
import cv2
from inplace_abn import InPlaceABN
from utils.miou import compute_mean_ioU, compute_confusion_matrix

DATA_DIRECTORY = './datasets/Helen'
IGNORE_LABEL = 255
NUM_CLASSES = 20
SNAPSHOT_DIR = './snapshots/'
INPUT_SIZE = (473,473)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CE2P Network")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--dataset", type=str, default='val',
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="choose gpu numbers") 
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    return parser.parse_args()

def valid(model, valloader, input_size, num_samples, dir=None):
    model.eval()
#    preds_parsing = np.zeros((num_samples, input_size[0], input_size[1]),
#                             dtype=np.uint8)
#    preds_parsing_bi = np.zeros((num_samples, input_size[0], input_size[1]),
#                             dtype=np.uint8)
#    preds_edge = np.zeros((num_samples, input_size[0], input_size[1]),
#                             dtype=np.uint8)

    scales = np.zeros((num_samples, 2), dtype=np.float32)
    centers = np.zeros((num_samples, 2), dtype=np.int32)

    ConfMat_parsing = np.zeros((11, 11))
    ConfMat_parsing_bi = np.zeros((2, 2))
    ConfMat_edge = np.zeros((2, 2))

    idx = 0
    interp = torch.nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)
    with torch.no_grad():
        for index, batch in enumerate(valloader):
            image, label, label1, meta = batch
            num_images = image.size(0)
            if index % 10 == 0:
                print('%d  processd' % (index * num_images))

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            scales[idx:idx + num_images, :] = s[:, :]
            centers[idx:idx + num_images, :] = c[:, :]

            pred, shallow_embedding, deep_embedding = model(image.cuda())

            def parse_to_label(pred, iterp_func):

                pred = iterp_func(pred).data.cpu().numpy()
                pred = pred.transpose(0, 2, 3, 1)
                pred = np.asarray(np.argmax(pred, axis=3), dtype=np.uint8)

                return pred
#            preds_parsing[idx:idx + num_images, :, :] = parse_to_label(pred_parsing, interp)
#            preds_parsing_bi[idx:idx + num_images, :, :] = parse_to_label(pred_parsing_bi, interp)
#            preds_edge[idx:idx + num_images, :, :] = parse_to_label(pred_edge, interp)
#            idx += num_images
            ConfMat_parsing += compute_confusion_matrix(parse_to_label(pred, interp), label, pred.size(1))

            if dir:
                pass

    def compute_mIoU_f1(m):
        n_class = len(m)
        mIoUs = []
        f1s = []
        recalls = []
        precisions = []
        for i in range(n_class):
            intersection = m[i][i] + 1
            n_true = np.sum(m[i, :]) + n_class
            n_pos = np.sum(m[:, i]) + n_class
            recall = intersection / n_true
            precision = intersection / n_pos
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(2 / (1 / recall + 1 / precision))
            mIoUs.append(intersection / (n_true + n_pos - intersection))

        return mIoUs, f1s, recalls, precisions

    parsing_mIoUs, parsing_f1s, parsing_recalls, parsing_precisions = compute_mIoU_f1(ConfMat_parsing)
    parsing_mean_accuracy = np.diag(ConfMat_parsing).sum() / ConfMat_parsing.sum()

    return {
        'parsing': {'mIoU': parsing_mIoUs, 'f1': parsing_f1s, 'recalls': parsing_recalls, 'precisions': parsing_precisions, 'mean_accuracy': parsing_mean_accuracy},
    }


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    gpus = [int(i) for i in args.gpu.split(',')]

    h, w = map(int, args.input_size.split(','))
    
    input_size = (h, w)

    model = EAGRNet(args.num_classes, InPlaceABN)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = HelenDataSet(args.data_dir, args.dataset, crop_size=input_size, transform=transform)
    num_samples = len(dataset)

    valloader = data.DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True)

    restore_from = args.restore_from
    state_dict_old = torch.load(restore_from,map_location='cuda:0')
    model.load_state_dict(state_dict_old)
    
    model.cuda()
    model.eval()

    save_path =  os.path.join(args.data_dir, 'full')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    parsing_preds, scales, centers = valid(model, valloader, input_size, num_samples, save_path)
    mIoU, f1 = compute_mean_ioU(parsing_preds, scales, centers, args.num_classes, args.data_dir, input_size, 'test', reverse=True)

    print(mIoU)
    print(f1)

if __name__ == '__main__':
    main()
