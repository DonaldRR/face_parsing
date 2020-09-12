from PIL import Image
import numpy as np
import torchvision
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2

# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def vis_embedding(embeddings):

    # embeddings: (n, c, h, w)
    embeddings = embeddings.detach().cpu().numpy()
    n, c, h, w = embeddings.shape
    images = []
    for i in range(n):
        images.append(draw_projection(embeddings[i]))
#        embedding_map = embeddings[i]
#        embedding_map = np.reshape(embedding_map, (c, -1))
#        embedding_map = np.moveaxis(embedding_map, 1, 0)
#        pca = PCA(n_components=3)
#        reduced_embedding_map = pca.fit_transform(embedding_map)
#        reduced_embedding_map = normalize(reduced_embedding_map) * 128 + 128
#        reduced_embedding_map = np.reshape(reduced_embedding_map, (h, w, 3)).astype(int)
#        images.append(np.moveaxis(reduced_embedding_map, 2, 0))
    images = np.stack(images, axis=0)

    return torch.from_numpy(images).permute(0, 3, 1, 2)


def draw_projection(embedding, dsize=(256, 256)):
    # embeddings: (C, H, W)
    C, H, W = embedding.shape
    embedding = np.moveaxis(embedding, 0, 2)

    embedding = embedding.reshape((H * W, C))

    pca_3d = PCA(n_components=3)
    embd_3d = pca_3d.fit_transform(embedding)
    embd_3d = normalize(embd_3d) * .5 + .5

    pca_2d = PCA(n_components=2)
    embd_2d = pca_2d.fit_transform(embedding)

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.scatter(embd_2d[:, 0], embd_2d[:, 1], color=list(map(tuple, embd_3d)), marker='.')
    ax.axis('off')

    canvas.draw()  # draw the canvas, cache the renderer

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = image.reshape((int(height), int(width), 3))
    image = cv2.resize(image, (dsize[1], dsize[0]), interpolation=cv2.INTER_NEAREST)
    embd_3d = embd_3d.reshape((H, W, 3)) * 255
    embd_3d = cv2.resize(embd_3d.astype(float), (dsize[1], dsize[0]), interpolation=cv2.INTER_LINEAR)
    ret = np.concatenate((embd_3d, image), axis=0).astype(int)

    return ret

def decode_parsing(labels, num_images=1, num_classes=21, is_pred=False):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    pred_labels = labels[:num_images].clone().cpu().data
    if is_pred:
        pred_labels = torch.argmax(pred_labels, dim=1)
    n, h, w = pred_labels.size()

    labels_color = torch.zeros([n, 3, h, w], dtype=torch.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]

    return labels_color

def inv_preprocess(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    rev_imgs = imgs[:num_images].clone().cpu().data
    rev_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_images):
        rev_imgs[i] = rev_normalize(rev_imgs[i])

    return rev_imgs

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

class SingleGPU(nn.Module):
    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.cuda(non_blocking=True))