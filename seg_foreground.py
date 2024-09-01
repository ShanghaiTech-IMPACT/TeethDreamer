import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import torchvision
import os
from skimage.io import imsave, imread
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import argparse
import sys

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 0, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=175):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def on_move(event):
    if event.inaxes:
        x = event.xdata
        y = event.ydata
        ax.set_title('Mouse Position: (%.2f, %.2f)' % (x, y))
        plt.draw()

def on_click(event):
    global tooth_mask
    if event.inaxes:
        ix, iy = float(event.xdata), float(event.ydata)
        clicks.append([ix, iy])
        if event.button==1:
            labels.append(1)
        else:
            labels.append(0)
        masks, scores, logits = predictor.predict(point_coords=np.array(clicks),
                                                    point_labels=np.array(labels),
                                                    multimask_output=False,
                                                    )
        plt.imshow(image)
        show_mask(masks[0], plt.gca())
        tooth_mask=masks[0].copy()
        plt.show()

def click_seg(image_path, mask_path):
    global out_name, clicks, labels, image, tooth_mask
    out_name = mask_path
    img = cv2.imread(image_path)
    img = np.asarray(img.data)
    out = []
    for i in range(16):
        image = cv2.cvtColor(img[:, i*256:(i+1)*256, :].copy(), cv2.COLOR_BGR2RGB)
        clicks = []
        labels = []
        predictor.set_image(image)
        fig, ax = plt.subplots()
        plt.imshow(image)
        plt.axis('on')
        tooth_mask=np.array([])
        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        res = np.asarray(image.data)
        res = np.concatenate([res, tooth_mask[:,:,None].astype(np.uint8)*255], axis=-1)
        out.append(res)

    out = np.concatenate(out, axis=1)
    out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(out_name, out)

if __name__=="__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='The path to your generated image')
    parser.add_argument('--seg', required=True, help='The directory to store the segmented image')
    args = parser.parse_args()
    mark_every_tooth=False
    image=None
    out_name=None
    clicks=None
    labels=None
    masks_tot=None
    tooth_mask=None
    sam_checkpoint = "ckpt/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    click_seg(args.img, args.seg)


