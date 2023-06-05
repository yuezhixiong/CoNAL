import sys
sys.path.append("..")
from CoNAL.model import CoNALArch
from CoNAL.utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr


DEVICE = 'cuda'
DATA_DIR = 'E:/Dataset/nyu'

model_path = '../logs/arch_hps_2023_0530_1637/arch_hps_e30.pth'
arch_path = '../CoNAL/arch_hps.json'

# get arch from json file
arch = load_arch(arch_path)

# prepare model
model = CoNALArch(arch).to(DEVICE)
model.load_state_dict(torch.load(model_path))
model.eval()

# sets image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    # img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    # img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    img = (img+1)/2
    return img

# resizes the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=100, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crops the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def verify(input_image):
    input_tensor = torch.from_numpy(np.moveaxis(input_image, -1, 0)).float().unsqueeze(0).cuda()
    # print(input_tensor.shape)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    ax[0, 0].imshow(input_image)
    ax[0, 0].set_title('input image')

    with torch.no_grad():
        output_preds = model.predict(input_tensor)
        output_result = {}
        for idx,kv in enumerate(output_preds.items()):
            k, v = kv
            mask_image = v.squeeze(0).cpu()
            if k == 'segmentation':
                mask_image = mask_image.argmax(0).numpy().astype("uint8")
                # print(np.unique(mask_image))
                vmin, vmax = 0, 12
            elif k == 'depth':
                mask_image = mask_image.numpy().transpose(1,2,0)
                vmin, vmax = 0, 3.7
            else:
                mask_image = mask_image.numpy().transpose(1,2,0)
                vmin, vmax = -1, 1
            ax[(idx+1)//2, (idx+1)%2].imshow(mask_image, vmin=vmin, vmax=vmax)
            # print(k, v.shape)
            mask_image = ((mask_image - vmin) / (vmax - vmin) -0.5)*2
            output_result[k] = mask_image
            ax[(idx+1)//2, (idx+1)%2].set_title(k)

    [axi.set_axis_off() for axi in ax.ravel()]
    output_path = 'output.png'
    plt.savefig(output_path)
    # print(output_result.shape, output_result.min(), output_result.max())
    return output_result

def inference(img):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  img = pre_process_edgetpu(img, (288, 384, 3))
  output_result = verify(img)

  return 'output.png'

examples = [['test{}.png'.format(i)] for i in range(5)]
gr.Interface(inference, gr.Image(type="filepath"), gr.Image(type="filepath"), title="HPS", description="", examples=examples).launch(share=False)