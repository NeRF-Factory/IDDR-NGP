import os.path
from PIL import Image
import shutil
import numpy as np
import math
import argparse
import cv2

np.random.seed(0)

"""
First, put the files to be processed into "squareImages" If you want to create datasets with different mask levels 
or different mask locations, remember to modify the corresponding directory name when saving. 
ex) images_8_mask25%_square_random
"""
def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dtype', type=str, default='llff', help='data type to be processed')
    parser.add_argument('--cut_path', type=str, default='./cut/', help='path used to store temporary files')
    parser.add_argument('--workdir', type=str,default='./images')
    parser.add_argument('--yolo_dir', type=str,default='./labels')
    parser.add_argument('--savedir', type=str, default='./mask', help='save path of processed pictures')
    parser.add_argument('--patch_size', type=int, default='10', help='size of occlusion patch')
    parser.add_argument('--padding', type=int, default='0', help='distance of each patch after splicing')
    parser.add_argument('--mask_level', type=float, default='0.25',
                        help='level of mask per image: 0.25 / 0.50 / 0.75...')
    parser.add_argument('--position', type=str, default='fix', help="position of mask: 'fix' / 'random' ")
    return parser


# Cut the picture into patches
def cut_image(image, patch_size, index, images_num, mask, mask_level, position,dtype):
    width, height = image.size

    image_row = math.ceil(height / patch_size)
    image_column = math.ceil(width / patch_size)
    """
    generate the serial number of the patch
    from left to right 
    from top to bottom
    """
    if mask is None:
        mask = np.zeros((images_num, image_row * image_column))
    total = image_row * image_column
    # randomly select the serial number of a certain proportion of patches according to the given level
    if position == 'fix':
        np.random.seed(0)
    image_list = []
    # (left, upper, right, lower)
    for i in range(0, image_row):
        for j in range(0, image_column):
            if (mask[index][i * image_column + j] == 0):
                box = (
                j * patch_size, i * patch_size, min((j + 1) * patch_size, width), min((i + 1) * patch_size, height))
                image_list.append(image.crop(box))
            else:
                # consider boundary
                patch_width = abs(width - (j + 1) * patch_size)
                patch_height = abs(height - (i + 1) * patch_size)
                if patch_width == 0:
                    patch_width = patch_size
                if patch_height == 0:
                    patch_height = patch_size
                if dtype == 'blender':
                    box = Image.new('RGBA', (min(patch_size, patch_width), min(patch_size, patch_height)), (96, 96,96))
                else:
                    box = Image.new('RGB', (min(patch_size, patch_width), min(patch_size, patch_height)), (96,96, 96))
                image_list.append(box)
    print("id{0}done.".format(index))
    return image_list, mask, image_row, image_column, width, height


def save_images(image_list, save_path):
    if os.path.exists(save_path):
        shutil.rmtree(cut_path)
    os.mkdir(save_path)
    index = 0
    for image in image_list:
        image.save(os.path.join(save_path, str(index) + '.png'))
        index += 1


def image_compose(image_size, image_row, image_column, padding, images_path, image_save_path, width, height, dtype,
                  name):
    IMAGES_FORMAT = ['.bmp', '.jpg', '.tif', '.png']
    image_names = [name for name in os.listdir(images_path) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]

    image_names.sort(key=lambda x: int(x.split(("."), 2)[0]))

    if len(image_names) != image_row * image_column:
        raise ValueError("error!")
    if dtype == 'blender':
        to_image = Image.new('RGBA', (width + padding * (image_column - 1), height + padding * (image_row - 1)),
                             'white')
    else:
        to_image = Image.new('RGB', (width + padding * (image_column - 1), height + padding * (image_row - 1)), 'white')
    for y in range(1, image_row + 1):
        for x in range(1, image_column + 1):
            from_image = Image.open(images_path + image_names[image_column * (y - 1) + x - 1]).resize(
                (image_size, image_size))
            to_image.paste(from_image, (
                (x - 1) * image_size + padding * (x - 1), (y - 1) * image_size + padding * (y - 1)))
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    return to_image.save(os.path.join(image_save_path, name))


def getmask(imgdir, yolodir, patch_size):
    img = cv2.imread(imgdir)
    #     print(imgdir)
    H, W = img.shape[:2]
    patch_W = math.ceil(W / patch_size)
    patch_H = math.ceil(H / patch_size)
    mask_matrix = np.zeros((patch_H, patch_W))
    # 去掉序号
    yolo_data = np.loadtxt(yolodir)[..., 1:]
    # 横轴缩放
    yolo_data[..., 0] = yolo_data[..., 0] * W
    yolo_data[..., 2] = yolo_data[..., 2] * W
    # 纵轴缩放
    yolo_data[..., 1] = yolo_data[..., 1] * H
    yolo_data[..., 3] = yolo_data[..., 3] * H
    print(yolo_data)
    for ele in yolo_data:
        left_edge = ele[0] - ele[2]
        right_edge = ele[0] + ele[2]
        top_edge = ele[1] - ele[3]
        bottom_edge = ele[1] + ele[3]

        left_patch = min(math.floor(left_edge / patch_size), patch_W - 1)
        right_patch = min(math.ceil(right_edge / patch_size), patch_W - 1)
        top_patch = min(math.floor(top_edge / patch_size), patch_H - 1)
        bottom_patch = min(math.ceil(bottom_edge / patch_size), patch_H - 1)
        #         print(left_patch,right_patch)
        #         print(top_patch,bottom_patch)
        for x in range(top_patch, bottom_patch):
            for y in range(left_patch, right_patch):
                mask_matrix[x][y] = 1
    return mask_matrix.reshape(-1).tolist()

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    dtype = args.dtype
    cut_path = args.cut_path
    workdir = args.workdir
    savedir = args.savedir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    patch_size = args.patch_size
    padding = args.padding
    mask_level = args.mask_level
    position = args.position
    yolo_dir = args.yolo_dir
    imgList = [f for f in sorted(os.listdir(workdir)) \
               if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    index = 0

    names = [name.split(".")[0] for name in os.listdir(workdir)if name.endswith((".jpg", ".JPG", ".png"))]
    mask = []
    for name in names:
        mask.append(getmask(os.path.join(workdir, name + '.png'), os.path.join(yolo_dir, name + '.txt'), patch_size))
    mask = np.array(mask)
    for name in imgList:
        image = Image.open(os.path.join(workdir, name))
        image_list, mask, image_row, image_column, width, height = cut_image(image, patch_size, index, len(imgList),
                                                                             mask, mask_level,position,dtype)
        index += 1
        save_images(image_list, cut_path)
        image_size = patch_size
        images_path = cut_path
        image_save_path = savedir
        image_compose(image_size, image_row, image_column, padding, images_path, image_save_path, width, height, dtype,
                      name)
        shutil.rmtree(cut_path)

    # The mask matrix mentioned in the paper
    np.savetxt(os.path.join(savedir, "position.txt"), mask, fmt='%d')
    print(mask.mean())
