import numpy as np
import cv2
from numba import jit


def overlay_image(img_in, mask, mask_color=(255, 0, 0), contour_width=1, alpha=0.5):
    """
    img_in: input BGR image, type: uint8
    mask: binary mask the same dimensions as input image, type: uint8
    mask_color: RGB color of the semi-transparent overlayed mask
    contour_width: width of the contour around the mask in pixels
    alpha: transparency parameter (larger value - less transparent)
    """
    img = np.copy(img_in)

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    img_r[mask > 0] = np.round(img_r[mask > 0] * (1 - alpha) + alpha * mask_color[0])
    img_g[mask > 0] = np.round(img_g[mask > 0] * (1 - alpha) + alpha * mask_color[1])
    img_b[mask > 0] = np.round(img_b[mask > 0] * (1 - alpha) + alpha * mask_color[2])

    M = (mask > 0).astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, contours, -1, mask_color[::-1], contour_width)
    return img

@jit(nopython=True)
def mask_to_rle(m):
    """
    # Input: 2-D numpy array
    # Output: list of numbers (1st number = #0s, 2nd number = #1s, 3rd number = #0s, ...)
    """
    # reshape mask to vector
    v = np.reshape(m, (m.shape[0] * m.shape[1]))

    # output is empty at the beginning
    rle = []
    # index of the last different element
    last_idx = 0
    # check if first element is 1, so first element in RLE (number of zeros) must be set to 0
    if v[0] > 0:
        rle.append(0)

    # go over all elements and check if two consecutive are the same
    for i in range(1, v.size):
        if v[i] != v[i - 1]:
            rle.append(i - last_idx)
            last_idx = i

    # handle last element of rle
    if last_idx < v.size - 1:
        # last element is the same as one element before it - add number of these last elements
        rle.append(v.size - last_idx)
    else:
        # last element is different than one element before - add 1
        rle.append(1)

    return rle

@jit(nopython=True)
def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_+j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    return np.reshape(np.array(v, dtype=np.uint8), (height, width))

def mask2bbox(mask):
    """
    mask: 2-D array with a binary mask
    output: coordinates of the top-left and bottom-right corners of the minimal axis-aligned region containing all positive pixels
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (cmin, rmin, cmax, rmax)

def encode_mask(mask):
    """
    mask: input binary mask, type: uint8
    output: full RLE encoding in the format: (x0, y0, w, h), RLE
    first get minimal axis-aligned region which contains all positive pixels
    extract this region from mask and calculate mask RLE within the region
    output position and size of the region, dimensions of the full mask and RLE encoding
    """
    # calculate coordinates of the top-left corner and region width and height (minimal region containing all 1s)
    x_min, y_min, x_max, y_max = mask2bbox(mask)
    tl_x = x_min
    tl_y = y_min
    region_w = x_max - x_min + 1
    region_h = y_max - y_min + 1

    # extract target region from the full mask and calculate RLE
    # do not use full mask to optimize speed and space
    target_mask = mask[tl_y:tl_y+region_h, tl_x:tl_x+region_w]
    rle = mask_to_rle(np.array(target_mask))

    return (tl_x, tl_y, region_w, region_h), rle

def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = [el for el in elements[4:]]

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)
