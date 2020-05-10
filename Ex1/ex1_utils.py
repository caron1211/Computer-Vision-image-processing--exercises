"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys





def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns and returns in converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    The function used cv2.imread function to read the image.
    It distinguished the case where the image is GRAYSCALE and RGB by representation param
    In the case where the representation is RGB i convert from BGR to RGB
    After converting to the matrix, I normalized all pixels to values between 0 and 1 by cv2.normalize

    """
    if (representation == LOAD_GRAY_SCALE):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        img_bgr = cv2.imread(filename)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert from BGR to RGB
    # normalize
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return norm_image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    The function call to 'imReadAndConvert' and then use 'matplotlib.pyplot' library to show the image
    """
    img = imReadAndConvert(filename, representation)
    if (representation == LOAD_GRAY_SCALE) :
        plt.imshow(img,cmap='gray')
    else:
        plt.imshow(img)
    plt.show()



def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    The function does dot product with the image and the matrix
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    return np.dot(imgRGB, yiq_from_rgb.T.copy()) # transposed


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    The function does dot product with the image and the matrix

    """
    rgb_from_yiq = np.array([[1.00, 0.956, 0.623],
                        [1.0, -0.272, -0.648],
                        [1.0, -1.105, 0.705]])
    return np.dot(imgYIQ, rgb_from_yiq.T.copy())  # transposed


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: imgEq:image after equalize , histOrg: the original histogram, histEq: the new histogram
        The function get image performing a histogram calculation, I used the numpy function
        and then computed the histogram cdf and then mapped the pixels in the image to the optimal cdf
    """
    imgOrigCopy = imgOrig.copy()
    # if the image is rgb convert to YIQ and them continue
    if len(imgOrig.shape) == 3:
        imgYiq = transformRGB2YIQ(imgOrig)
        imgOrig = imgYiq[:, :, 0]

    imgOrig = normalizeTo256(imgOrig)
    histOrg , bins = np.histogram(imgOrig.flatten(), 256, [0, 256])

    # Original Histogram:
    plt.subplot(2, 1, 1)
    histOrig, bins = np.histogram(imgOrig.flatten(), 256, [0, 255])
    cdf = histOrig.cumsum()  # cumulative
    cdf_normalized = cdf * histOrig.max() / cdf.max()
    plt.title('Original image histogram with CDF')
    plt.plot(cdf_normalized, color='b')
    plt.hist(imgOrig.flatten(), 256, [0, 255], color='r')
    plt.xlim([0, 255])
    plt.legend(('cdf - ORIGINAL', 'histogram - ORIGINAL'), loc='upper left')
    # plt.show()

    cdf = histOrg.cumsum()
    # cdf_normalized = cdf * histOrg.max() / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    imgEq = cdf[imgOrig]
    histEq, bins = np.histogram(imgEq.flatten(), 256, [0, 256])

    # histogram for equalized image:
    histEq, bins = np.histogram(imgEq.flatten(), 256, [0, 255])
    cdf = histEq.cumsum()  # cumulative
    cdf_normalized = cdf * histEq.max() / cdf.max()
    plt.subplot(2, 1, 2)
    plt.title('Equalized image histogram with CDF ')
    plt.plot(cdf_normalized, color='b')
    plt.hist(imgEq.flatten(), 256, [0, 255], color='r')
    plt.xlim([0, 255])
    plt.legend(('cdf - EQUALIZED', 'histogram - EQUALIZED'), loc='upper right')

    # if the  original image was RGB return back to RGB
    if len(imgOrigCopy.shape) == 3:
        imgEq = cv2.normalize(imgEq.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        imgYiq [:, :, 0]= imgEq
        imgEq = transformYIQ2RGB(imgYiq)
        imgEq = normalizeTo256(imgEq)

    # plt.savefig('histoOfhsitogramEqualize.png')
    # saveImageWithCv2('hsitogramEqualizeRes.jpg',imgEq)
    return imgEq, histOrg, histEq



def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
        I divided the histogram into nQuant equal parts and then found the average in each section.
        After finding the average, I changed all the pixels in that range to their average.
        Repeat this process nIter times and at each stage find the mistake between the image we calculated and the original image

    """
    imgOrigCopy = imOrig.copy()
    if len(imOrig.shape) == 3:
        imgYiq = transformRGB2YIQ(imOrig)
        imOrig = imgYiq[:, :, 0]

    imgnew = normalizeTo256(imOrig)
    hist, bins = np.histogram(imgnew.flatten(), 256, [0, 256])
    init = 255 / nQuant
    qImage_lst = []
    err_lst = []
    k = int(256 / nQuant)
    borders = np.arange(0, 257, k)
    borders[nQuant] = 255
    # print("borders{}".format(borders))

    for i in range(0, nIter):
        imgQuant = imgnew
        # print("borders_n{}".format(borders))
        weightedMean = np.zeros(nQuant) 
        for j in range(0, nQuant):
            low_bound = int(borders[j])
            # print("low_bound{}".format(low_bound))
            high_bound = int (borders[j+1] +1)
            # print("high_bound{}".format(high_bound))
            weightedMean[j] = getWeightedMean(range(low_bound,high_bound), hist[low_bound:high_bound])
        # print("means{}".format(weightedMean))
        for j in range(0, nQuant):
            bool_pixels = (imgQuant >= borders[j]) & (imgQuant <= borders[j + 1])
            imgQuant[bool_pixels] = weightedMean[j]
            imgQuant = np.rint(imgQuant).astype(int)  # Round elements of the array to the nearest integer.

        borders = (weightedMean[:-1] + weightedMean[1:]) / 2  # get the middle between 2 avg
        borders = np.insert(borders, 0, 0)  # add 0 to begin
        borders = np.append(borders, 255) # add 255 to end
        borders = np.rint(borders).astype(int)

        mse = getMse(imgnew, imgQuant)
        err_lst.append(mse)  # add err to list

        if len(imgOrigCopy.shape) == 3:
            imgQuant = cv2.normalize(imgQuant.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            imgYiq[:, :, 0] = imgQuant
            imgQuant = transformYIQ2RGB(imgYiq)
            imgQuant = normalizeTo256(imgQuant)
        qImage_lst.append(imgQuant)  # add the image to list

    # saveImageWithCv2('quantimageRes.jpg',imgQuant)
    return qImage_lst, err_lst

# normalize fron [0,1] to [0,255]

def normalizeTo256(imgOrig: np.ndarray) -> np.ndarray:
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = np.ceil(imgOrig)
    imgOrig = imgOrig.astype('uint8')
    return imgOrig
# WeightedMean of np
def getWeightedMean(intens: np.ndarray, vals: np.ndarray) -> int:
    # print("intens {}".format(intens))
    # print("vals{}".format(vals))
    weightedMean = np.average(intens, weights=vals)
    return weightedMean

# I used https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
def getMse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

# Save img with cv2 (BGR)
def saveImageWithCv2(filename , img:np.ndarray):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)




