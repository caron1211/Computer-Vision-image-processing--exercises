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
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
alpha_slider_max = 100
title_window = 'Gamma Correction'


def on_trackbar(val):
    pass



# norm from 0-100 to 0-2 , multy by 0.02
def norm_val(val):
    norm = val *0.02
    return norm

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    I used Trackbar of cv2 The values in the trackbar are between 0-100  and default is 50.
    I converted this range to the [0-2] ( default is 1) and called tha value gamma.
    Each time the user moved the range I updated the values of gamma powerful image
    To exit GUI, click esc
    """

    cv.namedWindow(title_window)  # windows
    trackbar_gamma = 'Gamma x %d' % alpha_slider_max
    cv.createTrackbar(trackbar_gamma, title_window, 50, alpha_slider_max, on_trackbar)

    while (True):
        if LOAD_GRAY_SCALE == rep:
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        else:
            img = cv.imread(img_path)

        #  normalize
        img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

        b = cv.getTrackbarPos(trackbar_gamma, title_window)
        if b == 50:
            gamma = 1
        else:
            gamma = norm_val(b)
            img = np.power(img, gamma)
        font = cv.FONT_HERSHEY_SIMPLEX
        # put the gamma value on the image
        cv.putText(img, str(gamma), (50, 150), font, 0.5, (0, 0, 0), 2)
        cv.imshow(title_window,img)



    cv.destroyAllWindows()


def main():
    gammaDisplay('gamma_example.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
