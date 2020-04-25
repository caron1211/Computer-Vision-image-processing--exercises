# Computer-Vision-image-processing--exercises
Exercises in computer vision and image processing course
## Exercise 1: Image Representations and Point Operations
### 4.1 Reading an image into a given representation

```
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
```

### 4.2 Displaying an image
```
def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    The function call to 'imReadAndConvert' and then use 'matplotlib.pyplot' library to show the image
    """
```
### 4.3 Transforming an RGB image to YIQ color space

```
def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    The function does dot product with the image and the matrix
    """
```
### 4.4 Histogram equalization

```
def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: imgEq:image after equalize , histOrg: the original histogram, histEq: the new histogram
        The function get image performing a histogram calculation, I used the numpy function
        and then computed the histogram cdf and then mapped the pixels in the image to the optimal cdf
    """
```
#### Results:  
orignal image:
![original](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/bac/bac_con.png?raw=true)  
hsitogram:  
![Image description](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/bac/histoOfhsitogramEqualize.png?raw=true)  
result:  
![Image description](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/bac/hsitogramEqualizeRes.png?raw=true)  
orignal image:  
![original](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/beach_rgb/beach.jpg?raw=true)  
hsitogram:  
![Image description](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/beach_rgb/histoOfhsitogramEqualize.png?raw=true)  
result:  
![Image description](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/beach_rgb/hsitogramEqualizeRes.png?raw=true)


### 4.5 Optimal image quantization

```
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
```
#### Results:  

original
![Image description](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/dark/dark.jpg?raw=true)
quantization with 4 color
![Image description](https://github.com/caron1211/Computer-Vision-image-processing--exercises/blob/master/Ex1/dark/quantimageRes.png?raw=true)

### 4.6 Gamma Correction

```
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
```
