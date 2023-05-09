import numpy as np
import cv2 as cv


def getContrast(
    image: np.ndarray, topHatSize: int, blackHatSize: int, dilateSize: int
) -> np.ndarray:
    kernelTop = cv.getStructuringElement(cv.MORPH_RECT, (topHatSize, topHatSize))
    kernelBlack = cv.getStructuringElement(cv.MORPH_RECT, (blackHatSize, blackHatSize))
    tophat = cv.morphologyEx(image, cv.MORPH_TOPHAT, kernelTop)
    blackhat = cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernelBlack)

    addition = cv.add(tophat, blackhat)
    kerneldilate = cv.getStructuringElement(cv.MORPH_RECT, (dilateSize, dilateSize))
    result = cv.dilate(addition, kerneldilate)

    return result


def perform_processing(image: np.ndarray) -> str:
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred = cv.blur(image, (5, 5))

    contrasted = getContrast(blurred, 17, 17, 3)

    canny = cv.Canny(contrasted, 10, 255)

    canny = cv.resize(canny, (1024, 768))
    cv.imshow("result", canny)
    while True:
        if cv.waitKey(10) == ord("q"):
            cv.destroyAllWindows()
            break
    return "PO12345"
