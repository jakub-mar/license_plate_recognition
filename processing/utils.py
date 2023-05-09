import numpy as np
import cv2 as cv


def perform_processing(image: np.ndarray) -> str:
    # print(f"image.shape: {image.shape}")
    # TODO: add image processing here
    blurred = cv.GaussianBlur(image, (7, 7), 0)
    canny = cv.Canny(blurred, 80, 160)

    # canny = cv.morphologyEx(canny, cv.MORPH_OPEN, (5, 5))
    # canny = cv.medianBlur(canny, 11)
    dilation_size = 5
    dilation_type = cv.MORPH_RECT  # cv.MORPH_CROSS cv.MORPH_ELLIPSE

    element = cv.getStructuringElement(
        dilation_type,
        (2 * dilation_size + 1, 2 * dilation_size + 1),
        (dilation_size, dilation_size),
    )
    canny = cv.dilate(canny, element)
    # canny = cv.morphologyEx(canny, cv.MORPH_CLOSE, (8, 8))

    # erosion_size = 3
    # erosion_type = cv.MORPH_RECT  # cv.MORPH_CROSS cv.MORPH_ELLIPSE

    # element = cv.getStructuringElement(
    #     erosion_type,
    #     (2 * erosion_size + 1, 2 * erosion_size + 1),
    #     (erosion_size, erosion_size),
    # )
    # canny = cv.erode(canny, element)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    for contour in contours:
        x1, y1 = contour[0][0]
        approx = cv.approxPolyDP(contour, 0.009 * cv.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(contour)
            ratio = float(h / w)
            area = cv.contourArea(contour)
            if ratio >= 0.2 and ratio <= 0.5 and area >= 100000:
                cv.drawContours(image, [contour], -1, (0, 255, 255), 10)
        # area = cv.contourArea(contour)
        # if area > 600000:
        #     x, y, w, h = cv.boundingRect(contour)
        #     cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    image = cv.resize(image, (800, 600))
    canny = cv.resize(canny, (800, 600))
    cv.imshow("canny", canny)
    cv.imshow("default", image)

    while True:
        if cv.waitKey(10) == ord("q"):
            break
    return "PO12345"
