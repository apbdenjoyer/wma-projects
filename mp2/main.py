import sys

import cv2
import numpy as np
import os
import tkinter as tk


def shift_hue_of_image(src, degrees):
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + degrees) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def morph_image_mask(src, erosion_steps, dilation_steps):
    kernel = np.ones((5, 5))
    dilated = cv2.dilate(src, kernel, iterations=dilation_steps)
    eroded = cv2.erode(dilated, kernel, iterations=erosion_steps)
    return eroded


def create_mask_from_color(src, lower, upper):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lower_color = np.array([lower, 50, 50])
    upper_color = np.array([upper, 255, 255])
    return cv2.inRange(hsv, lower_color, upper_color)


def nothing(x):
    pass


def merge_images(images):
    # Ensure all images are in the same 3-channel format (BGR) for concatenation
    for i, img in enumerate(images):
        if img.ndim != 3:
            images[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Concatenate all images horizontally in a single row
    full = cv2.hconcat(images)

    return full

def load_image(path, image_name):
    image_path = os.path.join(path, image_name)
    return cv2.imread(image_path)


#
# def norm_size():
#     global image
#     h, w = image.shape[:2]
#     if h > w:
#         if h > 800:
#             s = (1 - (800 / h)) * (-1)
#             w = w + int(w * (s))
#             h = h + int(h * (s))
#             image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
#     else:
#         if w > 800:
#             s = (1 - (800 / w)) * (-1)
#             w = w + int(w * (s))
#             h = h + int(h * (s))
#             image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
#     cv2.imshow('obrazek', image)
#
#
# # key e
# def hsv_range():
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     # Convert the HSV colorspace
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#     # Threshold the HSV image to get only blue color
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower, upper)
#     cv2.imshow('obrazek', mask)
#
#
# # key r
# def hsv_bitwise():
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower, upper)
#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(index, index, mask=mask)
#     cv2.imshow('obrazek', res)
#
#
# # key t
# def hsv_median():
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     ksize = cv2.getTrackbarPos('ksize', 'obrazek')
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower, upper)
#     res = cv2.bitwise_and(index, index, mask=mask)
#     res = cv2.medianBlur(res, ksize=ksize)
#     cv2.imshow('obrazek', res)
#
#
# # key f
# def morphology():
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     ksize = cv2.getTrackbarPos('ksize', 'obrazek')
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower, upper)
#     kernel = np.ones((ksize, ksize), np.uint8)
#     mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     cv2.imshow('obrazek', mask_without_noise)
#
#
# # key g
# def morphology2():
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     ksize = cv2.getTrackbarPos('ksize', 'obrazek')
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower, upper)
#     kernel = np.ones((ksize, ksize), np.uint8)
#     # mask_without_noise = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
#     mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     cv2.imshow('obrazek', mask_closed)
#
#
# # key h
# def marker():
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#
#     mask = cv2.inRange(hsv_frame, lower, upper)
#     contours, hierarchy = cv2.findContours(mask, 1, 2)
#     print(contours)
#     M = cv2.moments(contours[0])
#     cx = int(M['m10'] / M['m00'])
#     cy = int(M['m01'] / M['m00'])
#     image_marker = index.copy()
#     cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(
#         0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
#     cv2.imshow('obrazek', image_marker)
#
#
# # key p
# def connect_mask():
#     # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     ksize = cv2.getTrackbarPos('ksize', 'obrazek')
#
#     # Konwersja obrazu na przestrzeń kolorów HSV
#     hsv_frame = cv2.cvtColor(index, cv2.COLOR_BGR2HSV)
#
#     # Utworzenie maski dla pierwszego zakresu kolorów
#     lower = np.array([low_color, 100, 100])
#     upper = np.array([high_color, 255, 255])
#     mask = cv2.inRange(hsv_frame, lower, upper)
#
#     # Nałożenie maski na obraz i wyświetlenie wyniku
#     res = cv2.bitwise_and(index, index, mask=mask)
#     cv2.imshow('mask 1', res)
#
#     # Utworzenie maski dla drugiego zakresu kolorów
#     lower = np.array([0, 100, 100])
#     upper = np.array([ksize, 255, 255])
#     mask2 = cv2.inRange(hsv_frame, lower, upper)
#
#     # Nałożenie drugiej maski na obraz i wyświetlenie wyniku
#     res = cv2.bitwise_and(index, index, mask=mask2)
#     cv2.imshow('mask 2', res)
#
#     # Połączenie dwóch masek za pomocą operacji bitowej OR
#     b_mask = cv2.bitwise_or(mask, mask2)
#
#     # Nałożenie połączonej maski na obraz i wyświetlenie wyniku
#     res = cv2.bitwise_and(index, index, mask=b_mask)
#     cv2.imshow('obrazek', res)
#
#
# # key j
# def find_circle():
#     # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
#     low_color = cv2.getTrackbarPos('low', 'obrazek')
#     high_color = cv2.getTrackbarPos('high', 'obrazek')
#     ksize = cv2.getTrackbarPos('ksize', 'obrazek')
#
#     # Utwórz kopię obrazu, aby nie modyfikować oryginału
#     c_img = index.copy()
#
#     # Konwersja obrazu na skalę szarości
#     gimg = cv2.cvtColor(c_img, cv2.COLOR_RGB2GRAY)
#
#     # Zastosowanie rozmycia na obrazie w skali szarości
#     bimg = cv2.blur(gimg, (ksize, ksize))
#
#     # Wykrywanie okręgów za pomocą transformacji Hougha
#     circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, high_color, low_color)
#     print(circles)  # Wyświetlenie wykrytych okręgów (surowe dane)
#
#     # Zaokrąglenie współrzędnych wykrytych okręgów do liczb całkowitych
#     circles = np.uint16(np.around(circles))
#     print(circles)  # Wyświetlenie zaokrąglonych współrzędnych okręgów
#
#     # Iteracja po wykrytych okręgach i rysowanie ich na obrazie
#     for i in circles[0, :]:
#         # Rysowanie okręgu na obrazie (środek: (i[0], i[1]), promień: i[2])
#         cv2.circle(c_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
#
#     # Wyświetlenie obrazu z narysowanymi okręgami
#     cv2.imshow('obrazek', c_img)
#

# key k
def line():
    global image
    # Pobierz wartości progów dolnego i górnego z trackbarów
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')

    # Konwersja obrazu na skalę szarości
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Wykrywanie krawędzi za pomocą algorytmu Canny'ego
    edges = cv2.Canny(gray, low_color, high_color, apertureSize=3)

    # Wykrywanie linii za pomocą transformacji Hougha
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90,
                            minLineLength=100, maxLineGap=5)

    # Utworzenie kopii obrazu, aby narysować linie
    image_l = image.copy()

    # Iteracja po wykrytych liniach i rysowanie ich na obrazie
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_l, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Wyświetlenie obrazu z narysowanymi liniami
    cv2.imshow("obrazek", image_l)


# key o
def rotate():
    global image
    # Pobierz wartość kąta obrotu z trackbara o nazwie 'low'
    rot = cv2.getTrackbarPos('low', 'obrazek')

    # Pobierz wymiary obrazu
    height, width = image.shape[:2]

    # Oblicz środek obrazu
    center_x, center_y = (width / 2, height / 2)

    # Utwórz macierz transformacji dla obrotu obrazu
    M = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)

    # Zastosuj macierz transformacji, aby obrócić obraz
    rotated_image = cv2.warpAffine(image, M, (width, height))

    # Wyświetl obrócony obraz w oknie o nazwie 'obrazek'
    cv2.imshow('obrazek', rotated_image)


images = []
path_to_images = r'..\mp2\data'
image = None
fun = None


def change_h(x):
    global fun
    if fun is not None:
        fun()


def find_circle():
    # Pobierz wartości z suwaków (trackbarów) dla dolnego i górnego zakresu koloru oraz rozmiaru maski
    low_color = cv2.getTrackbarPos('low', 'obrazek')
    high_color = cv2.getTrackbarPos('high', 'obrazek')
    ksize = cv2.getTrackbarPos('ksize', 'obrazek')

    # Utwórz kopię obrazu, aby nie modyfikować oryginału
    image_with_circles = image.copy()

    # Konwersja obrazu na skalę szarości
    gimg = cv2.cvtColor(image_with_circles, cv2.COLOR_RGB2GRAY)

    # Zastosowanie rozmycia na obrazie w skali szarości
    bimg = cv2.blur(gimg, (ksize, ksize))

    # Wykrywanie okręgów za pomocą transformacji Hougha
    circles = cv2.HoughCircles(bimg, cv2.HOUGH_GRADIENT, high_color, low_color)
    print(circles)  # Wyświetlenie wykrytych okręgów (surowe dane)

    if circles is not None:
        # Zaokrąglenie współrzędnych wykrytych okręgów do liczb całkowitych
        circles = np.uint16(np.around(circles))
        print(circles)  # Wyświetlenie zaokrąglonych współrzędnych okręgów

        # Iteracja po wykrytych okręgach i rysowanie ich na obrazie
        for i in circles[0, :]:
            # Rysowanie okręgu na obrazie (środek: (i[0], i[1]), promień: i[2])
            cv2.circle(image_with_circles, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return image_with_circles


window_name = 'obrazek'

def get_image(index):
    return images[index % len(images)]

def main():
    global window_name, images

    files = os.listdir(path_to_images)
    for i in range(len(files)):
        image_name, extension = os.path.splitext(files[i])

        if extension.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            image = load_image(path_to_images, files[i])
            images.append(image)

    cv2.namedWindow(window_name)

    cv2.createTrackbar("hue shift", window_name, 0, 180, nothing)
    cv2.createTrackbar("mask lower", window_name, 0, 180, nothing)
    cv2.createTrackbar("mask upper", window_name, 0, 180, nothing)
    cv2.createTrackbar("blur size", window_name, 0, 25, nothing)
    cv2.createTrackbar("erosion iters", window_name, 0, 25, nothing)
    cv2.createTrackbar("dilation iters", window_name, 0, 25, nothing)

    previous_values = (-1, -1, -1, -1, -1, -1)

    index = 0
    image = images[index]
    while True:
        hue = cv2.getTrackbarPos("hue shift", window_name)
        lower = cv2.getTrackbarPos("mask lower", window_name)
        upper = cv2.getTrackbarPos("mask upper", window_name)
        blur = cv2.getTrackbarPos("blur size", window_name)
        erosion = cv2.getTrackbarPos("erosion iters", window_name)
        dilation = cv2.getTrackbarPos("dilation iters", window_name)

        current_values = (hue, lower, upper, blur, erosion, dilation)

        if current_values != previous_values:
            hue_shifted_img = shift_hue_of_image(image, hue)
            mask = create_mask_from_color(hue_shifted_img, lower, upper)
            morphed = morph_image_mask(mask, erosion, dilation)

            group = [hue_shifted_img, morphed, hue_shifted_img, morphed]

            merged = merge_images(group)

            height, width = merged.shape[:2]
            window_width = width + 100  # Add some space for trackbars
            window_height = height + 200  # Add some space for trackbars

            # Resize the window
            cv2.resizeWindow(window_name, window_width, window_height)

            cv2.imshow(window_name,merged)

            previous_values=current_values

        key = cv2.waitKey(1)
        # -----------wybor obrazka----------------
        if key == ord(','):  # go left [<]
            index+=1
            image = get_image(index)
        elif key == ord('.'):
            index -= 1
            image = get_image(index)
        elif key == ord('j'):
            try:
                find_circle()
                fun = find_circle
            except Exception as e:
                continue

        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
