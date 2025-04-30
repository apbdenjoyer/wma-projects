import numpy as np
import os
import cv2
import tkinter as tk


def nothing(x):
    pass


def resize_image_to_screen_size(src, divider):
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    h, w = src.shape[:2]
    if w > h:
        return cv2.resize(src, (int(screen_width / divider), int(screen_width * h / (w * divider))),
                          interpolation=cv2.INTER_LINEAR)
    else:
        return cv2.resize(src, (int(screen_height * w / (h * divider)), int(screen_height / divider)),
                          interpolation=cv2.INTER_LINEAR)


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
    lower_color = np.array([lower, 0, 0])
    upper_color = np.array([upper, 255, 255])
    return cv2.inRange(hsv, lower_color, upper_color)


def merge_images(images):
    for i, img in enumerate(images):
        if img.ndim != 3:
            images[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return resize_image_to_screen_size(cv2.hconcat(images), 1.5)


def load_image(path, image_name, scale=0.5):
    img = cv2.imread(os.path.join(path, image_name))
    if img is not None:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img



def detect_circles(blurred):
    equalized = cv2.equalizeHist(blurred)
    circles = cv2.HoughCircles(equalized, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40, param1=100, param2=30, minRadius=20,
                               maxRadius=60)

    if circles is not None:
        return np.uint16(np.around(circles[0]))

    return []


def classify_coins(circles, mask, avg_radius):
    coins = {0: [], 1: []}  # 0 = 5gr, 1 = 5zł
    for (x, y, r) in circles:
        c_mask = np.zeros_like(mask, dtype=np.uint8)

        # black mask with white circle
        cv2.circle(c_mask, (x, y), r, 255, -1)

        # check overlap, if smaller than 0.5, assume circle is fake
        overlap = cv2.bitwise_and(mask, mask, mask=c_mask)
        total_area = np.count_nonzero(c_mask)
        white_in_mask = np.count_nonzero(overlap)
        ratio = white_in_mask / total_area if total_area > 0 else 0
        if ratio > 0.5:
            if r <= avg_radius:
                coins[0].append((x, y, r))
            else:
                coins[1].append((x, y, r))
    return coins


def draw_coins(image, coins):
    five_gr_color = (255, 0, 0)
    five_zl_color = (0, 0, 255)
    for (x, y, r) in coins[0]:
        cv2.circle(image, (x, y), r, five_gr_color, 3)
    for (x, y, r) in coins[1]:
        cv2.circle(image, (x, y), r, five_zl_color, 3)
    return image, coins


def find_circles(src, blurred, mask):
    circles = detect_circles(blurred)
    if not circles.any():
        return src.copy()

    avg_r = np.mean([c[2] for c in circles])
    print(f"average radius: {avg_r:.2f}")

    coins = classify_coins(circles, mask, avg_r)

    return draw_coins(src.copy(), coins)


# if their coordinates are similar, keep only one
def are_lines_similar(line1, line2, threshold=5):
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    dist1 = np.hypot(x11 - x21, y11 - y21)
    dist2 = np.hypot(x12 - x22, y12 - y22)

    return dist1 < threshold and dist2 < threshold


def get_polygon(lines):
    vertices = []
    for line in lines:
        x1, y1, x2, y2 = line
        vertices.append((x1, y1))
        vertices.append((x2, y2))

    hull = cv2.convexHull(np.array(vertices))

    return hull


def find_lines(src, lower, upper):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, lower, upper, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        100,
        minLineLength=150,
        maxLineGap=150)

    image_l = src.copy()

    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            too_close = any(are_lines_similar((x1, y1, x2, y2), l, threshold=250) for l in filtered_lines)

            if not too_close:
                filtered_lines.append((x1, y1, x2, y2))
                cv2.line(image_l, (x1, y1), (x2, y2), (127, 127, 127), 3)

    polygon = get_polygon(filtered_lines)

    if polygon.size > 0:
        cv2.polylines(image_l, [polygon], isClosed=True, color=(0, 255, 0), thickness=5)
        return image_l, polygon

    return image_l, None


images = []
path_to_images = 'data'
window_name = 'obrazek'


def get_image(index):
    return images[index % len(images)]


def calculate(coins, src, polygon):
    print('='*60)
    poly_mask = np.zeros_like(src, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon], (255, 255, 255))

    poly_area = np.count_nonzero(poly_mask)
    print(f"Total area of the tray: {poly_area:.0f}")

    # 0 -> outside, 1 -> inside
    coins_in_tray = {0: [], 1: []}

    for i, label in zip(range(len(coins)), ['5 groszy', '5 zlotych']):
        print(f"\nDenomination: {label}, Count: {len(coins[i])}")
        area_sum = 0
        for (x, y, r) in coins[i]:
            c_area = np.pi * r ** 2
            area_sum += c_area
            if cv2.pointPolygonTest(polygon, (x, y), False) < 0:
                coins_in_tray[0].append(i)  # Outside tray
            else:
                coins_in_tray[1].append(i)  # Inside tray

        if i == 1:
            print(f"\tTotal area for {label}: {area_sum:.0f} ({(area_sum / poly_area) * 100:.2f}% of the tray area)")
        else:
            print(f"\tTotal area for {label}: {area_sum:.0f}")

    for i, coins_list in coins_in_tray.items():
        five_gr_coins = sum(c == 0 for c in coins_list)
        five_zl_coins = sum(c == 1 for c in coins_list)

        if i == 0:
            print(f"\nValue outside the tray: {five_zl_coins * 5} zlotych, {five_gr_coins * 5} groszy.")
        else:
            print(f"\nValue inside the tray: {five_zl_coins * 5} zlotych, {five_gr_coins * 5} groszy.")


def main():
    global images

    files = [f for f in os.listdir(path_to_images) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    images = [load_image(path_to_images, f) for f in sorted(files)]
    images = [img for img in images if img is not None]

    # i love opencv on windows 🙃
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("hue shift", window_name, 0, 180, nothing)
    cv2.createTrackbar("coins lower", window_name, 0, 180, nothing)
    cv2.createTrackbar("coins upper", window_name, 0, 180, nothing)
    cv2.createTrackbar("blur size", window_name, 1, 25, nothing)
    cv2.createTrackbar("lines lower", window_name, 0, 180, nothing)
    cv2.createTrackbar("lines upper", window_name, 0, 180, nothing)

    previous_values = (-1,) * 6
    index = 0
    image = images[index]

    while True:
        hue = cv2.getTrackbarPos("hue shift", window_name)
        coins_lower = cv2.getTrackbarPos("coins lower", window_name)
        coins_upper = cv2.getTrackbarPos("coins upper", window_name)
        blur = cv2.getTrackbarPos("blur size", window_name)
        lines_lower = cv2.getTrackbarPos("lines lower", window_name)
        lines_upper = cv2.getTrackbarPos("lines upper", window_name)

        current_values = (hue, coins_lower, coins_upper, blur, lines_lower, lines_upper)
        if current_values != previous_values:

            if blur % 2 == 0:
                blur += 1

            hue_shifted = shift_hue_of_image(image, hue)
            gray_blurred = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
            mask = create_mask_from_color(hue_shifted, coins_lower, coins_upper)

            with_lines, polygon = find_lines(image, lines_lower, lines_upper)
            with_lines_and_coins, coins = find_circles(with_lines, gray_blurred, mask)

            if polygon is not None:
                calculate(coins, image, polygon)

            merged = merge_images([hue_shifted, mask, with_lines_and_coins])
            cv2.imshow(window_name, merged)

            previous_values = current_values
        key = cv2.waitKey(1)
        if key == ord(','):
            index += 1
            image = get_image(index)
            previous_values = (-1,) * 6
        elif key == ord('.'):
            index -= 1
            image = get_image(index)
            previous_values = (-1,) * 6
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
