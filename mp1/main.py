import os.path
import cv2
import numpy as np
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
    lower_color = np.array([lower, 50, 50])
    upper_color = np.array([upper, 255, 255])
    return cv2.inRange(hsv, lower_color, upper_color)


def merge_images(images):
    for i, img in enumerate(images):
        if img.ndim != 3:
            images[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    top_row = cv2.hconcat([images[0], images[1]])
    bottom_row = cv2.hconcat([images[2], images[3]])
    return resize_image_to_screen_size(cv2.vconcat([top_row, bottom_row]), 2)


def add_marker(img, mask):
    contours, _ = cv2.findContours(mask, 1, 2)

    if len(contours) == 0:
        return img

    largest_contour = max(contours, key=cv2.contourArea)

    moments = cv2.moments(largest_contour)

    if moments['m00'] == 0:
        return img
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    img_marker = img.copy()
    cv2.drawMarker(img_marker, (cx, cy), color=(255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)
    return img_marker


def add_labels(image, label):
    cv2.putText(image, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)


def process_frame(src, shift, lower, upper, erosion, dilation):
    shifted = shift_hue_of_image(src, shift)
    mask = create_mask_from_color(shifted, lower, upper)
    morphed = morph_image_mask(mask, erosion, dilation)
    marked = add_marker(src, morphed)
    bitwise = cv2.bitwise_and(marked, marked, mask=morphed)

    return shifted, mask, morphed, marked, bitwise


def track_object_in_image(filepath):
    global window_name

    image = cv2.imread(filepath)
    assert image is not None
    cv2.namedWindow(window_name)

    cv2.createTrackbar("hue shift", window_name, 0, 180, nothing)
    cv2.createTrackbar("mask lower", window_name, 0, 180, nothing)
    cv2.createTrackbar("mask upper", window_name, 0, 180, nothing)
    cv2.createTrackbar("blur size", window_name, 0, 25, nothing)
    cv2.createTrackbar("erosion iters", window_name, 0, 25, nothing)
    cv2.createTrackbar("dilation iters", window_name, 0, 25, nothing)

    previous_values = (-1, -1, -1, -1, -1, -1)

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



            marked = add_marker(image, morphed)
            bitwise = cv2.bitwise_and(marked, marked, mask=morphed)
            images = [hue_shifted_img, mask, morphed, bitwise]
            labels = ["hue shifted:", "hue mask:", "morphed mask:", "result:"]

            for img, label in zip(images, labels):
                add_labels(img, label)

            merged = merge_images(images)
            cv2.imshow(window_name, merged)
            previous_values = current_values

        key = cv2.waitKey(16)
        if key == 27:
            cv2.destroyAllWindows()
            break


def track_object_in_video(filepath):
    global window_name

    capture = cv2.VideoCapture()
    capture.open(filepath)

    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_width = int(capture.get(3))
    frame_height = int(capture.get(4))
    size = (frame_width, frame_height)

    src_name = os.path.basename(filepath).split(".")[0]

    output_dir = f'output/{src_name}/'
    os.makedirs(output_dir, exist_ok=True)

    tracked_result = cv2.VideoWriter(output_dir + src_name + '_tracked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
    bitwise_result = cv2.VideoWriter(output_dir + src_name + '_bitwise.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    cv2.namedWindow(window_name)

    cv2.createTrackbar("frame", window_name, 0, int(total_frames - 1), nothing)
    cv2.createTrackbar("hue shift", window_name, 0, 180, nothing)
    cv2.createTrackbar("mask lower", window_name, 0, 180, nothing)
    cv2.createTrackbar("mask upper", window_name, 0, 180, nothing)
    cv2.createTrackbar("dilation iters", window_name, 0, 25, nothing)
    cv2.createTrackbar("erosion iters", window_name, 0, 25, nothing)
    cv2.createTrackbar("start processing", window_name, 0, 1, nothing)

    previous_values = (-1, -1, -1, -1, -1, -1)

    success, image = capture.read()

    while True:
        frame_num = cv2.getTrackbarPos("frame", window_name)
        hue = cv2.getTrackbarPos("hue shift", window_name)
        lower = cv2.getTrackbarPos("mask lower", window_name)
        upper = cv2.getTrackbarPos("mask upper", window_name)
        erosion = cv2.getTrackbarPos("erosion iters", window_name)
        dilation = cv2.getTrackbarPos("dilation iters", window_name)
        start = cv2.getTrackbarPos("start processing", window_name)

        shifted, mask, morphed, marked, bitwise = process_frame(image, hue, lower, upper, erosion, dilation)

        current_values = (frame_num, hue, lower, upper, erosion, dilation)
        if current_values != previous_values:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, image = capture.read()

            images = [shifted, mask, morphed, bitwise]
            labels = ["hue shifted:", "hue mask:", "morphed mask:", "result:"]

            for img, label in zip(images, labels):
                add_labels(img, label)

            merged = merge_images(images)
            cv2.imshow(window_name, merged)
            previous_values = current_values

        if start == 1:
            print(f"Processing video: {src_name}...")
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracked_result.write(marked)
            bitwise_result.write(bitwise)
            break

        key = cv2.waitKey(16)
        if key == 27:
            cv2.destroyAllWindows()
            break

    for i in range(1, int(total_frames)):
        success, image = capture.read()

        if not success:
            break

        shifted, mask, morphed, marked, bitwise = process_frame(image, hue, lower, upper, erosion, dilation)

        tracked_result.write(marked)
        bitwise_result.write(bitwise)
    print(f"Video {src_name} processed and saved to /output/{src_name}/")


global window_name


def main():
    global window_name

    window_name = 'Object tracking'

    process_video = True
    if process_video:
        track_object_in_video('data/movingball.mp4')
    else:
        track_object_in_image('data/ball.png')


if __name__ == '__main__':
    main()
