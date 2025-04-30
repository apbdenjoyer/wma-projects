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


def norm_size(src):
    h, w = src.shape[:2]
    if h > w:
        if h > 800:
            s = (1 - (800 / h)) * (-1)
            w = w + int(w * (s))
            h = h + int(h * (s))
            normed = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        if w > 800:
            s = (1 - (800 / w)) * (-1)
            w = w + int(w * (s))
            h = h + int(h * (s))
            normed = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    return normed


def shift_hue_of_image(src):
    global window_name
    hue = cv2.getTrackbarPos("hue shift", window_name)

    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def create_mask_from_color(src, lower, upper):
    global window_name
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


def get_image(index):
    keys = sorted(images.keys())
    return images[keys[index % len(keys)]]



def cut(src):
    x = cv2.getTrackbarPos('bound lower', window_name) * 2
    y = cv2.getTrackbarPos('bound upper', window_name) * 2
    ksize = cv2.getTrackbarPos('k size', window_name) * 2

    if ksize < 10:
        ksize = 10

    cut_img = src[x: x + int(ksize / 2), y: y + ksize]
    return cut_img


def get_sift_keys_for_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    low = cv2.getTrackbarPos('bound lower', window_name)
    high = cv2.getTrackbarPos('bound upper', window_name)
    ksize = cv2.getTrackbarPos("k size", window_name)
    if ksize % 2 == 0:
        ksize += 1

    blur = cv2.medianBlur(gray, ksize=ksize)
    mask = create_mask_from_color(img, low, high)

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(blur, mask)

    return keypoints, descriptors

def get_orb_keys_for_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    low = cv2.getTrackbarPos('bound lower', window_name)
    high = cv2.getTrackbarPos('bound upper', window_name)
    ksize = cv2.getTrackbarPos("k size", window_name)

    if ksize % 2 == 0:
        ksize += 1

    blur = cv2.medianBlur(gray, ksize=ksize)
    mask = create_mask_from_color(img, low, high)

    orb = cv2.ORB_create()

    keypoints, descriptors = orb.detectAndCompute(blur, mask)

    return keypoints, descriptors


'''
    perform sift match between img1 (source image) and img2 (video frame)
        >map[image] - sift matches to frame
        >pick image with the highest match count
        >generate new frame with matches shown
'''
def sift_match(frame):
    global images, images_k_d

    frame_keypoints, frame_descriptors = get_sift_keys_for_image(frame)

    images_matches = {}

    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    for fname, (keypoints, descriptors) in images_k_d.items():
        matches = matcher.match(descriptors, frame_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        print(f"\t{fname}: {len(matches)} matches")

        images_matches[fname] = (matches, keypoints)

    best_fname = max(images_matches, key=lambda fname: len(images_matches[fname][0]))
    best_matches, best_keypoints = images_matches[best_fname]

    best_image = images[best_fname]

    matched_img = cv2.drawMatches(
        best_image, best_keypoints, frame, frame_keypoints, best_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return matched_img

def orb_match(frame):
    global images, images_k_d

    orb = cv2.ORB_create()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame_gray, None)

    images_matches = {}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for fname, (keypoints, descriptors) in images_k_d.items():
        matches = matcher.match(descriptors, frame_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        print(f"\t{fname}: {len(matches)} matches")

        images_matches[fname] = (matches, keypoints)

    best_fname = max(images_matches, key=lambda fname: len(images_matches[fname][0]))
    best_matches, best_keypoints = images_matches[best_fname]

    best_image = images[best_fname]

    matched_img = cv2.drawMatches(
        best_image, best_keypoints, frame, frame_keypoints, best_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return matched_img


def compute_k_d_for_images():
    global images_k_d
    for fname, img in images.items():
        keypoints, descriptors = get_sift_keys_for_image(img)
        images_k_d[fname] = (keypoints, descriptors)



images = {}
path_to_images = 'data/img'
path_to_video = 'data/video'
window_name = 'wma mp3'
images_k_d = {}
matched_frames = []


def main():
    global images

    feature_type='sift'     #or 'orb'


    files = [f for f in os.listdir(path_to_images) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    images = {f: load_image(path_to_images, f) for f in sorted(files)}
    images = {f: img for f, img in images.items() if img is not None}

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cv2.createTrackbar("hue shift", window_name, 0, 180, nothing)
    cv2.createTrackbar("bound lower", window_name, 0, 180, nothing)
    cv2.createTrackbar("bound upper", window_name, 0, 180, nothing)
    cv2.createTrackbar("k size", window_name, 1, 50, nothing)
    cv2.createTrackbar("start", window_name, 0, 1, nothing)

    index = 0
    image = get_image(index)

    vid_path = os.path.join(path_to_video, 'video_10s.mp4')

    cap = cv2.VideoCapture(vid_path)
    cv2.namedWindow(window_name)

    while True:
        start = cv2.getTrackbarPos("start", window_name)

        if start == 1:
            compute_k_d_for_images()
            print(f"Processing video using {feature_type.upper()}...")

            matched_frames.clear()

            src_name = os.path.basename(vid_path).split(".")[0]


            match_function = sift_match if feature_type == "sift" else orb_match

            count = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                print(f"Frame {count}:")
                matched_img = match_function(frame)
                matched_img_resized = norm_size(matched_img)
                matched_frames.append(matched_img_resized)
                count += 1

            cap.release()
            cap = cv2.VideoCapture(vid_path)
            print(f"Video {src_name} processed!")

            for frame in matched_frames:
                cv2.imshow("Video with matching", frame)
                if cv2.waitKey(30) == 27:
                    break

            cv2.setTrackbarPos("start", window_name, 0)

        hue_shifted = shift_hue_of_image(image)

        low = cv2.getTrackbarPos('bound lower', window_name)
        high = cv2.getTrackbarPos('bound upper', window_name)
        ksize = cv2.getTrackbarPos("k size", window_name)
        if ksize % 2 == 0:
            ksize += 1

        blur = cv2.medianBlur(hue_shifted, ksize=ksize)
        mask = create_mask_from_color(blur, low, high)


        merged = merge_images([blur, mask])

        cv2.imshow(window_name, merged)

        key = cv2.waitKey(1)
        if key == ord(','):
            index += 1
            image = get_image(index)
        elif key == ord('.'):
            index -= 1
            image = get_image(index)
        elif key == 27:
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
