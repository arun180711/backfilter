from cv2 import cv2
import cvzone
import numpy as np


def SnapFilters():
    filter_type = int(input("Enter any number to get into the sticker : "))
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    overlay = cv2.imread('doggie.png', cv2.IMREAD_UNCHANGED)
    overlay1 = cv2.imread('native.png', cv2.IMREAD_UNCHANGED)
    overlay2 = cv2.imread('pirate.png', cv2.IMREAD_UNCHANGED)
    overlay3 = cv2.imread('star.png', cv2.IMREAD_UNCHANGED)
    overlay4 = cv2.imread('sunglass.png', cv2.IMREAD_UNCHANGED)
    overlay5 = cv2.imread('demon.png', cv2.IMREAD_UNCHANGED)
    overlay6 = cv2.imread('neonglass.png', cv2.IMREAD_UNCHANGED)

    def StarGlasses(videos):
        overlay_resize = cv2.resize(overlay3, (int(w * 1), int(h * 1)))
        frame1 = cvzone.overlayPNG(videos, overlay_resize, [x + 5, y - 17])
        cv2.imshow('StarGlasses', frame1)
        cv2.imwrite('selfie.png', frame1)

    def NormGlass(videos):
        overlay_resize = cv2.resize(overlay4, (int(w * 1), int(h * 1)))
        sun_glass = cvzone.overlayPNG(videos, overlay_resize, [x + 5, y - 17])
        cv2.imshow('SunGlasses', sun_glass)
        cv2.imwrite('selfie.png', sun_glass)

    def pirate(videos):
        overlay_resize = cv2.resize(overlay2, (int(w * 1.5), int(h * 1.5)))
        pirate1 = cvzone.overlayPNG(videos, overlay_resize, [x - 39, y - 70])
        cv2.imshow('Pirate', pirate1)
        cv2.imwrite('selfie.png', pirate1)

    def native(videos):
        overlay_resize = cv2.resize(overlay1, (int(w * 2), int(h * 2)))
        nativepic = cvzone.overlayPNG(videos, overlay_resize, [x - 70, y - 88])
        cv2.imshow('Native', nativepic)
        cv2.imwrite('selfie.png', nativepic)

    def doggie(videos):
        overlay_resize = cv2.resize(overlay, (int(w * 2), int(h * 1.7)))
        doggiepic = cvzone.overlayPNG(videos, overlay_resize, [x - 50, y - 50])
        cv2.imshow('Doggie', doggiepic)
        cv2.imwrite('selfie.png', doggiepic)

    def neonglass(videos):
        overlay_resize = cv2.resize(overlay6, (int(w * 1), int(h * 1)))
        neonpic = cvzone.overlayPNG(videos, overlay_resize, [x, y - 18])
        cv2.imshow('Doggie', neonpic)
        cv2.imwrite('selfie.png', neonpic)

    def demon(videos):
        overlay_resize = cv2.resize(overlay5, (int(w * 1.5), int(h * 1.5)))
        demonicpic = cvzone.overlayPNG(videos, overlay_resize, [x - 50, y - 50])
        cv2.imshow('Demon', demonicpic)
        cv2.imwrite('selfie.png', demonicpic)

    def normal(videos):
        normal_view = videos
        cv2.imshow('filter', normal_view)

    def greyscale(video4):
        grey_scale = cv2.cvtColor(video4, cv2.COLOR_BGR2GRAY)
        cv2.imshow('filter', grey_scale)

    def violet_scale_filter(video):
        violet_scale = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        cv2.imshow('filter', violet_scale)

    def bright(video1, brightness_level):
        brightness = cv2.convertScaleAbs(video1, beta=brightness_level)
        cv2.imshow('FILTER', brightness)

    def negative(video1, brightness_level):
        negative_video = cv2.convertScaleAbs(video1, beta=brightness_level)
        cv2.imshow('FILTER', negative_video)

    def sharpen(video4):
        kernel = np.array([[-1, -1, -1], [-1, 9.5, -1], [-1, -1, -1]])
        img_sharpen = cv2.filter2D(video4, -1, kernel)
        cv2.imshow('filter', img_sharpen)

    def sepia(video5):
        video_sepia = np.array(video5, dtype=np.float64)
        video_sepia = cv2.transform(video_sepia,
                                    np.matrix([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]))
        video_sepia[np.where(video_sepia > 255)] = 255
        video_sepia = np.array(video_sepia, dtype=np.uint8)
        cv2.imshow('filter', video_sepia)

    def _HDR_format(video6):
        hdr_filter = cv2.detailEnhance(video6, sigma_s=12, sigma_r=0.15)
        cv2.imshow('filter', hdr_filter)

    def pencil_sketch(video3):
        sk_gray, sk_color = cv2.pencilSketch(video3, sigma_s=3000, sigma_r=0.01, shade_factor=0.1)
        cv2.imshow('FILTER', sk_gray)

    if filter_type <= 14:
        while True:
            _, frame = cap.read()
            if filter_type >= 9:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray)
                face = face_cascade.detectMultiScale(gray, 1.3, 5)
                for x, y, w, h in faces:
                    if filter_type == 9:
                        StarGlasses(frame)
                    elif filter_type == 10:
                        NormGlass(frame)
                    elif filter_type == 11:
                        pirate(frame)
                    elif filter_type == 12:
                        native(frame)
                    elif filter_type == 13:
                        doggie(frame)
                    elif filter_type == 14:
                        neonglass(frame)
                    face_roi = frame[y:y + h, x:x + w]
                    gray_roi = gray[y:y + h, x:x + w]
                    smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)
                    for x1, y1, w1, h1 in smile:
                        print("")
            else:
                if filter_type == 0:
                    normal(frame)
                elif filter_type == 1:
                    greyscale(frame)
                elif filter_type == 2:
                    violet_scale_filter(frame)
                elif filter_type == 3:
                    bright(frame, 70)
                elif filter_type == 4:
                    negative(frame, -70)
                elif filter_type == 5:
                    sharpen(frame)
                elif filter_type == 6:
                    sepia(frame)
                elif filter_type == 7:
                    _HDR_format(frame)
                elif filter_type == 8:
                    pencil_sketch(frame)

            if cv2.waitKey(10) == ord('p'):
                break
    else:
        print("No Sticker Found")
