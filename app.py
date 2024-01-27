from flask import Flask, render_template, request, send_from_directory, send_file, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import pytesseract
from gtts import gTTS
import cv2
import imutils
import numpy as np
import os
import shutil

app = Flask(__name__, static_url_path="", static_folder="static")

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def warp_images(img1, img2, H):
    if img1 is not None and img2 is not None:
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img

def binarize(img):
    if img is not None:
        img = cv2.medianBlur(img, 5)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img

def stitch(img1, img2, min_match_count):
    sift = cv2.SIFT_create()

    if img1 is None or img1.dtype != np.uint8:
        print("Invalid image 1.")
        return None

    if img2 is None or img2.dtype != np.uint8:
        print("Invalid image 2.")
        return None
    
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m1 for m1, m2 in matches if m1.distance < 0.7 * m2.distance]

    if len(good_matches) > min_match_count:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        result = warp_images(img2, img1, M)
        return result
    else:
        print("We don't have enough number of matches between the two images.")
        print("Found only %d matches. We need at least %d matches." % (len(good_matches), min_match_count))

def frame_capture(filename):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()

    count = 0
    total = 1
    print('\n\n')

    images_directory = 'images'
    if not os.path.exists(images_directory):
        os.makedirs(images_directory)
    
    while success:
        success, image = vidcap.read()
        if success and count % 80 == 0:
            cv2.imwrite("images/%d.jpg" % total, image) 
            print('Capturing frame ' + str(total))
            total += 1
        if cv2.waitKey(10) == 27:  
            break
        count += 1

    print('\n\n')
    return total

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        total = frame_capture('uploads/' + file.filename)

        min_match_count = 7
        img1 = cv2.imread('images/1.jpg', 0)
        img1 = binarize(img1)
        if img1 is not None:
            img1 = imutils.resize(img1, width=1000)

        for i in range(2, total):
            print('Stitching image ' + str(i))
            img2 = cv2.imread('images/' + str(i) + '.jpg', 0)
            img2 = binarize(img2)
            if img1 is not None:
                img2 = imutils.resize(img2, width=1000)
            img1 = stitch(img1, img2, min_match_count)

        if img1 is not None :
            cv2.imwrite('images/0.jpg', img1)
        else:
            print("Error: Unable to write image. img1 is empty or None.")
        
        if img1 is not None:
            cv2.imwrite('static/0.jpg', img1)  
        else:
            print("Error: img1 is None.")

        print('\n\nConverting Image to Text')
        string = pytesseract.image_to_string(Image.open('images/0.jpg'))
        print('\n\nOCR OUTPUT\n\n' + string + '\n\n')

        with open("static/test.txt", "w") as f:
            f.write(string)

        string = '"{}"'.format(string)
        print('Converting Text to Speech\n\n')
        tts = gTTS(text=string, lang='en')
        tts.save("static/tts.mp3")

        return render_template('main.html', results_available=True, video_preview=True, video_src=url_for('uploaded_file', filename=file.filename))
    else:
        return render_template('main.html', results_available=False)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/all')
def download_all():
    shutil.make_archive('results', 'zip', 'static')
    return send_file('results.zip', as_attachment=True)

@app.route('/download/text')
def download_text():
    return send_from_directory('static', 'test.txt', as_attachment=True)

@app.route('/download/audio')
def download_audio():
    return send_from_directory('static', 'tts.mp3', as_attachment=True)

if __name__ == '__main__':
    app.run(host="127.0.0.3", port=3000, debug=True)
