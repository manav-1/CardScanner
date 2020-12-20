import json 
from pymongo import MongoClient
import flask
from waitress import serve
import werkzeug
import pytesseract
from flask.json import jsonify
from flask import render_template
import cv2
import numpy as np
from PIL import Image
import imutils
from skimage.filters import threshold_local
import re
import json 
from pymongo import MongoClient 

emaildict = dict()
phonedict = dict()
# Making Connection 
myclient = MongoClient("mongodb://localhost:27017/")  
db = myclient["Manav"]
Collection = db["CardScanner"]

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
 
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped

def grayedImage(imagefile):
    image = imagefile
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)
    
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # show the original image and the edge detected image
    return edged, ratio

def findingcontours(grayimg):
    cnts = cv2.findContours(grayimg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        # our approximated contour should have four points
        if len(approx) == 4:
            screenCnt = approx
            break
 
    return screenCnt

def addtodatabase(numbers, emails):
    try:
        f = open("emails.json", "w")
        f.write("{ }")
        f.close()
        if numbers == [] and emails == []:
            return 0;
        for number in numbers:
            phonedict["Mobile Number"] = number
            with open("emails.json", "r+") as file:
                data = json.load(file)
                data.update(phonedict)
                file.seek(0)
                json.dump(data,file)
        for email in emails:
            emaildict["Email"] = email
            with open("emails.json", "r+") as file:
                data = json.load(file)
                data.update(emaildict)
                file.seek(0)
                json.dump(data,file)
        with open("emails.json") as file:
            file_data = json.load(file)
        if isinstance(file_data, list): 
            Collection.insert_many(file_data)   
        else: 
            Collection.insert_one(file_data)
        f = open("emails.json", "w")
        f.write("{ }")
        f.close()

        return 1
    except:
        return 0


def trnsform(orig, screenCnt, ratio):
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
 
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset = 10, method = "gaussian")
    warped = (warped > T).astype("uint8") * 255
    # show the scanned image and save one copy in out folder
    imS = cv2.resize(warped, (650, 650))
    return imS
    cv2.imwrite('out/'+'Output Image.PNG', imS)

def tesseractfunc(img):
    return pytesseract.image_to_string(img)


app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    filestr =  flask.request.files["img"].read()
    npimg  = np.frombuffer(filestr, np.uint8)
    imagefile = cv2.imdecode(npimg , cv2.IMREAD_COLOR)
    print("Now the program is starting")
    # grayed, ratio = grayedImage(imagefile)
    # contours= findingcontours(grayed)
    # transforming = trnsform(imagefile,contours, ratio)

    output=tesseractfunc(imagefile)
    #regular expression to find emails
    emails = re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", output)
    #regular expression to find phone numbers
    numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', output)

    print(numbers)
    print(emails)

    x = addtodatabase(numbers,emails)
    
    if (x==1):
        return render_template("correct.html")
    else:
        return render_template("wrong.html")


app.run(host="127.0.0.1" , port=5000, debug=True)
