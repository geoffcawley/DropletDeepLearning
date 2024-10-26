import math
import cv2
import argparse
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import config
import os
import numpy as np
from pathlib import Path
import glob
from trainingutils import getBoundingSquare, getcb2cImg, getModelInput

# 306 pixels = 50 microns
# 6.12 pixels per micron
# 0.163399 microns per pixel
g_pixelsPerUnit = 6.12
g_frameskip = 1
g_framesPerSecond = 26.
g_currentFrame = 1
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', "--input", required=True,
    help="path to input video(s)")
ap.add_argument("-o", "--output", required=False,
	help="path to output csv(s)")
ap.add_argument('-v', '--video', required=False,
    help="path to output video(s)")
ap.add_argument('-s', '--frameskip', required=False,
    help='number of frames to skip in between each measurement')
args = vars(ap.parse_args())

if args['frameskip'] is not None:
    g_frameskip = int(args['frameskip'])

# load models
print("[INFO] loading models...")
bbmodel = load_model(config.BB_MODEL_PATH)
# c1model = load_model(config.C1_MODEL_PATH)
# c2model = load_model(config.C2_MODEL_PATH)
c1boxmodel = load_model(config.C1_BOX_MODEL_PATH)
c2boxmodel = load_model(config.C2_BOX_MODEL_PATH)
b2cmodel = load_model(config.DUAL_B2C_MODEL_PATH)

# Output headers in order:
# Time Stamp, Adjusted Time, 
# Droplet 1 Radius, Droplet 1 Volume, Droplet 2 Radius, Droplet 2 Volume,
# Total Volume, DIB Radius, Contact Angle, Radial Distance
# Math from:
# Droplet Shape Analysis and Permeability Studies in Droplet Lipid Bilayers by Sanhita S. Dixit et. al.
def processframe(c1, c2):
    rDistance = math.sqrt((c2[0]-c1[0])**2 + (c2[1]-c1[1])**2)

    # Two unconnected droplets; do not record
    if rDistance ** 2 >= (c1[2] + c2[2]) ** 2:
        return
    
    # print([c1, c2])
    
    # convert from pixels to microns and compensate for top-down 
    # perspective of microscope
    r1 = float(c1[2]) / g_pixelsPerUnit
    r2 = float(c2[2]) / g_pixelsPerUnit
    lf = rDistance / g_pixelsPerUnit
    lr = math.sqrt((r2-r1)**2. + lf**2.)

    thetab = math.acos(
        (lr**2. - (r1**2. + r2**2.))
        / (2. * r1 * r2)
    )

    rdib = (r1 * r2 * math.sin(thetab)) / lr

    # 1/2 of DIB angle is what chemists use
    theta_degrees = (180. * thetab) / math.pi
    theta_degrees /= 2.

    # dome heights for volume of sphere - dome
    a = ((r1**2. - r2**2.) + lr**2.) / (2. * lr)
    b = lr - a
    c1h = r1 - a
    c2h = r2 - b

    v1 = (4. * math.pi * r1**3.) / 3.
    v1 -= (math.pi * c1h * (3. * rdib**2. + c1h**2.)) / 6.

    v2 = (4. * math.pi * r2**3.) / 3.
    v2 -= (math.pi * c2h * (3. * rdib**2. + c2h**2.)) / 6.

    tv = v1 + v2

    # print([r1, v1, r2, v2, tv, rdib, theta_degrees, lr])
    # input()
    return r1, v1, r2, v2, tv, rdib, theta_degrees, lr

def predictVideo(vpath, csvpath, ovpath):
    # open input video
    cap = cv2.VideoCapture(vpath)
    vname = os.path.basename(vpath)
    g_framesPerSecond = cap.get(cv2.CAP_PROP_FPS)
    g_currentFrame = 1
    outvname = os.path.join(ovpath, Path(vpath).stem + '.avi')
    csvname = os.path.join(csvpath, Path(vpath).stem + '.csv')
    print(f'csv out: {csvname}\nvideo out: {outvname}\n')
    print(f"{vname} at {g_framesPerSecond} FPS")

    outfile = open(csvname, "w")
    header = ("Time Stamp,Droplet 1 Radius,Droplet 1 Volume,"
              "Droplet 2 Radius,Droplet 2 Volume,Total Volume,"
              "DIB Radius,Contact Angle,Radial Distance,\n")
    outfile.write(header)

    # Initialize video writer object
    frame_size = (600,600)
    output_video = cv2.VideoWriter(outvname, cv2.VideoWriter_fourcc(*'XVID'), 20, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        # cv2.imshow("Input", imutils.resize(frame, width=600))
        # cv2.waitKey(0)
        # image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        if g_currentFrame % g_frameskip != 0:
            g_currentFrame += 1
            continue
        try:
            image = cv2.GaussianBlur(frame, (5, 5), 0)
        except:
            break
        # threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        bb_model_input = getModelInput(image, 224)

        # make circle predictions on the input image
        preds = bbmodel.predict(bb_model_input, verbose=0)[0]
        (bbx1, bby1, bbx2, bby2) = preds
        
        (h, w) = image.shape[:2]
        # scale the predicted circle coordinates based on the image
        # dimensions
        bbx1 = int(bbx1 * w)
        bbx2 = int(bbx2 * w)
        bby1 = int(bby1 * h)
        bby2 = int(bby2 * h)
        
        bsx1, bsy1, bsx2, bsy2 = getBoundingSquare((bbx1, bby1, bbx2, bby2), h, w)
        # circleimage = image.copy()
        # circleimage = circleimage[bsy1:bsy2, bsx1:bsx2]
        circleimage = np.zeros((h,w,3), np.uint8)
        circleimage[:] = (255,255,255)
        (bh, bw) = (bsy2-bsy1, bsx2-bsx1)
        cibox = [(w/2)-(bw/2), (h/2)-(bh/2), (w/2)+(bw/2), (h/2)+(bh/2)]
        cibox = list(map(lambda x: int(x), cibox))
        circleimage[cibox[1]:cibox[3],cibox[0]:cibox[2]] = image[bsy1:bsy2,bsx1:bsx2].copy()
        # show bounding box image
        # print((bbx1, bby1, bbx2, bby2))
        # cv2.imshow("circle image", circleimage)
        # cv2.waitKey(0)
        
        c_model_input = getModelInput(circleimage, 224)
        # make circle bounding box predictions on the input image
        preds = c1boxmodel.predict(c_model_input, verbose=0)[0]
        (c1x1, c1y1, c1x2, c1y2) = preds
        
        preds = c2boxmodel.predict(c_model_input, verbose=0)[0]
        (c2x1, c2y1, c2x2, c2y2) = preds
        
        (c1x1, c1y1, c1x2, c1y2, c2x1, c2y1, c2x2, c2y2) = map(lambda x: int(x*w), (c1x1, c1y1, c1x2, c1y2, c2x1, c2y1, c2x2, c2y2))
        # not sure about next two lines
        bsw = int((bsx2-bsx1)/2)
        (cox, coy) = ((w/2)-bsw, (h/2-bsw))
        (tvx, tvy) = (bsx1-cox, bsy1-coy)
        (c1x1, c1x2, c2x1, c2x2) = map(lambda x: x+tvx, (c1x1, c1x2, c2x1, c2x2))
        (c1y1, c1y2, c2y1, c2y2) = map(lambda y: y+tvy, (c1y1, c1y2, c2y1, c2y2))
        # (c1x1, c1x2, c2x1, c2x2) = map(lambda x: x+bsx1, (c1x1, c1x2, c2x1, c2x2))
        # (c1y1, c1y2, c2y1, c2y2) = map(lambda y: y+bsy1, (c1y1, c1y2, c2y1, c2y2))
        
        c1bsq = getBoundingSquare((c1x1, c1y1, c1x2, c1y2), h, w)
        c2bsq = getBoundingSquare((c2x1, c2y1, c2x2, c2y2), h, w)
        c1bsqimg = getcb2cImg(image, c1bsq)
        # cv2.imshow("c1bsqimg", c1bsqimg)
        c2bsqimg = getcb2cImg(image, c2bsq)
        c1b2c_mi = getModelInput(c1bsqimg, 224)
        c2b2c_mi = getModelInput(c2bsqimg, 224)
        
        c1 = b2cmodel.predict(c1b2c_mi, verbose=0)[0]
        c2 = b2cmodel.predict(c2b2c_mi, verbose=0)[0]
        
        print(c1)
        c1 = list(map(lambda x: int(x*(w/2)), c1))
        print(c1)
        c2 = list(map(lambda x: int(x*(w/2)), c2))
        cv2.circle(c1bsqimg, (c1[0],c1[1]), c1[2], (255,0,0), 2)
        cv2.imshow("c1bsqimg", c1bsqimg)
        c1[0] += c1x1
        c1[1] += c1y1
        c2[0] += c2x1
        c2[1] += c2y1
        (c1x, c1y, c1r) = c1
        (c2x, c2y, c2r) = c2
        
        # print([c1,c2])
        timestamp = float(g_currentFrame) / g_framesPerSecond
        try:
            (r1, v1, r2, v2, tv, rdib, theta_degrees, lr) = processframe(c1, c2)
        except:
            g_currentFrame += 1
            continue
        # print(f'{timestamp},{r1},{v1},{r2},{v2},{tv},{rdib},{theta_degrees},{lr}\n')
        outfile.write(f'{timestamp},{r1},{v1},{r2},{v2},{tv},{rdib},{theta_degrees},{lr}\n')
        
        # show the output image
        dispimg = cv2.resize(image, (600,600))
        
        (c1x, c1y, c1r, c2x, c2y, c2r, c1x1, c1y1, c1x2, c1y2, c2x1, c2y1, c2x2, c2y2, bbx1, bby1, bbx2, bby2, bsx1, bsy1, bsx2, bsy2) \
            = map(lambda x: int(x*(600/w)), (c1x, c1y, c1r, c2x, c2y, c2r, c1x1, c1y1, c1x2, c1y2, c2x1, c2y1, c2x2, c2y2, bbx1, bby1, bbx2, bby2, bsx1, bsy1, bsx2, bsy2))
        
        cv2.rectangle(dispimg, (bbx1, bby1), (bbx2, bby2),
            (0, 255, 0))
        cv2.rectangle(dispimg, (bsx1, bsy1), (bsx2, bsy2), (0,255,0))
        cv2.rectangle(dispimg, (c1x1, c1y1), (c1x2, c1y2), (255,0,0))
        cv2.rectangle(dispimg, (c2x1, c2y1), (c2x2, c2y2), (255,0,0))
        cv2.circle(dispimg, (c1x, c1y), c1r,
            (0, 255, 0))
        cv2.circle(dispimg, (c2x, c2y), c2r,
            (0, 0, 255))
        cv2.imshow("Output", dispimg)
        # if args.output is not None:
        output_video.write(dispimg)
        key = cv2.waitKey(10)
        if key == 27:
            exit()
        g_currentFrame += 1
    output_video.release()
    outfile.close()

if args['video'] is None:
    args['video'] = os.path.sep.join([os.getcwd(), 'output', 'videos'])
if args['output'] is None:
    args['output'] = os.path.sep.join([os.getcwd(), 'output', 'csv'])
    
if not os.path.exists(args['video']):
    os.makedirs(args['video'])
if not os.path.exists(args['output']):
    os.makedirs(args['output'])
    
video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.wmv', '*.flv', '*.webm']
invideopaths = []
if os.path.isdir(args['input']):
    for ext in video_extensions:
        pattern = os.path.join(args['input'], ext)
        invideopaths.extend(glob.glob(pattern))
else:
    invideopaths = [args['input']]
    
print(f'Reading {len(invideopaths)} videos')

for vpath in invideopaths:
    predictVideo(vpath, args['output'], args['video'])