import os
import argparse
import cv2
import random
import numpy as np
from datagenutils import *

def writeCircleImage(box, circle, image, imfname, csvoutf, debugmode=False):
    print(f'{box}, {circle}')
    (h, w) = image.shape[:2]
    # for i in range(10):
    box = randomizeBox(box, (h,w), 200)
    box = list(box)
    circle = list(circle)
    print(f'{box}, {circle}')
    # if not validateBox(box, (h,w)):
    #     continue
    bc = (int((box[2]+box[0])/2), int((box[3]+box[1])/2))
    box = ensureBoxWithinImage(box, (h,w))
    
    circle = (circle[0]-box[0], circle[1]-box[1], circle[2])
    
    (h, w) = map(lambda x: int(x/2), (h, w))
    cimage = np.zeros((h, w, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    cimage[:] = (255, 255, 255)
    print(f'{box}, {circle}')
    cimage[0:(box[3]-box[1]), 0:(box[2]-box[0])] = image[box[1]:box[3], box[0]:box[2]].copy()
    cv2.imwrite(imfname, cimage)
    csvoutf.write(f'{imfname},{circle[0]},{circle[1]},{circle[2]}\n')
    if debugmode is True:
        dispimg = cimage.copy()
        cv2.circle(dispimg, (circle[0], circle[1]), circle[2], (255,0,0), 2)
        dispimg = cv2.resize(dispimg, (600,600))
        cv2.imshow("Circle", dispimg)
        key = cv2.waitKey(0)
        if key==27:
            exit()
    return

parser = argparse.ArgumentParser()
parser.add_argument("indir", help="Input directory")
parser.add_argument("outdir", help="Output directory")
args = parser.parse_args()

print(f'{args.indir},{args.outdir}')

cboutfile = open(os.path.sep.join([args.outdir, 'cb2c.csv']), 'w')

rows = open(os.path.sep.join(
    [args.indir, 'master.csv'])).read().strip().split("\n")

print(f'{len(rows)} rows')
count = 0

for row in rows:
    row = row.split(',')

    (filename, c1x, c1y, c1r, c2x, c2y, c2r) = row
    (c1x, c1y, c1r, c2x, c2y, c2r) = map(lambda x: int(x), (c1x, c1y, c1r, c2x, c2y, c2r))
    imagePath = os.path.sep.join([args.indir, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    if h < 1000:
        cv2.imshow(filename, image)
        cv2.waitKey(0)
    
    slices = 16
    angle = 0.
    inc = 360. / float(slices)
    while angle < 360:
        c1 = (c1x, c1y, c1r)
        c2 = (c2x, c2y, c2r)
        rotatedimg, c1, c2 = rotateImageAndCircles(image, c1, c2, angle)
        c1box = (c1[0]-c1[2], c1[1]-c1[2], c1[0]+c1[2], c1[1]+c1[2])
        c2box = (c2[0]-c2[2], c2[1]-c2[2], c2[0]+c2[2], c2[1]+c2[2])
        
        c1fname = os.path.sep.join([args.outdir, 'cbox', 'images', f'{filename[:-4]}{angle}c1.jpg'])
        c2fname = os.path.sep.join([args.outdir, 'cbox', 'images', f'{filename[:-4]}{angle}c2.jpg'])
        # print(f'{filename}\n{c1fname}')
        # input()
        writeCircleImage(c1box, c1, rotatedimg, c1fname, cboutfile)
        writeCircleImage(c2box, c2, rotatedimg, c2fname, cboutfile)

        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
        angle += inc
        count += 2

print(f'{count} processed')
cboutfile.close()