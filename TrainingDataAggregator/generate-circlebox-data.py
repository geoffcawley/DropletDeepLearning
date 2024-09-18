import os
import argparse
import cv2
import random
import numpy as np
from datagenutils import *

def writeCircleBoxImage(box, c1b, c2b, image, imfname, c1csv, c2csv, debugmode=False):
    print(f'{box}, {c1b}')
    (h, w) = image.shape[:2]
    # for i in range(10):
    box = randomizeBox(box, (h,w), 300)
    box = list(box)
    c1b = list(c1b)
    c2b = list(c2b)
    print(f'{box}, {c1b}, {c2b}')
    # if not validateBox(box, (h,w)):
    #     continue
    bc = (int((box[2]+box[0])/2), int((box[3]+box[1])/2))
    box = ensureBoxWithinImage(box, (h,w))

    # Forces the bounding box to be a square, necessary to save square image for model input        
    box = list(getBoundingSquare(box, h, w))
    c1b[0] -= box[0]
    c1b[2] -= box[0]
    c2b[0] -= box[0]
    c2b[2] -= box[0]
    c1b[1] -= box[1]
    c1b[3] -= box[1]
    c2b[1] -= box[1]
    c2b[3] -= box[1]
        
    # circle box coordinates should be relative to original image dimensions, not bounding box dimensions
    # Circle box coords are relative to bounding box image at (0,0) and image dimensions 1952x1952
    # but the only old circlebox trainer I see does not generate coordinates outside the bounding box
    # not sure about these two lines, just need to try it both ways
    # c1b = (c1b[0]-box[0], c1b[1]-box[1], c1b[2]-box[0], c1b[3]-box[1])
    # c2b = (c2b[0]-box[0], c2b[1]-box[1], c2b[2]-box[0], c2b[3]-box[1])
    boximg = image[box[1]:box[3], box[0]:box[2]].copy()
    writeimg = np.zeros((h,w,3), np.uint8)
    writeimg[:] = (255,255,255)
    (bh, bw) = boximg.shape[:2]
    writeimgboxcoords = [(w/2)-(bw/2), (h/2)-(bh/2), (w/2)+(bw/2), (h/2)+(bh/2)]
    writeimgboxcoords = list(map(lambda x: int(x), writeimgboxcoords))
    writeimg[writeimgboxcoords[1]:writeimgboxcoords[3],writeimgboxcoords[0]:writeimgboxcoords[2]] = boximg.copy()
    # (h, w) = map(lambda x: int(x/2), (h, w))
    # cimage = np.zeros((h, w, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    # cimage[:] = (255, 255, 255)
    (c1b[0], c1b[2],c2b[0],c2b[2]) = map(lambda x: int(x+(w/2)-(bw/2)), (c1b[0], c1b[2],c2b[0],c2b[2]))
    (c1b[1], c1b[3],c2b[1],c2b[3]) = map(lambda x: int(x+(h/2)-(bh/2)), (c1b[1], c1b[3],c2b[1],c2b[3]))
    
    print(f'{box}, {c1b}')
    # cimage[0:(box[3]-box[1]), 0:(box[2]-box[0])] = image[box[1]:box[3], box[0]:box[2]].copy()
    # cv2.imwrite(imfname, cimage)
    cv2.imwrite(imfname, writeimg)
    c1csv.write(f'{imfname},{c1b[0]},{c1b[1]},{c1b[2]},{c1b[3]}\n')
    c2csv.write(f'{imfname},{c2b[0]},{c2b[1]},{c2b[2]},{c2b[3]}\n')
    if debugmode is True:
        dispimg = writeimg.copy()
        cv2.rectangle(dispimg, (c1b[0], c1b[1]), (c1b[2], c1b[3]), (255,0,0), 2)
        cv2.rectangle(dispimg, (c2b[0], c2b[1]), (c2b[2], c2b[3]), (255,0,0), 2)
        dispimg = cv2.resize(dispimg, (600,600))
        cv2.imshow("Bounding Box", dispimg)
        key = cv2.waitKey(0)
        if key==27:
            exit()
    return

parser = argparse.ArgumentParser()
parser.add_argument("indir", help="Input directory")
parser.add_argument("outdir", help="Output directory")
args = parser.parse_args()

print(f'{args.indir},{args.outdir}')

c1outfile = open(os.path.sep.join([args.outdir, 'c1box.csv']), 'w')
c2outfile = open(os.path.sep.join([args.outdir, 'c2box.csv']), 'w')

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
    # if h < 1000:
    #     cv2.imshow(filename, image)
    #     cv2.waitKey(0)

    # Translate the box randomly 10 times
    # for i in range(10):
    #     box = (min([c1x-c1r, c2x-c2r]), min([c1y-c1r, c2y-c2r]), max([c1x+c1r, c2x+c2r]), max([c1y+c1r, c2y+c2r]))
    #     c1box = (c1x-c1r, c1y-c1r, c1x+c1r, c1y+c1r)
    #     c1 = (c1x, c1y, c1r)
    #     c2box = (c2x-c2r, c2y-c2r, c2x+c2r, c2y+c2r)
    #     c2 = (c2x, c2y, c2r)
    #     imfname = os.path.sep.join([args.outdir, 'boundingbox', 'images', f'{filename[:-4]}{i}bb.jpg'])
    #     # print(f'{filename}\n{c1fname}')
    #     # input()
    #     writeCircleBoxImage(box, c1box, c2box, image, imfname, c1outfile, c2outfile)

    #     # key = cv2.waitKey(0)
    #     # if key == 27:
    #     #     break
    #     count += 1
    
    # rotate the image in 16 slices
    slices = 16
    angle = 0.
    inc = 360. / float(slices)
    while angle < 360:
        c1 = (c1x, c1y, c1r)
        c2 = (c2x, c2y, c2r)
        imfname = os.path.sep.join([args.outdir, 'boundingbox', 'images', f'{filename[:-4]}{angle}bb.jpg'])
        rotatedimg, c1, c2 = rotateImageAndCircles(image, c1, c2, angle)
        box = (min([c1[0]-c1[2], c2[0]-c2[2]]), min([c1[1]-c1[2], c2[1]-c2[2]]), max([c1[0]+c1[2], c2[0]+c2[2]]), max([c1[1]+c1[2], c2[1]+c2[2]]))
        c1box = (c1[0]-c1[2], c1[1]-c1[2], c1[0]+c1[2], c1[1]+c1[2])
        c2box = (c2[0]-c2[2], c2[1]-c2[2], c2[0]+c2[2], c2[1]+c2[2])
        # print(f'{filename}\n{c1fname}')
        # input()
        writeCircleBoxImage(box, c1box, c2box, rotatedimg, imfname, c1outfile, c2outfile)

        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
        angle += inc
        count += 1

print(f'{count} processed')
c1outfile.close()
c2outfile.close()