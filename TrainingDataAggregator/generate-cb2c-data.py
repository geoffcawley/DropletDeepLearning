import os
import argparse
import cv2
import random
import numpy as np

def randomizeBox(box, circle, dims, randrange):
    (bx1, by1, bx2, by2) = box
    
    xt = random.randrange(randrange) - int(randrange/2)
    yt = random.randrange(randrange) - int(randrange/2)
    border = int((random.randrange(randrange) - int(randrange/2))/2)
    bx1 += xt - border
    bx2 += xt + border
    by1 += yt - border
    by2 += yt + border
    (x, y, r) = circle
    # x += xt
    # y += yt
    
    box = (bx1, by1, bx2, by2)
    circle = (x, y, r)
    return box, circle

def validateBox(box, dims):
    if box[0] < 0 or box[2] < 0:
        return False
    if box[1] > dims[1] or box[3] > dims[0]:
        return False
    return True

def writeCircleImage(box, circle, image, imfname, csvoutf):
    print(f'{box}, {circle}')
    (h, w) = image.shape[:2]
    # for i in range(10):
    box, circle = randomizeBox(box, circle, (h,w), 300)
    box = list(box)
    circle = list(circle)
    print(f'{box}, {circle}')
    # if not validateBox(box, (h,w)):
    #     continue
    bc = (int((box[2]+box[0])/2), int((box[3]+box[1])/2))
    if box[0] < 0:
        d = -box[0]
        box[0] = 0
        box[2] += d
    if box[1] < 0:
        d = -box[1]
        box[1] = 0
        box[3] += d
    if box[2] >= w:
        d = box[2] - w
        box[2] = w
        box[0] -= d
    if box[3] >= h:
        d = box[3] - h
        box[3] = h
        box[1] -= d
        
    circle = (circle[0]-box[0], circle[1]-box[1], circle[2])
    c1dispimg = image[box[1]:box[3], box[0]:box[2]].copy()
    
    (h, w) = map(lambda x: int(x/2), (h, w))
    cimage = np.zeros((h, w, 3), np.uint8)
    # Fill image with red color(set each pixel to red)
    cimage[:] = (255, 255, 255)
    print(f'{box}, {circle}')
    cimage[0:(box[3]-box[1]), 0:(box[2]-box[0])] = image[box[1]:box[3], box[0]:box[2]].copy()
    cv2.imwrite(imfname, cimage)
    csvoutf.write(f'{imfname},{circle[0]},{circle[1]},{circle[2]}\n')
    cv2.circle(c1dispimg, (circle[0], circle[1]), circle[2], (255,0,0), 2)
    # cv2.imshow("C1", c1dispimg)
    return

parser = argparse.ArgumentParser()
parser.add_argument("indir", help="Input directory")
parser.add_argument("outdir", help="Output directory")
args = parser.parse_args()

print(f'{args.indir},{args.outdir}')

c1outfile = open(os.path.sep.join([args.outdir, 'c1b2c.csv']), 'w')
c2outfile = open(os.path.sep.join([args.outdir, 'c2b2c.csv']), 'w')

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
    
    for i in range(10):
        c1box = (c1x-c1r, c1y-c1r, c1x+c1r, c1y+c1r)
        c1 = (c1x, c1y, c1r)
        c1fname = os.path.sep.join([args.outdir, 'c1b2c', 'images', f'{filename[:-4]}{i}c1.jpg'])
        # print(f'{filename}\n{c1fname}')
        # input()
        writeCircleImage(c1box, c1, image, c1fname, c1outfile)

        c2box = (c2x-c2r, c2y-c2r, c2x+c2r, c2y+c2r)
        c2 = (c2x, c2y, c2r)
        c2fname = os.path.sep.join([args.outdir, 'c2b2c', 'images', f'{filename[:-4]}{i}c2.jpg'])
        writeCircleImage(c2box, c2, image, c2fname, c2outfile)

        # key = cv2.waitKey(0)
        # if key == 27:
        #     break
        count += 1

print(f'{count} processed')
c1outfile.close()
c2outfile.close()