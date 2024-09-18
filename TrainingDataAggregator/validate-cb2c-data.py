import argparse
import os
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('indir', help='Input directory')
args = parser.parse_args()

rows = open(os.path.join(args.indir, 'cb2c.csv'), 'r').read().strip().split('\n')

for row in rows:
    row = row.split(',')
    filename = row[0]
    row[1:] = list(map(lambda x:int(x), row[1:]))
    c = row[1:4]
    image = cv2.imread(filename)
    cv2.circle(image, (c[0],c[1]), c[2], (255,0,0))
    cv2.imshow("cb2c", image)
    key = cv2.waitKey(0)
    if key==27:
        exit()