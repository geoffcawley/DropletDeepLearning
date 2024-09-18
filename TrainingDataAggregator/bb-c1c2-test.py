import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("indir", help="Input directory")

args = parser.parse_args()

rows = open(os.path.sep.join(
    [args.indir, 'master.csv'])).read().strip().split("\n")
print(f'{len(rows)} rows')
count = 0
for row in rows:
    row = row.split(',')

    (filename, c1X, c1Y, c1R, c2X, c2Y, c2R) = row
    (c1X, c1Y, c1R, c2X, c2Y, c2R) = (int(c1X), int(c1Y), int(c1R), int(c2X), int(c2Y), int(c2R))
    imagePath = os.path.sep.join([args.indir, filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    (bbx1, bby1, bbx2, bby2) = (min([c1X-c1R, c2X-c2R]), min(
        [c1Y-c1R, c2Y-c2R]), max([c1X+c1R, c2X+c2R]), max([c1Y+c1R, c2Y+c2R]))
    (bcx, bcy) = ((bbx1+bbx2)/2, (bby1+bby2)/2)
    halfside = max([bbx2-bbx1, bby2-bby1])/2
    (bsx1, bsy1, bsx2, bsy2) = (int(bcx-halfside),
                                int(bcy-halfside), int(bcx+halfside), int(bcy+halfside))
    if bsx1 < 0:
        bsx2 = int(2*halfside)
        bsx1 = 0
    if bsy1 < 0:
        bsy2 =int(2*halfside)
        bsy1 = 0
    if bsx2 >= w:
        bsx1 -= (bsx2-(w-1))
        bsx2 = w-1
    if bsy2 >= h:
        bsy1 -= (bsy2-(h-1))
        bsy2 = h-1

    bs = (image[bsy1:bsy2, bsx1:bsx2]).copy()
    
    bc1c = (c1X-bsx1, c1Y-bsy1)
    bc2c = (c2X-bsx1, c2Y-bsy1)
    
    try:
        cv2.circle(bs, bc1c, c1R, (255,0,0), 2)
        cv2.circle(bs, bc2c, c2R, (0,255,0), 2)
        
        cv2.resize(image, (600, 600))
        cv2.imshow("Original", image)
        cv2.imshow("Bounding square", bs)
        key = cv2.waitKey(10)
        if key == 27:
            break
        if bsx2-bsx1 != bsy2-bsy1:
            raise Exception('Not square')
    except:
        print(image.shape[:2])
        print(f'({bbx1},{bby1}),({bbx2},{bby2})')
        print(f'({bsx1},{bsy1}),({bsx2},{bsy2})')
        print(f'{bsx2-bsx1}x{bsy2-bsy1}')
        break
    count += 1

print(f'{count} processed')
