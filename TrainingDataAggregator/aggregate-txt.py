from glob import glob
import os
import sys

os.chdir(sys.argv[1])

master = open("master.csv", "w")
geometry = open("geometry.csv", "w")
translation = open("translation.csv", "w")
c1 = open("c1.csv", "w")
c2 = open("c2.csv", "w")

print( len(glob("*.txt")))
print(sys.argv[1] + "\\master.csv")

for f in glob("*.txt"):
    infile = open(f, "r")
    s = infile.read()
    master.write(s + "\n")
    (fname, c1x, c1y, c1r, c2x, c2y, c2r) = s.split(',')
    l = ((float(c2x)-float(c1x))**2 + (float(c2y)-float(c1y))**2)**0.5
    gs = f"{fname},{c1r},{c2r},{round(l)}\n"
    ts = f"{fname},{c1x},{c1y},{c2x},{c2y}\n"

    geometry.write(gs)
    translation.write(ts)
    c1.write(f"{fname},{c1x},{c1y},{c1r}\n")
    c2.write(f"{fname},{c2x},{c2y},{c2r}\n")
    infile.close()

master.close()
geometry.close()
translation.close()
