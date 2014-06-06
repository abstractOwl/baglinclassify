#!/usr/bin/python

import sys

pos_c = 0
pos_x = 0.0
pos_y = 0.0

neg_c = 0
neg_x = 0.0
neg_y = 0.0

for line in sys.stdin:
    if not line.strip():
        continue
    x, y, group = line.strip().split(" ")

    if group == "True":
        pos_x += float(x)
        pos_y += float(y)
        pos_c += 1
    else:
        neg_x += float(x)
        neg_y += float(y)
        neg_c += 1

# Compute centroid
pos_x /= pos_c
pos_y /= pos_c

neg_x /= neg_c
neg_y /= neg_c

# Compute W
w_x = pos_x - neg_x
w_y = pos_y - neg_y

# Compute T
t = w_x * (pos_x + neg_x) + w_y * (pos_y + neg_y)
t /= 2

# Y-I, Slope
print (-w_x / w_y), (t / w_y)
