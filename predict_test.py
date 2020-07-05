"""
https://github.com/FrederikSchorr/sign-language
Train a pre-trained I3D convolutional network to classify videos
"""

import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import keras
from keras import backend as K
from model_i3d import I3D_load
from datagenerator import VideoClasses, FramesGenerator
from model_i3d import Inception_Inflated3d, add_i3d_top

#sFolder = "%03d-%d"%(63, 200)
sFolder = "%03d-%d"%(100, 200)
sOflowDir  = "data-temp/%s/%s/oflow"%('tsl', sFolder)
genFramesTest = FramesGenerator(sOflowDir + "/test_videos", 1, 
        200, 224, 224, 2, bShuffle=False)
label = genFramesTest.dfVideos["sLabel"].tolist()

#sModelFile = "model/20200619-0535-tsl063-oflow-i3d-entire-best.h5"
sModelFile = "model/20200704-1221-tsl100-oflow-i3d-entire-best.h5"

h, w = 224, 224
keI3D = I3D_load(sModelFile, 200, (h, w, 2), 63)


nTop = 3
acc = 0
fail = []
for step in range(genFramesTest.__len__()):
    x, y = genFramesTest.__getitem__(step)
    arProbas = keI3D.predict(x, verbose = 1)[0]

    arTopLabels = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[arTopLabels]
    top_str = []
    for i in range(nTop):
        top_str.append("Top %d: [%3d] %s (confidence %.1f%%)" % \
            (i+1, arTopLabels[i], label[arTopLabels[i]] + " ", arTopProbas[i]*100.))
        print("Top %d: [%3d] %s (confidence %.1f%%)" % \
            (i+1, arTopLabels[i], label[arTopLabels[i]] + " ", arTopProbas[i]*100.))

    nLabel, sLabel, fProba = arTopLabels[0], label[arTopLabels[0]], arTopProbas[0]
    sResults = "Sign: %s (%.0f%%)" % (sLabel, fProba*100.)
	#print(sResults)

    groundth = np.argmax(y ,axis=1)[0]
    if groundth != arTopLabels[0] or arTopProbas[0]*100 < 50:
        fail.append( [top_str,  f' groundth={label[groundth]}, predict={sLabel}, confidence={fProba*100}' ] )
    if groundth == arTopLabels[0]:
        acc += 1
    #print( y, groundth, fProba*100)
    print( f' groundth={label[groundth]}, predict={sLabel}, confidence={fProba*100}')

print(f'ACC: {acc} / {genFramesTest.__len__()}')

import json
with open('fail_summary.json', 'w', encoding='utf-8') as f:
    json.dump(fail, f, ensure_ascii=False, indent=4)
