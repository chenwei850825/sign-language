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
from datagenerator import VideoClasses, FramesGenerator, generate_generator_multiple
from model_i3d import Inception_Inflated3d, add_i3d_top
import json

#==== model frame number
frames_num = 115

#==== model input type
#sModelFile = "model_flow_mirror/20200706-0517-tsl100-oflow-i3d-entire-best.h5"
#sModelFile = "model_rgb_mirror/20200711-0410-tsl100-115-oflow-i3d-entire-best.h5"
sModelFile = "model_combined_mirror/115_rgb.h5"
#==== model load
h, w = 224, 224
keI3D = I3D_load(sModelFile, frames_num, (h, w, 2), 63)
#keI3D = I3D_load(sModelFile, frames_num, (h, w, 3), 63)
#keI3D = I3D_load(sModelFile, frames_num, (h, w, 2), 63)
input_type = 'combined_test'


sFolder = "%03d-%d"%(100, frames_num)
sOflowDir  = "data-temp/%s/%s/oflow"%('tsl', sFolder)
sImageDir  = "data-temp/%s/%s/image"%('tsl', sFolder)

genFramesTest_flow = FramesGenerator(sOflowDir + "/test_videos", 1, 
        frames_num, 224, 224, 2, bShuffle=False)
genFramesTest_rgb = FramesGenerator(sImageDir + "/test_videos", 1, 
        frames_num, 224, 224, 3, bShuffle=False, test_phase=True)
genFramesTest_combined = generate_generator_multiple(genFramesTest_rgb, genFramesTest_flow)
#==== model input generator
select_gen = genFramesTest_combined



label = genFramesTest_rgb.dfVideos["sLabel"].tolist()
nTop = 3
acc = 0
fail = {}
fail['fail'] = []
fail['low_confidence'] = []
for step in range(select_gen.__len__()):
    x, y = select_gen.__getitem__(step)
    print(len(x))
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
    if groundth != arTopLabels[0]:
        fail['fail'].append( [top_str,  f' groundth={label[groundth]}, predict={sLabel}, confidence={fProba*100}' ] )
    if arTopProbas[0]*100 < 50:
        fail['low_confidence'].append( [top_str,  f' groundth={label[groundth]}, predict={sLabel}, confidence={fProba*100}' ] )

    if groundth == arTopLabels[0]:
        acc += 1
    #print( y, groundth, fProba*100)
    print( f' groundth={label[groundth]}, predict={sLabel}, confidence={fProba*100}')

fail['accuracy'] = f'ACC: {acc} / {select_gen.__len__()}'
print(f'ACC: {acc} / {select_gen.__len__()}')


##
with open(f'fail_{input_type}_{frames_num}.json', 'w', encoding='utf-8') as f:
    json.dump(fail, f, ensure_ascii=False, indent=4)
