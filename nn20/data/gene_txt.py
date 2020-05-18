# cd dataset
# python gene_txt.py

import os
import cv2
import numpy as np

note='Chi2Num.txt'

trainPath='train/'
traintxt='train.txt'
valtxt='val.txt'

testPath='test2/'
testtxt='test2.txt'

mode='test1'

if mode =='train':
    fTrain=open(traintxt,'w', encoding='utf-8')
    fVal=open(valtxt,'w', encoding='utf-8')
    with open(note,'w', encoding='utf-8') as note:
        label=0
        for root, dirs, files in os.walk(trainPath):
            
            if len(dirs)==0:
        
                trainfiles=files[0:380]
                valfiles=files[380:400]

                
                for imgPath in trainfiles:
                    line = root + '/' + imgPath+'  '+str(label)+'\n'
                    fTrain.write(line)
                for imgPath in valfiles:
                    line = root + '/' + imgPath+'  '+str(label)+'\n'
                    fVal.write(line)
                
                line_note=root[-1]+'  '+str(label)
                note.write(root[-1])
                note.write(str(label))
                note.write('\n')

                label+=1

    fTrain.close()
    fVal.close()     

elif mode == 'test1':
    fTest=open(testtxt, 'w', encoding='utf-8')
    for root, dirs, files in os.walk(testPath):  
        if len(dirs)==0:         
            for imgPath in files:
                    line= root + imgPath + '\n'
                    fTest.write(line)