#!/bin/bash
fileid="1rpK8CzAAirq9sWZhe9nlfvxMF1dRgFbF"
filename="waveglow_256channels_universal_v5.pt"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}