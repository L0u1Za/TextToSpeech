#!/bin/bash
fileid="1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx"
filename="waveglow_256channels_ljs_v2.pt"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}