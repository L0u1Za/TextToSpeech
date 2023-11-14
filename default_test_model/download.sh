#!/bin/bash
fileid="12VOXv300Hl5UtNueRNhnfrWWhLPDaYP2"
filename="checkpoint.pth"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}