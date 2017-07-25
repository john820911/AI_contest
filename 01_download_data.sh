#!/bin/bash
echo "Download data and unzip them..."

curl -H "Authorization: Bearer ya29.GltIBBxLQoj8_lSGNSTszXnHmpvW1V2UkpVpM425teDOYo4HoAaVMfshn3U_rBIgDDnQPv_Ryhe5D0PNBRGmVmdOda6g0VbSoerCiu_XtgmwoPS4uffiS_w-9ciK" https://www.googleapis.com/drive/v3/files/0B2hNk0_VowQmZnk2YU5zYUpKTms?alt=media -o data.zip
unzip -u-o data.zip
rm -rf data.zip

echo "Finish downloading data!!"
