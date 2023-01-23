#!/bin/bash

read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD

wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

unzip gtFine_trainvaltest.zip
unzip VisDrone2019-MOT-test-dev.zip

rm -rf license.txt README
rm -rf cookies.txt index.html
