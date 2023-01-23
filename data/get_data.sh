#!/bin/bash

read -p "Username: " USERNAME
read -s -p "Password: " PASSWORD

wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu" -O VisDrone2019-MOT-test-dev.zip && rm -rf /tmp/cookies.txt
