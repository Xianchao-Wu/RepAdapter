#########################################################################
# File Name: download.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun 20 00:15:44 2023
#########################################################################
#!/bin/bash


#wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p' -O vtab-1k.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p" -O vtab-1k.zip && rm -rf /tmp/cookies.txt
