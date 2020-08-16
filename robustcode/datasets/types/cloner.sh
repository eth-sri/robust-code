#!/bin/bash

cat repo-SHAs.txt | head -n 10 | xargs -P8 -n1 -I% bash -c 'echo %; \
 sha=$(echo % | cut -d" " -f2); \
 name=$(echo % | cut -d" " -f1); \
 head=$(echo $name | cut -d"/" -f1); \
 mkdir -p data/Repos/$head; \
 git clone -q https://github.com/$name data/Repos/$name; \
 git -C data/Repos/$name reset --hard $sha;'