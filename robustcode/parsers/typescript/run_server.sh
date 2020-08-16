#!/usr/bin/env bash

#trap "killall background" EXIT

for i in {3000..3015}
do
   nodejs server --port=$i &
done

while true
do
    sleep 1
done