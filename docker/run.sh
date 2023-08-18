#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -it \
    --ipc=host \
    --name=kaggle-prac \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$PWD/../data/Googlefonts:/data/Googlefonts \
    kaggle-prac:latest \
    fish
