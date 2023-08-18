#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -it \
    --gpus=all \
    --ipc=host \
    --name=cnn_docker \
    --env-file=.env \
    --volume=$PWD:/workspace \
    --volume=$PWD/../data/Googlefonts:/data/Googlefonts \
    my_docker:latest \
    fish
