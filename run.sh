xhost +local:root # Allow the container to access the display

docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
genesis
