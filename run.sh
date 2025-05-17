xhost +local:root # Allow the container to access the display

docker run --gpus all --rm -it \
--ipc host \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v ~/.Xauthority:/root/.Xauthority \
-e XAUTHORITY=/root/.Xauthority \
-v $PWD:/workspace \
bi_agent_rl_scheduler_genesis_wandb
