# /bin/bash
mkdir build_docker_image_mmir
mv ./Dockerfile build_docker_image_mmir/
cd build_docker_image_mmir
docker build -t mmir:1.0 -f Dockerfile .
