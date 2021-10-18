#docker container stop text_affinity
#docker container rm text_affinity
#docker network rm text_affinity_net
#docker network create text_affinity_net
#docker volume create volume_text_affinity_storage
#docker build -t text_affinity:latest .
#docker run -idt -p 8000:8000 --mount source=volume_text_affinity_storage,target=/text-affinity-storage  --name text_affinity --net=text_affinity_net text_affinity:latest