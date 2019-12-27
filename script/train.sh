#!/bin/bash

docker run --rm --name training -d -t -v $(pwd):/code -e PYTHONUSERBASE=/code/.fun/python --entrypoint /bin/bash aliyunfc/runtime-python3.6
docker exec -t training python /code/train.py --use_embedding --input_file data/poetry.txt --name poetry --learing_rate 0.005 --num_steps 26 --num_seqs 32 --max_steps 10
docker stop training