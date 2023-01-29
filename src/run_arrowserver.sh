#!/bin/bash

cd "$(dirname "$0")"

python arrow_server.py &
server_pid=$!
echo $server_pid > server.pid