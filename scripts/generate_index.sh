#!/bin/bash


declare -A server_commands=( ["mate"]="/tank/local/suhail/MATE/src/run_arrowserver.sh" \
                             ["lshe"]="/tank/local/suhail/relic-datalake/arrow/run_arrowserver.sh" \
                             ["gb-kmv"]="/tank/local/suhail/gb-kmv-code/GBKMVquery/run_arrowserver.sh" )


# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
        server_pid=`cat server.pid`
        echo "Killing server $method with PID $server_pid"
        kill $server_pid
        exit 0
}


method='mate'
maxcolset=2
input_dir='/tank/local/suhail/data/relic-datalake/gittables/samples/5000_sample.txt'
threshold='0.4'
#query_output_dir='/tank/local/suhail/data/relic-datalake/gittables/outputs/500_3/artifacts/'
#query_ops_file='/tank/local/suhail/data/relic-datalake/gittables/outputs/500_3/operations.parquet'



# Ensure no other process is using tcp/33333
fuser -k 33333/tcp

# # Start the Server
server_command=${server_commands[$method]}
echo "Starting server: $method"
eval "$server_command"
sleep 2

client_cmd="python ../src/arrow_client.py"

#Generate Sketches
echo "Starting Sketch Generation at: `date`"
result_dir="/tank/local/suhail/data/relic-datalake/gittables/outputs/500_3//${method}/"
mkdir -p $result_dir
$client_cmd --mode=sketch --input=$input_dir --result_dir=$result_dir --threshold=$threshold --maxcolset=$maxcolset
echo "Finished Sketch Generation at: `date`"

while :; do
    sleep 5
done
