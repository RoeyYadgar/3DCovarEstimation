num_instances=$1


cleanup(){

	pkill -P $$
	exit 1

}


trap cleanup SIGINT


for ((i=1 ; i<=$num_instances; i++)); do
	python covar_estimate_test.py &
       	sleep 5
done

wait

echo "All instances finished execution"
