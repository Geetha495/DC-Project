# Run a.py and server.py by varying argument num_procs from 2 to 40 with step 2
for i in $(seq 2 2 40)
do
    echo "Running with num_procs = $i"
    python a.py -n $i
    python server.py -n $i
done