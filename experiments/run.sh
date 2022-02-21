for i in $(seq 0 249)
    do
        echo 'Collecting results for Experiment 1'
        python train_from_hyperparameters.py sequential_no_act.py hyperparameters/hyperparameters_sequential_no_act.py $i outputs/results_sequential_no_act.csv
    done

for i in $(seq 0 799)
    do
        echo 'Collecting results for Experiment 2'
        python train_from_hyperparameters.py sequential.py hyperparameters/hyperparameters_sequential.py $i outputs/results_sequential.csv
    done

for i in $(seq 0 799)
    do
        echo 'Collecting results for Six Rooms'
        python train_from_hyperparameters.py six_rooms.py hyperparameters/hyperparameters_six_rooms.py $i outputs/results_six_rooms.csv
    done
