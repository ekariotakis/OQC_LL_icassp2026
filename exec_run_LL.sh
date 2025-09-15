dt_name_vec='github web-Stanford caidaRouterLevel'

initialization_vec='S_oh S_greedy'

for dt_name in $dt_name_vec
do
    for initialization in $initialization_vec
    do
        for alpha in 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99
        do
            python test_LL.py --dataset-name "$dt_name" --alpha $alpha --initialization "$initialization"
        done
    done
done

