#for N_POINTS in 250 500 1000 2500 5000
for N_POINTS in 2500
do
python scripts/benchmark_registration.py --source_path ./snapshot/tdmatch_conv_test_20/3DLoMatch --benchmark 3DLoMatch --n_points $N_POINTS
done

