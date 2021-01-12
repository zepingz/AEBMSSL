# Additional resources


## CSV creation

```bash
python create_csv.py --data_root /scratch/jz3224/WaymoDataset --store_dir /scratch/zz2332/WaymoDataset
```


## HDF5 creation

```
python batch_create_hdf5.py --data_root /scratch/jz3224/WaymoDataset --store_dir /scratch/zz2332/WaymoDataset --slurm_dir /home/zz2332/slurm --process_num 5 --mode batch
```

## Compute mean and standard deviation
```
python compute_mean_std.py --data_root /scratch/zz2332/WaymoDataset
```
