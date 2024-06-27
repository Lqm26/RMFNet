# RMFNet_4x  NFS-syn
python  infer_RMFNet.py \
        --model_path /path/to/model \
        --data_list /path/to/data.txt  \
        --infer_mode 1 \
        --output_path /path/to/output \
        --scale 4 \ # define the SR scale 2/4/8
        --seqn 3 \
        --seql 9 \
        --step_size 1 \
        --time_bins 1 \
        --ori_scale down8 \
        --mode events \
        --window 2048 \
        --sliding_window 1024 \
        --need_gt_events 

# RMFNet_4x  RGB-syn
python  infer_RMFNet.py \
        --model_path /path/to/model \
        --data_list /path/to/data.txt  \
        --infer_mode 1 \
        --output_path /path/to/output \
        --scale 4 \ # define the SR scale 2/4
        --seqn 3 \
        --seql 9 \
        --step_size 1 \
        --time_bins 1 \
        --ori_scale down4 \
        --mode events \
        --window 16384 \
        --sliding_window 8192 \
        --need_gt_events 


# RMFNet_4x  eventNFS
python  infer_RMFNet.py \
        --model_path /path/to/model \
        --data_list /path/to/data.txt  \
        --infer_mode 1 \
        --output_path /path/to/output \
        --scale 4 \ # define the SR scale 2/4
        --seqn 3 \
        --seql 9 \
        --step_size 1 \
        --time_bins 1 \
        --ori_scale down4 \
        --mode events \
        --window 1024 \
        --sliding_window 512 \
        --need_gt_events 