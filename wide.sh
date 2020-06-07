KERAS_BACKEND=tensorflow
ipython dta.py -- \
			  --word_representation True \
                          --batch_size 256 \
                          --num_epoch 100 \
                          --max_seq_len 1000 \
                          --max_smi_len 100 \
                          --dataset_path 'data/kiba/' \
                          --is_log 0 \
                          --log_dir 'logs/' \
			  --smi_wordlen 8 \
			  --smi_filter_length 8 \
			  --seq_filter_length 12 
