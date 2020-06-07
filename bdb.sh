KERAS_BACKEND=tensorflow
ipython dta.py -- \
                          --batch_size 256 \
                          --num_epoch 100 \
                          --max_seq_len 2000 \
                          --max_smi_len 100 \
                          --dataset_path 'data/bindingDB/' \
                          --is_log 0 \
                          --log_dir 'logs/' \
			  --word_representation True \
			  --smi_wordlen 8 \
			  --provided_domains True
