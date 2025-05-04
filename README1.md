   # Parameters 
./dstree --db_filepath /home/qwang/projects/leafi/dataset/deep1b-96-1m.bin \
--query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.4-10k.bin \
--series_length 96 \
--db_size 1000000 \
--query_size 10000 \
--leaf_size 10000 \
--exact_search \
--require_neurofilter \
--filter_train_is_gpu \
--filter_infer_is_gpu \
--learning_rate 0.01 \
--filter_train_min_lr 0.000001 \
--filter_train_clip_grad \
--filter_train_nepoch 1000 \
--filter_collect_nthread 1 \
--filter_remove_square \
--filter_is_conformal \
--filter_model_setting mlp \
--device_id 0 \
--dump_index \
--filter_trial_nnode 1 \
--filter_allocate_is_gain \
--filter_conformal_is_smoothen \
--filter_train_mthread \
--filter_train_nthread 48 \
--filter_train_num_local_example 1000 \
--filter_train_num_global_example 3000 \
--filter_conformal_k_parts 1 \
--filter_conformal_n_parts 15 \
--filter_train_val_split 0.5 \
--filter_conformal_coverage 0.90 \
--filter_conformal_recall 0.99 \
--filter_conformal_num_batches 100 \
--log_filepath /home/qwang/projects/leafi/dstree/log/R99_C95_K1_QN4_B100_S100.log \
--index_dump_folderpath /home/qwang/projects/leafi/dstree/index_R99_C95_K1_QN4_B100_S100 \
--results_path /home/qwang/projects/leafi/dstree/R99_C95_K1_QN4_B100_S100 \
--filter_query_max_noise 0.1 \
--filter_query_min_noise 0.4 \
--n_nearest_neighbor 1 \
--filter_conformal_use_combinatorial 


./dstree --db_filepath /home/qwang/projects/leafi/dataset/deep1b-96-1m.bin \
--query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10k.bin \
--series_length 96 \
--db_size 1000000 \
--query_size 10000 \
--leaf_size 10000 \
--exact_search \
--require_neurofilter \
--filter_train_is_gpu \
--filter_infer_is_gpu \
--learning_rate 0.01 \
--filter_train_min_lr 0.000001 \
--filter_train_clip_grad \
--filter_train_nepoch 1000 \
--filter_collect_nthread 1 \
--filter_remove_square \
--filter_is_conformal \
--filter_model_setting mlp \
--device_id 1 \
--dump_index \
--filter_trial_nnode 1 \
--filter_allocate_is_gain \
--filter_conformal_is_smoothen \
--filter_train_mthread \
--filter_train_nthread 48 \
--filter_train_num_local_example 1000 \
--filter_train_num_global_example 3000 \
--filter_conformal_k_parts 1 \
--filter_conformal_n_parts 15 \
--filter_train_val_split 0.5 \
--filter_conformal_coverage 0.90 \
--filter_conformal_recall 0.99 \
--filter_conformal_num_batches 100 \
--log_filepath /home/qwang/projects/leafi/dstree/log/R99_C90_K1_QN2_B100_S100.log \
--index_dump_folderpath /home/qwang/projects/leafi/dstree/index_R99_C90_K1_QN2_B100_S100 \
--results_path /home/qwang/projects/leafi/dstree/R99_C90_K1_QN2_B100_S100 \
--filter_query_max_noise 0.1 \
--filter_query_min_noise 0.4 \
--n_nearest_neighbor 1 \
--filter_conformal_use_combinatorial 



 ./dstree --db_filepath /home/qwang/projects/leafi/dataset/deep1b-96-1m_1000.bin --query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.4-1k.bin --series_length 96 --db_size 1000 --query_size 1000 --leaf_size 50 --exact_search --require_neurofilter --filter_train_is_gpu --filter_infer_is_gpu --learning_rate 0.01 --filter_train_min_lr 0.000001 --filter_train_clip_grad --filter_train_nepoch 1000 --filter_collect_nthread 1 --filter_remove_square --filter_is_conformal --filter_model_setting mlp --device_id 0 --dump_index --filter_trial_nnode 1 --filter_allocate_is_gain --filter_conformal_is_smoothen --filter_train_mthread --filter_train_nthread 48 --filter_train_num_local_example 100 --filter_train_num_global_example 300 --filter_conformal_k_parts 1 --filter_conformal_n_parts 10 --filter_train_val_split 0.5 --filter_conformal_coverage 0.95 --filter_conformal_recall 0.99 --filter_conformal_num_batches 5 --log_filepath /home/qwang/projects/leafi/dstree/log/results_check_errors9.log --index_dump_folderpath /home/qwang/projects/leafi/dstree/index_dump9 --results_path /home/qwang/projects/leafi/dstree/results_check_errors9 --filter_query_max_noise 0.1 --filter_query_min_noise 0.4 --n_nearest_neighbor 1 --filter_conformal_use_combinatorial
