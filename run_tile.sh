python run_infer.py \
--gpu='0,1' \
--nr_types=5 \
--type_info_path=type_info.json \
--batch_size=64 \
--model_mode=original \
--model_path=logs/ulsam/01/net_epoch=18.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=dataset/CoNSeP/Test/Images \
--output_dir=dataset/sample_tiles \
--draw_dot \
--save_qupath
