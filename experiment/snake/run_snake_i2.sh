python -m experiment.snake.run_snake_i2 \
    --output_root local_data/snake/snake_i2_300_recall \
    --log_name logs_snake_i2_300_recall \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --idx_offset 0 \
    --metric recall 

python -m experiment.snake.run_snake_i2 \
    --output_root local_data/snake/snake_i2_rir_300_recall \
    --log_name logs_snake_i2_rir_300_recall \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --idx_offset 0 \
    --metric recall 

python -m experiment.snake.run_snake_i2 \
    --output_root local_data/snake/snake_i2_300_em \
    --log_name logs_snake_i2_300_em \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --idx_offset 0 \
    --metric em 

python -m experiment.snake.run_snake_i2 \
    --output_root local_data/snake/snake_i2_rir_300_em \
    --log_name logs_snake_i2_rir_300_em \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --idx_offset 0 \
    --metric em 