# Idefics-2
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

# GPTs
python -m experiment.snake.run_snake \
    --output_root local_data/snake/snake_gpt4o_300_0514 \
    --log_name logs_snake_gpt4o_300_0514 \
    --model gpt-4o-2024-05-13 \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --screenshot_dir local_data/snake/snake_screenshot/ \
    --idx_offset 0 
python -m experiment.snake.run_snake \
    --output_root local_data/snake/snake_gpt4o_rir_300_0514 \
    --log_name logs_snake_gpt4o_rir_300_0514 \
    --model gpt-4o-2024-05-13 \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --screenshot_dir local_data/snake/snake_screenshot/ \
    --idx_offset 0 
python -m experiment.snake.run_snake \
    --output_root local_data/snake/snake_gpt4turbo_300_0514 \
    --log_name logs_snake_gpt4turbo_300_0514 \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --model gpt-4-turbo-2024-04-09 \
    --use_screenshot 0 \
    --screenshot_dir local_data/snake/snake_screenshot/ \
    --idx_offset 0 
python -m experiment.snake.run_snake \
    --output_root local_data/snake/snake_gpt4turbo_rir_300_0514 \
    --log_name logs_snake_gpt4turbo_rir_300_0514 \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --model gpt-4-turbo-2024-04-09 \
    --use_screenshot 1 \
    --screenshot_dir local_data/snake/snake_screenshot/ \
    --idx_offset 0 
python -m experiment.snake.run_snake \
    --output_root local_data/snake/snake_gpt4v_300_0514 \
    --log_name logs_snake_gpt4v_300_0514 \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --model gpt-4-1106-vision-preview \
    --use_screenshot 0 \
    --screenshot_dir local_data/snake/snake_screenshot/ \
    --idx_offset 0 
python -m experiment.snake.run_snake \
    --output_root local_data/snake/snake_gpt4v_rir_300_0514 \
    --log_name logs_snake_gpt4v_rir_300_0514 \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --model gpt-4-1106-vision-preview \
    --use_screenshot 1 \
    --screenshot_dir local_data/snake/snake_screenshot/ \
    --idx_offset 0 
