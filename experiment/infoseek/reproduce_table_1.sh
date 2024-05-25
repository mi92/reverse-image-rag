# Idefics-2
EXP_NAME=infoseek_i2
python -m experiment.infoseek.run_sample_i2 \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --idx_offset 0
# Idefics-2 Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09


# GPT-4V
EXP_NAME=infoseek_gpt4v
python -m experiment.infoseek.run_sample \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --model gpt-4-1106-vision-preview \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --generate_new_screenshot 0 \
    --screenshot_dir local_data/infoseek_finished/screenshot/ \
    --idx_offset 0
# GPT-4V Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09

# GPT-4-turbo
EXP_NAME=infoseek_gpt4turbo
python -m experiment.infoseek.run_sample \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09 \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --generate_new_screenshot 0 \
    --screenshot_dir local_data/infoseek_finished/screenshot/ \
    --idx_offset 0
# GPT-4V Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09


# GPT-4o
EXP_NAME=infoseek_gpt4o
python -m experiment.infoseek.run_sample \
    --output_root local_data/infoseek/infoseek_gpt4o_1650 \
    --log_name logs_infoseek_gpt4o_1650 \
    --model gpt-4o-2024-05-13 \
    --sys_msg_filename query_system_no_screenshot.jinja2 \
    --use_screenshot 0 \
    --generate_new_screenshot 0 \
    --screenshot_dir local_data/infoseek_finished/screenshot/ \
    --idx_offset 0
# GPT-4o Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09


# Idefics-2 RIR
EXP_NAME=infoseek_i2_rir
python -m experiment.infoseek.run_sample_i2 \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --idx_offset 0
# Idefics-2 RIR Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09

# GPT-4V RIR
EXP_NAME=infoseek_gpt4v_rir
python -m experiment.infoseek.run_sample \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --model gpt-4-1106-vision-preview \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --generate_new_screenshot 0 \
    --screenshot_dir local_data/infoseek_finished/screenshot/ \
    --idx_offset 0
# GPT-4V RIR Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09

# GPT-4-turbo RIR
EXP_NAME=infoseek_gpt4turbo_rir
python -m experiment.infoseek.run_sample \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09 \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --generate_new_screenshot 0 \
    --screenshot_dir local_data/infoseek_finished/screenshot/ \
    --idx_offset 0
# GPT-4V Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09


# GPT-4o
EXP_NAME=infoseek_gpt4o_rir
python -m experiment.infoseek.run_sample \
    --output_root local_data/infoseek/${EXP_NAME} \
    --log_name logs_${EXP_NAME} \
    --model gpt-4o-2024-05-13 \
    --sys_msg_filename query_system_with_screenshot.jinja2 \
    --use_screenshot 1 \
    --generate_new_screenshot 0 \
    --screenshot_dir local_data/infoseek_finished/screenshot/ \
    --idx_offset 0
# GPT-4o Judge
python -m experiment.infoseek.judge \
    --output_root local_data/infoseek/${EXP_NAME}/ \
    --exp_name logs_${EXP_NAME} \
    --model gpt-4-turbo-2024-04-09

# Summarize the results
