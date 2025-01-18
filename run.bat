@echo off
py .\train\train.py config_len_2_index_2ax_zeros.yaml 3000
py .\train\train.py config_len_5_index_2ax_zeros.yaml 3000
py .\train\train.py config_len_10_index_2ax_zeros.yaml 3000
py .\train\train.py config_len_20_index_2ax_zeros.yaml 3000
py .\train\train.py config_len_40_index_2ax_zeros.yaml 3000
py .\train\train.py config_len_d20_index_2ax_zeros.yaml 1000
py .\train\train.py config_len_d10_index_2ax_zeros.yaml 1000
py .\train\train.py config_len_d5_index_2ax_zeros.yaml 1000
py .\train\train.py config_len_d2_index_2ax_zeros.yaml 1000
pause

@REM py .\train\train.py config_flat_3ax_distance.yaml
@REM py .\train\train.py config_flat_3ax_route.yaml
@REM py .\train\train.py config_flat_3ax_zeros.yaml