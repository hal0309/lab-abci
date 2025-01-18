@echo off
py .\train\train.py config_index_in5_d32_zeros.yaml 3000
py .\train\train.py config_index_in5_d64_zeros.yaml 3000
py .\train\train.py config_index_in5_d128_zeros.yaml 3000
py .\train\train.py config_index_in5_d256_zeros.yaml 3000

pause

@REM py .\train\train.py config_flat_3ax_distance.yaml 
@REM py .\train\train.py config_flat_3ax_route.yaml 
@REM py .\train\train.py config_flat_3ax_zeros.yaml