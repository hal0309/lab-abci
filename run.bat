@echo off
py .\train\train.py config_flat_2ax_zeros.yaml 
py .\train\train.py config_flat_3ax_distance.yaml 
py .\train\train.py config_flat_3ax_route.yaml 
py .\train\train.py config_flat_3ax_zeros.yaml
pause