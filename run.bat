@echo off
py .\train\train.py config_index_2ax_distance.yaml
py .\train\train.py config_index_2ax_route.yaml
py .\train\train.py config_index_2ax_zeros.yaml
py .\train\train.py config_index_3ax_distance.yaml
py .\train\train.py config_index_3ax_route.yaml
py .\train\train.py config_index_3ax_zeros.yaml

pause