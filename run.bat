@echo off
py .\train\train.py config_dec_2ax_zeros.yaml
py .\train\train.py config_dec_2ax_distance.yaml
py .\train\train.py config_dec_2ax_route.yaml
py .\train\train.py config_dec_2ax_route_diff.yaml
pause