@echo off
py .\train\train.py config_20d_80in_128m.yaml
py .\train\train.py config_10d_80in_128m.yaml
py .\train\train.py config_5d_80in_128m.yaml
py .\train\train.py config_2d_80in_128m.yaml
pause