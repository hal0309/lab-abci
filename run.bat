@echo off
py .\train\train.py config_divide1.yaml
py .\train\train.py config_divide2.yaml
py .\train\train.py config_divide4.yaml
py .\train\train.py config_divide8.yaml
py .\train\train.py config_divide10.yaml
py .\train\train.py config_divide20.yaml
pause