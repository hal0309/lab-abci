@echo off
py .\train\train.py config_d20l40.yaml
py .\train\train.py config_d20l80.yaml
py .\train\train.py config_d20l120.yaml
py .\train\train.py config_d20l160.yaml
py .\train\train.py config_d20l320.yaml
pause