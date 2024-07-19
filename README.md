## 初期化手順

仮想環境の作成と起動
``` bash
python3 -m venv ./env
sourse env/bin/Activate
```

torchのinstall  
`nvcc --version`の結果によって適切なcuda対応torchをインストール


その他のinstall  
うまくいかない場合は`requirements.txt`の内容に一個ずつpipを叩くことを推奨
``` bash

pip install -r requirements.txt
```