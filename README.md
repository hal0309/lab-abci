## 初期化手順

module(ABCI)のアクティベート
``` bash
module load python/3.10/3.10.14 cuda/11.8/11.8.0 cudnn/8.6/8.6.0
```

仮想環境の作成と起動
``` bash
python3 -m venv ./env
sourse env/bin/Activate
```

torchのinstall  
`nvcc --version`の結果によって適切なcuda対応torchをインストール


その他のinstall  
``` bash
pip install -r requirements.txt
```