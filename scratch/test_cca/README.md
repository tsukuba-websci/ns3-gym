# テスト用
これはテスト用の簡易的なコードがあります. シミュレーションのネットワーク環境やパラメータは元の実験と同じです.

## 実行方法
このディレクトリを ns3gym/scratch に置いて
```
export OPENAI_API_KEY=sk-hogehoge
python3  main.py
```
とすれば実行できます (OPENAI_API_KEY は各自の).

`generated_cca.py` に生成した輻輳制御アルゴリズムを保存して, それを `my_cca.py` で呼び出して更新する仕組みになっています.
`main.py` の simulation関数内で `obs[1] = 1` とすると NewReno が実行されます.

実行すると `generated_cca.py` が更新され, data ディレクトリにスループットの csv ファイル, result_graph ディレクトリに輻輳ウィンドウサイズのグラフが保存されます.