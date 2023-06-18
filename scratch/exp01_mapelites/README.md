# MAP-Elites
これは, 変異に LLM を使用する単純な MAP-ELites アルゴリズムを実装したコードがあります.
LLM は GPT3.5 をしようしています.

## 実行方法
このディレクトリを ns3gym/scratch に置いて
```
export OPENAI_API_KEY=sk-hogehoge
python3  main.py
```
とすれば実行できます (OPENAI_API_KEY は各自の).

`main.py` の simulation関数の処理を並列化しています. 輻輳制御アルゴリズムは exec 関数によって文字列から関数に処理する仕組みになっています.
その関数を `cca.py` の get_action 関数に入れて使っています.

実行で生成した輻輳制御アルゴリズムは, `輻輳ウィンドウサイズの平均値_輻輳ウィンドウサイズの標準偏差` の規則でディレクトリがつくられ codes に保存されます. 
実行が完了すると, data ディレクトリにアーカイブマップの csv ファイル, result_graph ディレクトリにアーカイブマップのヒートマップが保存されます.
`make_archivemap.py` を実行することでも csv ファイルからアーカイブマップを生成できます.