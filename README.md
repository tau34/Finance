# 株式・暗号資産の価格予測AI（初期版）

現段階は、5分足などのyfinance CSVを1時間足に集約する方式Aを実装しています。
30分後・1時間後・3時間後の対数リターン回帰と、`Short / Neutral / Long`
分類を同時に学習します。未来情報を使わないよう、時系列順の80/20ホールドアウトで評価します。

## CSVのどの値を使うか

入力CSVは、`yf.download(...).to_csv(...)` で作成したOHLCV形式を想定しています。
AAPL.csvの場合、実際に読み込む元データは次のとおりです。

| CSVの値 | 使用方法 |
|---|---|
| `Datetime` | 時刻の並び替え、1時間足への集約、`hour`・`weekday`の作成に使用 |
| `Open` | 1時間足の始値を作るために使用 |
| `High` | 1時間足の高値、値幅、ATRを作るために使用 |
| `Low` | 1時間足の安値、値幅、ATRを作るために使用 |
| `Close` | リターン、移動平均、EMA、RSI、ボラティリティ、予測対象を作るために使用 |
| `Volume` | 1時間足の出来高合計と出来高変化率を作るために使用 |

`Adj Close`、銘柄名、CSVの3行ヘッダーにあるTicker値は、現在の入力特徴量には
使用していません。`symbol_code`はデータセット内部で作成されますが、学習に渡す
`FEATURE_COLUMNS`には含まれていないため、現状のモデルはこの値も使用しません。

### 1時間足への集約

例えば5分足CSVの場合、同じ1時間枠のデータを次のルールで集約します。

- `Open`: その時間枠の最初の値
- `High`: その時間枠の最大値
- `Low`: その時間枠の最小値
- `Close`: その時間枠の最後の値
- `Volume`: その時間枠の合計値

その後、各時点について以下の14個の特徴量を計算し、学習入力にします。

| 特徴量 | 意味 |
|---|---|
| `return_1` | 直前の1時間足からの価格変化率 |
| `return_3` | 3時間前からの価格変化率 |
| `return_6` | 6時間前からの価格変化率 |
| `return_12` | 12時間前からの価格変化率 |
| `range_pct` | `(High - Low) / Close`。1時間足の値幅 |
| `volume_change` | 直前の1時間足からの出来高変化率 |
| `sma_ratio_6` | 現在値と6時間単純移動平均の乖離率 |
| `sma_ratio_24` | 現在値と24時間単純移動平均の乖離率 |
| `ema_ratio_12` | 現在値と12時間指数移動平均の乖離率 |
| `rsi_14` | 14本の値動きから計算したRSI |
| `volatility_12` | 直近12本のリターンの標準偏差 |
| `atr_14` | 14本の平均真の値幅を現在値で割った値 |
| `hour` | 時刻の0〜23の時間番号 |
| `weekday` | 曜日の番号（月曜=0、日曜=6） |

初期計算に必要な過去データが不足する行や、目的変数を計算できない行は削除します。
そのため、CSVの行数と`evaluation.json`の`rows`は一致しません。

### 予測する値

予測対象は価格そのものではなく、次の対数リターンです。正解値のCloseも、設定した
`feature_intervals`のうち最も短い時間足に揃えます。

```text
target_return = log(基準時間足での将来のClose / 基準時間足での現在のClose)
```

例えば基準時間足が5分足の場合、各5分足時点から次の本数先を使用します。

| 予測期間 | 先の5分足本数 |
|---|---:|
| 30分後 | 6本 |
| 1時間後 | 12本 |
| 3時間後 | 36本 |

したがって、5分足データでは最後の36本分は3時間後の正解値を作れず、3時間モデルの
学習対象から除外されます。ここでいう「本後」は時刻の差ではなく、利用可能な足を
順番に数えた本数です。株式市場の夜間・週末・昼休みをまたぐ場合も、次の観測された
5分足を1本目として数えます。基準時間足が10分足なら、それぞれ3本・6本・18本先です。

`30m`、`1h`、`3h`ごとに別のモデルを作ります。さらにこの値を使って方向を分類します。

- `Short`: `target_return <= -0.0005`
- `Neutral`: `-0.0005 < target_return < 0.0005`
- `Long`: `target_return >= 0.0005`

この`0.0005`は約0.05%の閾値です。予測値を価格に戻す場合は、
`現在のClose * exp(予測target_return)`で計算できます。

## 実行

```powershell
.\.venv\Scripts\python.exe forecasting.py --data .\AAPL.csv --scope both
```

共通モデルは複数CSVをまとめて指定します。

```powershell
.\.venv\Scripts\python.exe forecasting.py --data .\AAPL.csv .\MSFT.csv --scope both
```

生成物は `models\` のjoblibモデルと `evaluation.json` です。評価値は将来の収益を保証せず、
自動売買やレバレッジ判断は行いません。現状のCSVが1銘柄だけの場合、共通モデルと個別モデルの
差は比較できないため、複数銘柄・十分な期間のデータを追加してから比較してください。

## evaluation.jsonの見方

`evaluation.json`の最上位項目は、今回の学習設定を表します。

| 項目 | 意味 |
|---|---|
| `scope` | `common`は共通モデル、`individual`は銘柄別モデル、`both`は両方を作成 |
| `model` | 使用したモデル。`lightgbm`または`logistic` |
| `horizons` | 予測時間ごとの評価結果。`30m`、`1h`、`3h`がキー |

各ホライズンの`common`または`individual`の中にある項目は次の意味です。

| 項目 | 意味 |
|---|---|
| `rows` | 特徴量と予測対象が揃い、欠損除去後に残った全データ行数 |
| `train_rows` | 学習に使った行数。時系列の先頭80% |
| `test_rows` | 評価に使った行数。時系列の末尾20% |
| `return_mae` | リターン予測の平均絶対誤差。小さいほど良い |
| `return_rmse` | リターン予測の二乗平均平方根誤差。小さいほど良い。大きな誤差をより強く評価 |
| `direction_accuracy` | Short/Neutral/Longの正解率。1.0が全問正解、0.333程度が3クラスの単純な目安 |

例えば現在の`evaluation.json`の`30m`では、`rows=118`のうち94行で学習し、
24行で評価しています。`return_mae=0.002748`は、予測リターンと実績リターンの
差の絶対値が平均約0.2748ポイント（リターンの小数表現）だったという意味です。
`direction_accuracy=0.4167`は、24件中およそ10件で方向クラスが一致したことを示します。

`return_mae`と`return_rmse`はリターンの指標であり、価格のドル差ではありません。
また、現在のAAPL.csvは1銘柄だけなので、`common`と`individual`の入力データが同じで、
評価値も同じになります。複数銘柄CSVを指定した場合に初めて、共通モデルと個別モデルを
意味のある形で比較できます。

保存済みの個別モデルで予測する場合、`predict.py`はCSVを読み込まず、
yfinanceからデータを取得します。予測の基準時点を入力し、`latest`なら最新足、
`1d`なら最新足から1日前、`3h`なら最新足から3時間前の足を使います。

yfinanceへの通信に失敗した場合や空のデータが返った場合は、最大3回再試行します。
再試行間隔は1秒、2秒です。3回とも失敗した場合は、銘柄・試行回数を含むエラーを
表示します。

`prediction-time=latest`では、最初から60日分を取得せず、まず4日分を取得します。
ただし、`feature_intervals`のウォームアップに必要な本数が返らなかった場合は、
7日、30日、60日の順に取得範囲を拡張します。例えば`["5m", "1h", "3h"]`では
3時間足の指標計算に必要な履歴も確保します。

```powershell
.\.venv\Scripts\python.exe predict.py `
  --ticker AAPL --interval 5m --bars 10000 --model-dir .\models
```

基準時点をコマンドライン引数で指定することもできます。

```powershell
.\.venv\Scripts\python.exe predict.py `
  --ticker AAPL --interval 5m --bars 10000 --prediction-time 1d `
  --model-dir .\models
```

## モデルセットによるディレクトリ管理

共通モデルと銘柄別モデルを別の実験として管理するため、モデルセット単位で設定と
生成物をまとめます。標準の構成は次のとおりです。

```text
model_sets/
└─ default/
   ├─ model_set.json       # モデルセットの設定
   ├─ data/                # yfinanceから取得したCSV
   │  ├─ AAPL.csv
   │  └─ MSFT.csv
   ├─ models/              # joblibモデル
   │  ├─ common_30m.joblib
   │  ├─ AAPL_30m.joblib
   │  └─ ...
   ├─ evaluation.json      # 学習時の詳細な評価結果
   ├─ evaluation.csv       # 比較用の評価結果
   └─ predictions/         # predict.pyのJSON出力
```

`evaluation.json`とjoblibモデルは、必ずモデルセットのディレクトリに対応づけます。
つまり、`model_sets/default/models`がモデルファイルの保存先であり、
`model_sets/default/evaluation.json`と`model_sets/default/evaluation.csv`がそのセットの
評価結果です。プロジェクト直下の
`models`や`evaluation.json`とは別管理になります。

### モデルセットJSON

`model_set.json`は次の情報を持ちます。

```json
{
  "name": "default",
  "scope": "both",
  "model": "lightgbm",
  "data": {
    "interval": "5m",
    "bars": 10000,
    "directory": "data",
    "feature_intervals": ["1h"]
  },
  "symbols": [
    {"ticker": "AAPL", "name": "Apple", "market": "us_stock"},
    {"ticker": "MSFT", "name": "Microsoft", "market": "us_stock"}
  ]
}
```

| JSON項目 | 意味 |
|---|---|
| `name` | モデルセットの名前 |
| `scope` | `common`、`individual`、`both`。共通モデル・銘柄別モデルの作成範囲 |
| `model` | `lightgbm`または`logistic` |
| `data.interval` | yfinanceから取得する足の間隔。例：`5m`、`15m`、`1h` |
| `data.bars` | 各銘柄について保存する最新の足の本数 |
| `data.directory` | セットディレクトリから見たCSV保存先 |
| `data.feature_intervals` | 学習特徴量に使う時間足。例：`["5m"]`、`["5m", "30m"]`、`["1h"]` |
| `symbols[].ticker` | yfinanceのティッカー。CSVファイル名にも使用 |
| `symbols[].name` | 人が読むための銘柄名 |
| `symbols[].market` | `us_stock`などの市場分類。現在は記録用 |

`interval`はダウンロードする元データの足の間隔です。`feature_intervals`は学習と
予測に使う特徴量の時間足です。`1h`を含める必要はありません。例えば`["5m"]`なら
5分足のみ、`["5m", "30m"]`なら5分足と30分足、`["1h", "10m"]`なら1時間足と10分足を
使います。複数指定時は最も短い時間足を基準にし、他の時間足の特徴量を時刻で結合します。
追加時間足の特徴量には`_30m`や`_10m`のような接尾辞が付きます。

`5m`を指定しても、`feature_intervals`が`["1h"]`なら学習特徴量は1時間足から作られます。
`bars`は
ダウンロード元から取得する本数ではなく、保存する最新本数です。yfinanceの期間制限
により取得できる期間は足種別で異なります（例えば1分足は短期間、5分足は通常60日以内）。

### モデルセットの作成

`create_model_set.py`は、セットディレクトリ、`data`、`models`、デフォルトJSONを
まとめて作成します。

```powershell
.\.venv\Scripts\python.exe create_model_set.py --name default
```

銘柄・足種別・本数を指定する例です。

```powershell
.\.venv\Scripts\python.exe create_model_set.py `
  --name us-stocks --scope both --interval 5m --bars 10000 `
  --symbols AAPL MSFT NVDA AMZN GOOGL
```

既存の`model_set.json`は上書きしません。別の実験を作る場合は別名を指定してください。

1時間足のみと、1時間足＋10分足を比較する場合は、モデルセットを2つ作ります。

```powershell
.\.venv\Scripts\python.exe create_model_set.py `
  --name hourly-only --feature-intervals 1h --symbols AAPL

.\.venv\Scripts\python.exe create_model_set.py `
  --name hourly-plus-10m --feature-intervals 1h 10m --symbols AAPL
```

5分足だけ、または5分足＋30分足を使う例です。

```powershell
.\.venv\Scripts\python.exe create_model_set.py `
  --name five-minute-only --feature-intervals 5m --interval 5m --symbols AAPL

.\.venv\Scripts\python.exe create_model_set.py `
  --name five-and-thirty-minute --feature-intervals 5m 30m `
  --interval 5m --symbols AAPL
```

各セットで同じ銘柄・同じ期間のCSVを取得して学習すると、それぞれの
`evaluation.json`の`return_mae`、`return_rmse`、`direction_accuracy`を比較できます。
データ期間や検証方法を揃えないと、時間足の違いによる比較にならないため注意してください。

### yfinanceからCSVを取得

```powershell
.\.venv\Scripts\python.exe download_yfinance.py `
  --config .\model_sets\default
```

`--config`を省略すると、実行時にモデルセットのディレクトリを入力できます。

JSONの`symbols`に列挙した各ティッカーを取得し、`data.directory`に
`<ticker>.csv`として保存します。取得結果が空の場合はエラーにして、空のCSVで学習を
続けないようにしています。

### JSONから学習・予測を起動

`run_model_set.py`はJSONを読み込み、必要なパスと設定を組み立てて既存の
`forecasting.py`または`predict.py`を起動します。`train`ではモデルセットの
`data`に保存されたCSVを使い、`predict`ではCSVを使わず、各銘柄をyfinanceから取得します。

```powershell
# CSV取得後に、共通・銘柄別モデルを学習
.\.venv\Scripts\python.exe run_model_set.py `
  --config .\model_sets\default train

# 保存済みモデルで予測
.\.venv\Scripts\python.exe run_model_set.py `
  --config .\model_sets\default predict
```

対話入力で実行する場合は、次のようにします。

```powershell
.\.venv\Scripts\python.exe download_yfinance.py
.\.venv\Scripts\python.exe run_model_set.py train
.\.venv\Scripts\python.exe run_model_set.py predict
```

`run_model_set.py`でactionも省略できます。その場合、最初に`train`または`predict`を
入力します。入力するパスは`model_sets\default`のようなモデルセットディレクトリです。
その中の`model_set.json`が自動的に読み込まれます。`model_set.json`を直接指定する
従来形式も引き続き使用できます。

`predict`を選択した場合は、予測基準時点を入力します。

```text
Enter prediction reference (latest / 1d / 3h / 30m): 1d
```

`latest`以外では、指定した経過時間以前の最新足を基準にします。例えば`1d`は
「最新データから1日前」であり、yfinanceから必要な履歴を取得したうえで、
その時点以前の最新観測足を選択します。`--bars`は特徴量計算に使う最大入力本数で、
特徴量のウォームアップに必要な本数を下回る場合は、内部で必要本数まで増やして
読み込みます。最後まで完全な特徴量行を作れない場合は、取得した本数・必要本数・
時間足を含むエラーを表示します。通常は`--bars`を増やすか、モデルと
`feature_intervals`を一致させてください。

predict結果はJSON配列として保存されます。`run_model_set.py`経由では
`model_sets/<セット名>/predictions/`に、銘柄選択・UTC時刻を含む重複しない
ファイル名で保存されます。`all`を選択した場合は、全銘柄の結果とcommon/individualの
結果を同じ1つの配列へまとめます。直接`predict.py`を実行した場合も、1件の結果を
要素とする配列として保存されます。

```json
[
  {
    "symbol": "BTC-JPY",
    "scope": "common",
    "forecasts": {}
  },
  {
    "symbol": "BTC-JPY",
    "scope": "individual",
    "forecasts": {}
  }
]
```

直接`predict.py`を実行した場合は既定で`predictions/`に保存され、
`--output`または`--output-dir`で保存先を変更できます。

`run_model_set.py`の銘柄入力で`all`を選択すると、モデルセットに登録された全銘柄を
順番に予測し、それぞれJSONを保存します。

対話入力で空欄や不正な値を入力した場合は、その場でエラー終了せず、正しい値を
入力するまで聞き返します。対象はモデルセットのパス、`train`/`predict`、
予測基準時点、銘柄選択、データ開始日時です。

### モデルの上書き確認

学習先の`models`に既存の`.joblib`モデルがある場合、学習開始前に上書き確認を表示します。
`y`または`yes`を入力した場合だけ上書きし、それ以外は学習を中止します。
`forecasting.py`を直接実行する場合は、確認を省略して上書きするために
`--overwrite`を指定できます。

`scope`が`both`の場合、学習では共通モデルと銘柄別モデルの両方を作成します。
予測では各CSVに対して共通モデルと銘柄別モデルをそれぞれ実行します。必要なCSVが
一つでもない場合は、誤った入力で学習・予測しないよう起動前にエラーになります。

### predict時の入力データ量

`predict`時にモデルへ渡すデータ量は、次の2段階に分けて理解してください。

1. `model_set.json`の`data.interval`で指定した元データを、`download_yfinance.py`が
   `data.bars`本保存します。例えば`interval: "5m"`、`bars: 10000`なら、各CSVに
   最新の5分足10000本が保存されます。
2. `predict.py`はそのCSV全体を読み込み、`feature_intervals`で指定した各時間足の
   指標を計算します。ただし、モデルに実際に入力するのは最新時点の**特徴量1行**です。
   その1行を作るために、CSV内の過去バーを使います。

したがって、`["5m"]`の場合は「5分足10000本を使って計算し、最新5分足時点の
特徴量1行をモデルへ入力」します。`["5m", "30m"]`の場合は「5分足10000本から
5分足と30分足の特徴量を計算し、両方を結合した最新時点の特徴量1行を入力」します。
モデルへ10000本をそのまま一度に入力しているわけではありません。必要な過去本数は
指標ごとに異なり、例えば24本移動平均なら各特徴量時間足の過去24本以上が必要です。
データ不足で特徴量を作れない先頭行は除外されます。

### evaluation.csv

学習時には`evaluation.json`と同じディレクトリに、比較用の
`evaluation.csv`も自動生成します。列は次の4つです。

| 列 | 意味 |
|---|---|
| `ticker` | 銘柄別モデルのティッカー。共通モデルは`common` |
| `period` | 予測期間。`30m`、`1h`、`3h` |
| `bars` | 評価に使ったテストデータの足数。`evaluation.json`の`test_rows`と同じ |
| `accuracy` | Short/Neutral/Longの方向正解率。`evaluation.json`の`direction_accuracy`と同じ |

`evaluation.csv`は、複数モデルセットの結果を表計算ソフトなどで簡単に比較する
ための要約ファイルです。リターンのMAE/RMSEは含まれないため、詳細な評価には
`evaluation.json`を使用してください。

## 過去データの取引シミュレーション

保存済みモデルを過去のCSVへ時系列順に適用し、1時間後予測と3時間後予測に
従った取引の損益をシミュレーションできます。

```powershell
.\.venv\Scripts\python.exe simulation.py `
  --model-set .\test_5m1h3h
```

`--model-set`にはモデルセットのディレクトリ、または`model_set.json`を指定します。
シミュレーション対象はモデルセットの`data`内CSVです。`scope`が`both`の場合は
共通モデルと銘柄別モデルを別々に評価し、1時間後モデルと3時間後モデルも別々に
集計します。

### シミュレーションの売買ルール

- 予測時点は各1時間足のCloseです。
- 予測がLongまたはShortの場合、次の1時間足のOpenで建てます。
- 1時間モデルは建玉から1時間後、3時間モデルは建玉から3時間後の1時間足Openで決済します。
- Neutralの場合、その時点で保有中の同一シミュレーションのポジションを次の足のOpenで決済し、新規建玉は作りません。
- 毎回の新規建玉には、現在の仮想資産を「その建玉を含む同時保有数」で割った金額を配分します。
- 初期資金は1.0（100%）として扱うため、金額ではなく`equity`と`cumulative_return`で確認します。
- Long/Shortの両方を有効にします。
- 片道手数料率と片道スリッページ率の既定値は各`0.0005`（0.05%）です。

結果はモデルセット直下の`equity.csv`に出力されます。主な列は次のとおりです。

| 列 | 意味 |
|---|---|
| `timestamp` | ポジションを決済した時刻 |
| `ticker` / `scope` / `horizon` | 銘柄、共通/個別、1時間/3時間モデル |
| `direction` | LongまたはShort |
| `entry_price` / `exit_price` | 建値と決済値 |
| `allocation` | その取引へ配分した仮想資産 |
| `gross_return` | コスト控除前の取引リターン |
| `net_return` | 手数料・スリッページ控除後の取引リターン |
| `pnl` | 仮想資産に対する損益 |
| `equity` | 取引決済後の仮想資産 |
| `cumulative_return` | 初期値1.0からの累積損益率 |
| `drawdown` | 過去最高`equity`からの下落率 |

コストや出力先を変更する場合:

```powershell
.\.venv\Scripts\python.exe simulation.py `
  --model-set .\test_5m1h3h `
  --scope individual `
  --fee-rate 0.0005 `
  --slippage-rate 0.0005 `
  --output .\test_5m1h3h\equity-individual.csv
```

これは将来情報を使わないよう、各時点でその時点までの特徴量を使って予測する
ウォークフォワード形式です。ただし、モデル自体は学習済みモデルをそのまま使うため、
学習期間とシミュレーション期間が重なる場合は結果が過大評価される可能性があります。

## GitHub Actionsでの実行

`.github/workflows/finance.yml`は手動実行用のworkflow_dispatchを提供します。
GitHubリポジトリの`Actions`から`Finance model workflow`を選択し、次の入力を指定します。

| 入力 | 値 |
|---|---|
| `action` | `download`、`train`、`predict`、`simulate`、`all` |
| `config` | モデルセットのディレクトリ、例:`model_sets/default` |
| `start` | データ取得開始日、例:`2026-04-01`。空欄は各時間足の既定期間 |
| `symbol` | 予測銘柄または`all` |
| `prediction_time` | `latest`または`1d`、`3h`など |
| `scope` | `common`、`individual`、`both` |
| `overwrite` | 学習時に既存joblibを上書きするか |

GitHub Actionsでは対話入力を使用しません。入力値は既存スクリプトのコマンドライン引数へ
変換されます。`all`アクションは、download、train、predict、simulateを順番に実行します。

実行後は`finance-results-<run_id>`という名前のArtifactから、CSV、joblib、evaluation、
prediction JSON、equity CSVをダウンロードできます。Artifactは既定で30日間保持します。

## データ形式

`yf.download(...).to_csv(...)` で保存したyfinanceの3行ヘッダーCSVを入力にします。
暗号資産や追加の株式データ取得（CCXT/yfinance）は次段階で追加できるよう、
学習処理は市場に依存しないOHLCV形式に分離しています。
