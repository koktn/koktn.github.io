---
title: HSTUとGenerative Recommenders：行動履歴から1.5兆パラメータの推薦モデルへ
description: MetaのHSTU論文を、推薦のsequential transduction化、pointwise attention、Stochastic Length、M-FALCON、公開実験と本番環境評価の限界から解説します。
publishedAt: 2026-09-29
updatedAt: 2026-10-01
category: AI
tags:
  - Recommendation System
  - Generative Recommendation
  - HSTU
  - Transformer
  - ICML
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値と手法は原論文と公開実装を確認して記載していますが、利用時は原文も確認してください。

大規模な推薦システムは、ユーザーID、商品ID、属性、集計値などを人手で設計した特徴量へ変換し、複数のネットワークで組み合わせるDLRM（Deep Learning Recommendation Model）を長く使ってきました。MetaのJiaqi Zhaiらによる「[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)」は、この構成をユーザーの行動列を処理するGenerative Recommender（GR）へ置き換え、そのためのエンコーダーHSTU（Hierarchical Sequential Transduction Unit）を提案したICML 2024論文です。対象はarXiv v3（2024年5月6日改訂）です。

この論文の価値は、推薦のランキングとretrievalをsequential transductionとして統一し、学習と推論の重複計算を減らす仕組みまで含めて、1.5兆パラメータのモデルを本番環境へ展開した点にあります。一方、「生成推薦」という名前からLLMが説明文や商品IDのトークン列を生成する手法を想像すると、論文の中心を取り違えます。

## DLRMから行動系列へ何を変えたのか

従来の産業向けDLRMは、数値特徴量とカテゴリ特徴量を別々に処理し、埋め込み、特徴量の相互作用、multi-task ヘッドなどを組み合わせます。新しい特徴量を加えるたびに、集計方法、保存、学習時の結合、serving時の整合性を管理する必要があります。

GRは、この異種特徴量を時間順のトークン列へまとめます。

```text
従来のDLRM
  user/item ID + 集計値 + 属性 + cross feature
    → 個別のembedding・network
    → feature interaction
    → ranking / retrieval

Generative Recommender
  content Φ0, action a0, content Φ1, action a1, ...
    + 時間変化の遅いカテゴリfeature
    → 統合された時系列
    → HSTU
    → ranking / retrieval
```

主系列には、ユーザーが接触したコンテンツと、その後のlike、skip、視聴完了、shareなどの行動を置きます。言語や居住地、follow中のcreatorのように変化が遅いカテゴリ特徴量は、同じ値が連続する区間を圧縮して主系列へマージします。

CTRや回数のような数値特徴量はinteractionごとに変わるため、すべてをトークン化すると保存領域と計算量が膨らみます。論文は、集計元となるカテゴリ情報が既に系列へ入っているなら、十分に表現力のあるtarget-aware modelが必要な集計を学べるとして、数値特徴量をモデル側の表現へ置き換えます。有限の履歴とモデル容量で完全に同じ情報を復元できる保証ではなく、本番環境実験で有効性を確かめた設計仮説です。

## ランキングとretrievalを同じsequential transductionで表す

入力トークン列を`x0, x1, ...`、各位置に対応する出力を`y0, y1, ...`とすると、論文は2つのタスクを次のように表します。

| タスク | 入力 | 予測するもの |
| --- | --- | --- |
| ランキング | コンテンツと行動を交互に並べた列 | 各コンテンツに対する次の行動 |
| retrieval | コンテンツと行動を対にした履歴 | positive 行動につながる次のコンテンツ |

ランキングでは、候補コンテンツを履歴の末尾へ置き、その候補に対する行動確率を予測します。これにより、候補と過去の履歴を早い段階で相互作用させるtarget-awareな処理になります。実運用では、コンテンツ位置の出力を小さなネットワークへ通し、複数タスクの予測へ変換します。

retrievalでは、履歴から次にpositive 行動を得るコンテンツの分布を学びます。ただし、TIGERのように短いSemantic IDを自己回帰生成する方式とは異なり、この論文の特徴量空間には大規模なatomic IDが残ります。著者らはretrieval時にMIPS、beam search、階層的retrievalなどを利用できるとしています。

ここでいう「generative」のもう1つの意味は学習例の作り方です。従来のimpression単位の学習では、同じユーザー履歴をtargetごとに何度もエンコーダーへ通します。GRは1本の系列から複数位置の損失をまとめて計算し、エンコーダーの仕事を複数targetで共有します。論文の解析では、ユーザー系列長に反比例する割合でサンプルすることで、全体の学習計算量を長さ`N`に対して1段階減らせます。[論文Section 2](https://arxiv.org/pdf/2402.17152#page=2)

## HSTUはTransformerと何が違うのか

HSTUは、同じ層をresidual connectionで積み重ねるcausalなsequence encoderです。論文のFigure 3は、左に従来のDLRM、右に3層だけ描いたHSTUを並べています。左側ではfeature extraction、特徴量の相互作用、表現変換を別々のmoduleが担当します。右側では、同じHSTU layerを繰り返して3つの役割をまとめます。

![論文Figure 3。左側ではDLRMが数値・カテゴリfeatureの抽出、feature interaction、表現変換を別moduleで処理し、右側ではHSTU layerを積み重ねて処理する](/img/posts/hstu-dlrm-gr-figure-3.png)

*出典：Jiaqi Zhai et al., [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations, Figure 3](https://arxiv.org/pdf/2402.17152#page=4)。原図からFigure 3を切り出して掲載。画像部分は[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)です。*

1層の処理を先に一本の流れへすると、次のようになります。

```text
入力 X（N個のtoken × d次元）
  ↓ 1つのlinear projection + SiLU
大きなtensorを U, V, Q, K に分割
  ↓
QKᵀ + 位置・時間bias → SiLU → attention weight A
  ↓
A × V → 履歴から集めたcontext
  ↓
LayerNorm(context) ⊙ U → f₂でd次元へ戻す
  ↓
元の入力をresidual connectionで加えて次のlayerへ
```

ここで`X`は、長さ`N`の系列を`d`次元ベクトルで表した行列です。1行が「ある時刻に表示されたコンテンツ」「そのときの行動」「途中へマージした属性」など、1トークンに対応します。HSTU layerは、各トークンが過去のどのトークンを参照し、何を受け取り、その情報を出力のどの成分へ通すかを計算します。

### 1. U、V、Q、Kは同じ入力を見る4つの役割

最初に`X`をlearnableなlinear layer `f₁`へ通し、SiLUを適用します。その大きな出力tensorを4つに分割したものが`U`、`V`、`Q`、`K`です。「同じベクトルを4個copyする」のではなく、学習された異なる重みにより、同じ入力から役割別の表現を一度に作ります。

| 記号 | 役割 | 直感的な読み方 |
| --- | --- | --- |
| `Q`（クエリ） | 現在位置が過去から探したい情報を表す | 「この候補を判断するため、何を知りたいか」 |
| `K`（key） | 各履歴トークンが何に関係するかを表す | 「この履歴は何についての情報か」 |
| `V`（value） | 選ばれた履歴から実際に運ぶ内容を表す | 「参照されたとき、何を渡すか」 |
| `U`（gate） | 集めた情報を出力の成分ごとに通す量を決める | 「この位置では、どの情報を効かせるか」 |

`Q`、`K`、`V`はTransformerとほぼ同じ役割です。`U`はHSTUの特徴で、attention後の出力を制御するgateです。論文のshapeでは、`Q`と`K`は`h × N × d_qk`、`U`と`V`は`h × N × d_v`です。`h`はattention ヘッド数で、複数の観点から同じ系列を見るための軸です。

4種類を別々の層で順番に計算するのではなく、`f₁(X)`という1回の大きなprojectionにまとめてから分割するのは、GPU上で計算をバッチ化・fuseしやすくするためです。

### 2. QKᵀで「どの履歴を見るか」を決める

現在位置`i`のクエリ`Qᵢ`と、履歴位置`j`のkey `Kⱼ`のdot productを取ると、両者の関連度を表すscoreが得られます。そこへ、系列上の距離と実時間の差を表すrelative attention biasを足します。

```text
score(i, j) = Qᵢ · Kⱼ + position_bias(i, j) + time_bias(i, j)
```

たとえば同じ「ランニングシューズを見た」という行動でも、5分前と半年前では次の推薦への影響を変えられます。系列上で何トークン離れているかだけでなく、実際の経過時間もバイアスに入れるのが推薦向けの設計です。causal maskがあるため、位置`i`から未来のトークンは見えません。ランキング候補を履歴の末尾へ置けば、その候補は自分より前の行動だけを参照できます。

通常のTransformerは、各クエリについてscoreをsoftmaxへ通し、履歴全体のweight合計を1にします。HSTUは代わりに各scoreへSiLUを個別適用します。

```text
Aᵢⱼ = SiLU(score(i, j))
contextᵢ = Σⱼ Aᵢⱼ Vⱼ
```

したがって`Aᵢⱼ`は合計1の確率ではありません。あるtopicに関係する履歴が増えれば、対応するvalueが複数回足し合わされます。softmaxは「10件の中でどれが最重要か」を表しやすい一方、1件しかない人と100件ある人でもweight総和は1です。HSTUがpointwise aggregationを使う狙いは、推薦で重要な興味の相対順位だけでなく、関連行動がどれだけ蓄積したかという強度も残すことです。

この仕組みはscoreの規模が系列長や履歴内容によって変わりやすいため、`A × V`で文脈を作った後にLayerNormを入れて学習を安定させます。論文では、このLayerNormがpointwise pooling後に必要だとしています。

### 3. Uで文脈をgateし、次の層へ渡す

履歴から得た文脈をnormalizeした後、同じ位置の`U`と要素ごとに掛けます。

```text
Zᵢ = LayerNorm(contextᵢ) ⊙ Uᵢ
Yᵢ = f₂(Zᵢ)
```

`⊙`はベクトルの要素ごとの積です。`Uᵢ`は履歴トークンを選ぶweightではなく、集約済み文脈の各channelを開閉するgateです。たとえば文脈に「ブランド」「価格帯」「カテゴリ」「直近性」に対応する成分があるなら、現在の候補やユーザーの状態に応じて、必要な成分を強く通し、不要な成分を弱めるイメージです。これは厳密に各channelが人間の概念へ対応するという意味ではなく、gateの働きを理解するための比喩です。

最後に`f₂`がmulti-headの出力をモデル次元`d`へ戻します。通常のTransformerはattentionの後に大きなfeed-forward networkを持ちますが、HSTUは`U`によるgateと`f₂`で表現変換を担い、独立したfeed-forward blockを使いません。論文は`LayerNorm(A × V) ⊙ U`をSwiGLUの比較案として解釈でき、従来DLRMのMixture of Expertsに近い条件付き計算も要素積で表現できると説明しています。

Figure 3の`Add&Norm`はresidual connectionです。層が作った`Y(X)`だけで入力を上書きせず、元の`X`を足してnormalizeしてから次の層へ渡します。これにより、元のトークン情報を残しながら、層を重ねるごとにより長い履歴とのinteractionを追加できます。

### 具体例：シューズ候補をランキングする場合

ユーザー履歴の末尾へ「新しいランニングシューズ」という候補を置いた場合を考えます。

1. 候補位置の`Q`が「この候補と関係する過去のシグナル」を探す。
2. 過去のシューズ閲覧、スポーツ用品購入、skipなどの`K`との関連度を計算する。
3. 位置・時間バイアスにより、直近の行動と古い行動の影響を調整する。
4. 関連度を使って、それぞれの履歴が持つ`V`を足し合わせる。
5. 候補位置の`U`が、集めた文脈のうち今回の予測へ通す成分を調整する。
6. `f₂`とresidual connectionを経て、次のHSTU layerまたは予測ヘッドへ渡す。

この処理を複数層で繰り返すことで、「この候補と直前の1行動」の関係だけでなく、複数の行動や属性を組み合わせたパターンを表現します。

通常のTransformerと比べた中核的な違いは、attention weightを系列全体のsoftmaxで正規化せず、SiLUを使ったpointwise aggregated attentionにしたことです。位置と経過時間をrelative attention biasへ含めるため、順序だけでなく時間間隔も扱えます。

非定常なvocabularyを模したsynthetic streaming dataでは、HR@10が標準Transformerの`0.0442`、softmax版HSTUの`0.0617`、pointwise版HSTUの`0.0893`でした。pointwise版はsoftmax版より44.7%の相対改善ですが、これは合成データ上のablationであり、本番環境全体における単独効果ではありません。[論文Table 2](https://arxiv.org/pdf/2402.17152#page=5)

## 長い履歴を現実の計算量へ収める4つの工夫

HSTU layerだけでは、最大8,192トークンの履歴、巨大なID vocabulary、数千から数万のランキング候補を処理できません。論文のシステムは複数の最適化を組み合わせています。

### 1. 長さの異なる系列をpaddingせず処理する

ユーザーごとの履歴長は大きく偏ります。HSTUのGPU kernelはraggedな系列をgrouped GEMMとして処理し、padding部分の計算を避けます。論文では、この実装だけで2〜5倍のスループット向上を報告しています。

### 2. Stochastic Lengthで学習時の履歴を間引く

ユーザー行動には、短期と長期の両方で似た行動が繰り返されるという仮定があります。Stochastic Length（SL）は、長い履歴を一定確率でsubsequenceへ縮め、attention計算のsparsityを増やします。30日分の履歴を使う本番環境相当の設定では、最大長4,096、`α = 1.6`のとき80.5%のsparsityになり、実際の系列は多くの場合776トークンまで縮みました。適切な`α`では主要タスクのNormalized Entropy（NE）の悪化が`0.002`未満だったと報告されています。

これは推論時に常に80%を捨てるという話ではなく、主に高コストな学習を安くするsamplingです。Appendixの比較では、同程度のsparsityを作るzero-shot／追加学習による長さ外挿よりSLのNE悪化が小さかったものの、非公開データに基づく結果です。[論文Section 3.2とAppendix F](https://arxiv.org/pdf/2402.17152#page=5)

### 3. 活性化関数とembedding optimizerのメモリを減らす

推薦モデルは大きなバッチを必要とするため、パラメータだけでなく活性化のメモリが制約になる箇所になります。論文の見積もりでは、HSTUはlinear layerの削減とoperator fusionにより1層あたりの活性化状態をTransformerの`33d`から`14d`へ減らし、2倍を超える深さを同じメモリで扱えます。

一方、1.5兆パラメータという規模を理解するには巨大なID embeddingを無視できません。論文は、100億語彙、512次元、fp32のAdamで埋め込みとoptimizer 状態だけで60TBになる例を示します。row-wise AdamWとoptimizer 状態のDRAM配置により、HBM使用量を1要素あたり12 byteから2 byteへ減らしています。「1.5兆パラメータ」は、1.5兆個のdense Transformer weightを持つLLMと同じ構成ではありません。

### 4. M-FALCONで候補間の履歴計算を共有する

ランキングでは、同じユーザー履歴に対して最大数万件の候補をscoreします。候補ごとに履歴全体を再計算すると、target-aware attentionの利点がそのままコストになります。

M-FALCON（Microbatched-Fast Attention Leveraging Cacheable OperatioNs）は、候補をmicrobatchへまとめ、attention maskとrelative biasを調整して履歴側の計算を共有します。さらにエンコーダーのKV cacheをmicrobatch間で再利用します。候補数を`m`、microbatch内の候補数を`bₘ`、履歴長を`n`とすると、候補ごとのcross-attentionに相当する計算を`O(bₘn²d)`から`O((n + bₘ)²d)`へ変えます。`bₘ`が`n`より十分小さければ、支配項はほぼ`O(n²d)`です。[論文Section 3.4](https://arxiv.org/pdf/2402.17152#page=6)

## Public datasetではどこまで改善したか

公開評価はMovieLens-1M、MovieLens-20M、Amazon Reviews Booksで行われています。比較手法は、sampled softmaxを使う強いSASRec recipeです。HSTUはSASRecと層数・ヘッド数などを揃え、HSTU-largeは層数を4倍、ヘッド数を2倍にしています。次の値はarXiv v3のTable 4です。

| Dataset | Model | HR@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| ML-1M | SASRec | 0.2853 | 0.1603 |
| ML-1M | HSTU | 0.3097（+8.6%） | 0.1720（+7.3%） |
| ML-1M | HSTU-large | 0.3294（+15.5%） | 0.1893（+18.1%） |
| ML-20M | SASRec | 0.2906 | 0.1621 |
| ML-20M | HSTU | 0.3252（+11.9%） | 0.1878（+15.9%） |
| ML-20M | HSTU-large | 0.3567（+22.8%） | 0.2106（+30.0%） |
| Books | SASRec | 0.0292 | 0.0156 |
| Books | HSTU | 0.0404（+38.4%） | 0.0219（+40.6%） |
| Books | HSTU-large | 0.0469（+60.6%） | 0.0257（+65.8%） |

括弧内はSASRecに対する相対改善です。HSTUと同じ構成の比較でも全データセットで改善し、モデルを大きくするとさらに伸びています。ただし、この評価はfull-shuffle・multi-epoch学習です。1回だけ時系列順に流す本番環境のストリーミング学習とは条件が異なり、この65.8%をそのままオンライン効果として読んではいけません。[論文Table 4](https://arxiv.org/pdf/2402.17152#page=7)

## 本番環境評価の12.4%をどう読むか

産業規模のエンコーダー比較では、1000億件のDLRM相当exampleを1 passで学習し、1 ジョブあたり64〜256基のNVIDIA H100を使っています。ランキングはmain engagement task（E-Task）とmain consumption task（C-Task）のNE、retrievalはlog perplexityで比較しています。

処理全体比較では、retrievalのGRを新しい候補参照元として加えると、匿名化されたオンライン指標がE-Taskで`+6.2%`、C-Taskで`+5.0%`でした。主要なDLRM 参照元をGRで置き換えた場合は、それぞれ`+5.1%`、`+1.9%`です。ランキングではGRが`+12.4%`、`+4.4%`を記録しました。

論文がabstractで掲げる「オンラインA/Bテストで12.4%改善」は、このうちランキングのE-Taskにおける最大値です。すべてのsurfaceや指標が12.4%改善したわけではありません。E-TaskとC-Taskの具体的定義、トラフィック量、実験期間、信頼区間は公開されていないため、別のサービスへ移したときの効果量は推定できません。

効率面では、8,192トークン、`d = 512`、8 ヘッド、H100、bfloat16のエンコーダー比較で、FlashAttention 2を使うTransformerに対して学習最大15.2倍、推論最大5.6倍でした。処理全体の本番のランキングでは、FLOPsが285倍のGRが、1,024候補で1.50倍、16,384候補で2.99倍のQPSを達成しています。この結果はHSTU単体ではなく、ragged kernel、M-FALCON、cachingを含むシステム全体の値です。[論文Section 4.2–4.3](https://arxiv.org/pdf/2402.17152#page=7)

## Recommendationにもスケーリング則は現れたのか

著者らは、HSTUの層数、埋め込み次元、ヘッド数、系列長、retrievalのnegative数などを変え、学習計算量を約3桁の範囲で増やしました。DLRMは約2,000億パラメータ付近で性能が飽和した一方、GRは1.5兆パラメータまで改善が続き、retrievalのHR@100／HR@500とランキングのNEが計算量に対してpower lawに従ったと報告しています。

最大構成は、系列長8,192、埋め込み次元1,024、HSTU 24層です。ストリーミング学習なので、計算量は365日分へ正規化してGPT-3やLLaMA 2の学習規模と比較されています。著者らは、言語モデルと違って系列長を他のパラメータと一緒に伸ばすことが特に重要だと述べています。

これは「推薦モデル一般の普遍的なスケーリング則」が確立したという意味ではありません。観測は1社の非公開データ、非公開タスク、限られた計算量範囲に基づきます。DLRM baselineの正確な本番環境設定も機密で、論文では高水準の構成だけが説明されています。再現可能なpublic datasetの表と、production scalingの主張はevidenceの強さを分けて読む必要があります。[論文Figure 7とAppendix E](https://arxiv.org/pdf/2402.17152#page=8)

## 公開実装で試せる範囲

[公式repository](https://github.com/meta-recsys/generative-recommenders)はApache-2.0で、MovieLensとAmazon Reviewsの公開実験、HSTUのTriton／CUDA kernel、学習・推論用のDLRM-v3などを公開しています。READMEの確認環境はUbuntu 22.04、CUDA 12.4、Python 3.10で、public datasetの多くは24GB以上のGPU memoryが目安です。

```sh
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders
pip3 install -r requirements.txt
mkdir -p tmp
python3 preprocess_public_data.py
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345
```

これはML-1Mなどの公開条件を試す手順であり、1.5兆パラメータの本番モデルを再現するものではありません。本番データ、特徴量定義、A/B test設定、学習cluster、全serving stackは公開されていません。また、リポジトリのREADMEが示す一部の再現値はarXiv v3のTable 4とわずかに異なります。比較するときはpaperの数字と現在のコードを混ぜず、commit、設定、データの前処理を固定する必要があります。

## 導入するなら何を分けて検証するか

この論文を実務へ適用するときは、「HSTUへ置き換える」という1つの変更にまとめず、次の順で効果を切り分けると判断しやすくなります。

1. 評価を固定する：時系列分割、strongなsequential baseline、HR／NDCGに加え、遅延、メモリ、QPSを測る。
2. エンコーダーだけ比較する：同じ特徴量、損失、negative sampling、モデル規模でSASRec／TransformerとHSTUを比べる。
3. 特徴量の系列化を試す：人手集計特徴量を一度に消さず、生の行動だけのGRとの差をablationで測る。
4. 学習最適化を分ける：ragged kernelとStochastic Lengthを別々に導入し、長い履歴での品質劣化を監視する。
5. servingを検証する：候補数ごとにM-FALCONのmicrobatch、KV cache、遅延の裾部分、HBM／DRAM転送を測る。
6. 小さいonline testから始める：オフライン指標だけでなく、主要指標、guardrail、長期的な満足度を確認する。

atomic IDを使うため、新規アイテム、rare item、埋め込みテーブルの更新、削除要求への対応も必要です。系列へ特徴量を統一すれば自動的にプライバシーが改善するわけではありません。どの行動を保存するか、保存期間、アクセス制御、ユーザーの同意を別途設計する必要があります。

## Limitation

この研究を読むうえで重要な制約は次の通りです。

公開実験は3データセットのオフライン評価で、本番環境と異なるmulti-pass・full-shuffle条件です。本番環境のタスク、データ、比較手法詳細、A/B test期間、サンプル数、統計的不確実性が非公開です。

12.4%は匿名化されたranking E-Taskの最大改善で、売上やCTRなど特定の事業指標ではありません。総パラメータ数には巨大なID embeddingも含まれ、同規模LLMとのパラメータ数だけの比較は誤解を招く。

HSTU encoderの高速化、M-FALCON、production infrastructureの効果が処理全体結果では一体になっている。新しいコンテンツや急変する嗜好にatomic ID表現がどう汎化するかは、公開結果だけでは十分に判断できません。


論文のImpact Statementは、手作り特徴量の削減がプライバシーや長期的なユーザーにとっての価値の改善につながる可能性を述べています。しかし、プライバシー指標や長期outcomeを直接評価した結果は示していません。ここは実証済みの効果ではなく、将来の方向性です。

## まとめ

Generative Recommenderの本質は、既存の推薦システムへLLMを足すことではありません。異種特徴量、ランキング、retrieval、学習例の作り方をユーザー行動のsequential transductionとして再設計し、計算量を増やすと品質が伸びる基盤を作ることです。

HSTUのpointwise attentionは行動の「相対的な重要度」だけでなく「蓄積量」を残し、Stochastic Lengthは長い履歴のtraining costを抑え、M-FALCONは候補間で同じ履歴計算を共有します。public datasetでの改善は再現可能な入口ですが、1.5兆パラメータ、12.4%のオンライン改善、285倍のFLOPsを扱う本番環境結果は非公開条件への依存が大きく、システム全体の事例研究として読むのが適切です。

## 参照

- Jiaqi Zhai et al., [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152), ICML 2024, arXiv v3, 2024-05-06.
- Jiaqi Zhai et al., [PDF全文](https://arxiv.org/pdf/2402.17152), 26 pages.
- Meta RecSys, [generative-recommenders](https://github.com/meta-recsys/generative-recommenders), official implementation, Apache-2.0.
