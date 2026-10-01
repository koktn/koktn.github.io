---
title: RPG：64トークンのSemantic IDを並列生成する推薦モデル
description: KDD 2025のRPGを、OPQによる長いSemantic ID、multi-token prediction、graph-constrained decoding、評価結果と再現時の注意点から解説します。
publishedAt: 2026-09-09
updatedAt: 2026-10-01
category: AI
tags:
  - Recommendation System
  - Generative Recommendation
  - Semantic ID
  - Vector Quantization
  - KDD
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文と公開コードを確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Yupeng HouらによるKDD 2025論文「[Generating Long Semantic IDs in Parallel for Recommendation](https://arxiv.org/abs/2506.05781)」です。UC San DiegoとMeta AIの研究者が、generative recommendationのSemantic IDを自己回帰せず、一度に並列予測するRPG（Recommendation with Parallel semantic ID Generation）を提案しています。

この論文の価値は、Semantic IDを順序付きの短いコード列として1トークンずつ生成する前提を外し、最大64トークンを並列予測したうえで、有効なアイテムだけを結ぶグラフ上を探索することで、表現力と推論効率を両立した点にあります。

## 課題：Semantic IDを長くすると自己回帰推論が重くなる

Semantic IDは、アイテムのテキストやimageから得た特徴を複数の離散トークンへ量子化したIDです。アイテム数だけ増える巨大な埋め込みテーブルを直接持つ代わりに、小さなtoken vocabularyを共有できます。また、意味の近いアイテムがトークンを共有するため、コールドスタートにも有利です。

近年のgenerative recommendationは、ユーザーのinteraction 履歴を入力し、次に選ばれるアイテムのSemantic IDをトークン列として生成します。代表的なTIGERは、RQ-VAEで作った4トークンのIDを左から右へ自己回帰生成し、top-K候補を得るためbeam searchを使います。

```text
user history → sequence model
                ↓
              token 1
                ↓ forward
              token 2
                ↓ forward
              token 3
                ↓ forward
              token 4
                ↓
              valid item
```

Semantic IDを長くすればアイテムの細かな意味を保持しやすくなります。しかし、長さを`m`、beam sizeを`b`とすると、自己回帰方式は概ね`O(bm)`回のsequence encoder forwardを必要とします。4トークンなら現実的でも、32や64トークンでは遅延とメモリが大きくなります。

論文の追試では、TIGERのSemantic IDを4から8、16トークンへ増やすと、Sports datasetの1エポック推論時間は259秒、788秒、2,544秒へ増え、NDCG@10は逆に`0.0243`、`0.0219`、`0.0054`へ低下しました。32トークンでは24GBのRTX 3090でout-of-memoryになっています。既存の自己回帰方式では「長いほど表現力が高い」と単純にはいきません。

## RPGの全体像

RPGは、Semantic IDの作り方、学習objective、推論方法をセットで変更します。

```text
[item indexing]
item text → semantic encoder → dense vector
          → OPQ → unorderedな長いSemantic ID
                    (c1, c2, ..., cm)

[training]
user historyの各item tokenをpooling
          → Transformer decoder
          → digit別projection head
          → m tokenをMulti-Token Predictionで同時学習

[inference]
全codebookのtoken logitを1回で計算・cache
          → valid itemをnodeとするgraphで初期beamを拡張
          → 上位beamを残して数step反復
          → top-K item
```

中核は3点です。

| 構成要素 | 役割 | 従来との違い |
| --- | --- | --- |
| Optimized Product Quantization（OPQ） | dense item表現を最大64トークンへ分割・量子化する | RQ-VAEのような前トークンから残差を引く逐次構造を使わない |
| Multi-Token Prediction（MTP） | 次アイテムの全トークンを独立なtargetとして同時に学習する | next-token predictionで左から右へ生成しない |
| Graph-Constrained Decoding | 実在するアイテムだけをnodeにし、類似nodeをたどって高score候補を探す | 全アイテム列挙や、無効なトークン組み合わせの生成を避ける |

## 仕組み1：OPQで「順に読む必要のない」長いIDを作る

RPGはSemantic IDの生成にResidual Quantization（RQ）ではなくOPQを使います。まずsemantic encoderでアイテムをdense vectorへ変換し、回転したベクトルを`m`個のsubvectorへ分割します。各subvectorを別々のコードブックで量子化し、`(c1, c2, ..., cm)`を得ます。

RQは前段で近似し切れなかった残差を次段が量子化するため、コード間にcoarse-to-fineな依存があります。OPQでは各コードが元ベクトルの異なるsubspaceを担当するため、前トークンの生成結果を待たず並列予測しやすくなります。

ここで論文のいう「unordered」は、トークンの所属先まで失ったbag-of-tokensという意味ではありません。`c1`は第1 コードブック、`c2`は第2 コードブックから選ばれ、それぞれ別のtoken embedding tableを持ちます。生成順序への依存が不要という意味です。

履歴側でアイテムをsequence modelへ入力するときは、最大64個のトークンをそのまま連結しません。各token embeddingをmeanまたはmax poolingし、アイテムごとに1ベクトルへ集約します。そのため、Semantic IDを長くしてもユーザー履歴の系列長は`history長 × m`には増えません。

## 仕組み2：Multi-Token Predictionで全桁を同時に学ぶ

Transformer decoderがユーザー履歴から系列表現`s`を作った後、RPGはSemantic IDの桁ごとに別のprojection ヘッド `g_j`を置きます。各ヘッドは対応するコードブック上の確率分布を出し、正解アイテムのトークンに対するnegative log-likelihoodを全桁で合計します。

```text
sequence representation s
  ├─ g1(s) → P(c1 | s)
  ├─ g2(s) → P(c2 | s)
  ├─ ...
  └─ gm(s) → P(cm | s)

loss = -Σ log P(cj | s)
```

これは、ユーザー履歴`s`が与えられた条件下で各トークンが独立だと仮定し、joint probabilityを各桁の確率の積へfactorizeしたものです。sequence modelのforwardは1回で済み、各ヘッドの計算は並列化できます。

推論時は、各コードブックの全`M`トークンについてlog probabilityを一度計算してキャッシュします。候補 item `(c1, ..., cm)`のscoreは、対応するcached log probabilityを`m`個取り出して足すだけです。同じトークンを持つ多数のアイテムでdot productを繰り返す必要がありません。

ただし、このまま各桁のargmaxを組み合わせると問題が起きます。たとえばコードブック sizeが256、ID長が32なら、理論上の組み合わせは`256^32 ≈ 10^77`です。アイテムが10億個あっても、有効なIDは空間全体のごく一部です。独立予測したトークンの組み合わせが実在アイテムを指す確率は極めて低くなります。

## 仕組み3：有効アイテムを結ぶグラフでdecodeする

RPGは、無効なトークン組み合わせを直接生成せず、item poolに存在するSemantic IDだけをnodeにしたdecoding グラフを事前構築します。node間のsimilarityは、対応するtoken embedding同士の内積を全桁で合計して求め、各nodeから上位`k`個の類似nodeへのedgeだけを残します。

リクエストごとのdecodeは次の4段階です。

1. Item poolから`b`個のSemantic IDを無作為に選び、初期beamとする
2. Beam中の各nodeから最大`k` neighborへ展開する
3. Cached token logitの和で候補をscoreし、上位`b` nodeを残す
4. これを`q` ステップ繰り返し、最終beamからtop-Kを返す

```text
random initial beam（b nodes）
  ↓ graph neighborへ展開（最大 b × k）
cached token logitsでscore
  ↓ top-bだけ保持
q回反復
  ↓
top-K recommendations
```

各nodeにはself-edgeもあるため、論文の定義ではbeamの平均scoreは反復で低下しません。ただし、global optimumへ到達する保証ではありません。無作為な初期beamの近傍に良い候補がなければ、局所的な領域に留まる可能性があります。

推論時間のcomplexityは`O(Mmd + bqkm)`です。最初の項が全コードブック tokenのlogit計算、後半がグラフ上で訪問したnodeのscore計算に対応し、アイテム総数`N`を含みません。TIGERの約`O(bm)`回に対して、sequence encoder forwardは`O(1)`回です。

ただし「アイテム数に依存しない」のはリクエスト時にfetchする計算量です。token table、item-to-token mapping、decoding グラフの総保存領域は`O(Mmd + N(m+k))`で、`N`とともに増えます。論文のbest設定で実際に訪問するアイテムは全体の約10.90〜24.79%でした。RPGは全カタログをメモリから消すのではなく、リクエスト時に触る範囲を限定しています。

## 評価設定

実験にはAmazon Reviews 2014の4カテゴリを使い、レビューをinteractionと見なして時系列に並べています。各ユーザーの最後のアイテムをテスト、最後から2番目をvalidationにするleave-last-outです。

| データセット | ユーザー | アイテム | Interactions | 平均履歴長 |
| --- | ---: | ---: | ---: | ---: |
| Sports and Outdoors | 18,357 | 35,598 | 260,739 | 8.32 |
| Beauty | 22,363 | 12,101 | 176,139 | 8.87 |
| Toys and Games | 19,412 | 11,924 | 148,185 | 8.63 |
| CDs and Vinyl | 75,258 | 64,443 | 1,022,334 | 14.58 |

指標はRecall@5／10とNDCG@5／10です。RPGはTIGERとパラメータ数を近づけるため、2-layer Transformer decoder、embedding dimension 448、feed-forward dimension 1,024、4 attention headsを使います。実装したモデルはバッチサイズ256で最大150エポック学習し、validationが20エポック改善しなければearly stoppingします。

全実験は単一のNVIDIA RTX 3090 24GBで実行されています。Sports、Beauty、Toysでは1ハイパーパラメータ設定あたり2 GPU時間未満、3種類のパラメータを45設定探索してデータセットごとに90 GPU時間未満、CDsでは全checkpoint合計約180 GPU時間と報告されています。

## 結果：NDCG@10でstrongest baselineを平均12.6%上回る

RPGは比較されたItem ID系・Semantic ID系比較手法に対し、著者らの集計で12指標中11指標の首位となりました。NDCG@10を各データセットのstrongest baselineと比べると次の通りです。

| データセット | Strongest baseline | Baseline NDCG@10 | RPG NDCG@10 | 相対改善 |
| --- | --- | ---: | ---: | ---: |
| Sports | TIGER | 0.0225 | 0.0263 | +16.9% |
| Beauty | HSTU | 0.0389 | 0.0464 | +19.3% |
| Toys | TIGER | 0.0432 | 0.0490 | +13.4% |
| CDs | TIGER | 0.0411 | 0.0415 | +1.0% |

4データセット平均の相対改善が論文abstractの12.6%です。絶対差はSports `+0.0038`、Beauty `+0.0075`、Toys `+0.0058`、CDs `+0.0004`で、データセットにより大きく異なります。Table 2ではbest baselineに対するpaired t-testで`p < 0.05`と報告されていますが、run数やeffect sizeの信頼区間は本文に記載されていません。

また、これは公開review dataset上のoffline next-item predictionです。CTR、conversion、retentionなどのオンライン効果を測ったproduction experimentではありません。

## 推論効率：TIGER比でメモリ約25分の1、速度約15倍

Sportsのitem poolへdummy itemを追加し、2万から50万アイテムまで増やした実験では、retrieval型のSASRecとVQ-Recはruntime memoryと推論時間がアイテム数に伴って増えました。TIGERとRPGはほぼ一定で、RPGはTIGERに対してruntime memoryを約25分の1へ減らし、推論を約15倍高速化したと報告されています。

このruntime memoryは主にlogit計算とnext-token生成に使うGPU memoryで、model parameter全体やsequence encoderのメモリは含みません。また、全モデルを同じ固定ハイパーパラメータで測っており、各モデルのbest-quality設定同士の比較ではありません。約25倍・15倍を一般的なserving環境の改善率として使うことはできません。

item pool上限も50万です。計算量が`N`に依存しないことは式から説明できますが、billion-scale catalogでグラフ 保存領域、cache locality、distributed servingまで実証した結果ではありません。

## Ablation：OPQ、桁別ヘッド、グラフの3つすべてが必要

Table 3のNDCG@10を見ると、単にTransformerから64トークンを出すだけでは性能が出ないことが分かります。

| Variant | Sports | Beauty | Toys | CDs |
| --- | ---: | ---: | ---: | ---: |
| OPQをrandom tokenへ置換 | 0.0179 | 0.0359 | 0.0288 | 0.0078 |
| OPQをRQへ置換 | 0.0242 | 0.0421 | 0.0458 | 0.0406 |
| Projection ヘッドなし | 0.0252 | 0.0423 | 0.0430 | 0.0361 |
| 全桁でヘッド共有 | 0.0256 | 0.0424 | 0.0438 | 0.0368 |
| Graph constraintなし | 0.0082 | 0.0214 | 0.0205 | 0.0183 |
| RPG | 0.0263 | 0.0464 | 0.0490 | 0.0415 |

Random tokenで大きく悪化するため、MTPが任意のコードを暗記しているだけではなく、Semantic IDに含まれる意味を利用していると著者らは解釈しています。RQへの置換でも一貫して低下し、並列予測にはsubspaceを独立に量子化するOPQが合っています。

Digitごとに別projection ヘッドを持つ方が、ヘッドなしや共有ヘッドより良い結果です。各コードブックが異なるsemantic subspaceを担当するため、同じユーザー表現を別空間へ写す必要がある、という設計と整合します。

最も大きいのはグラフ constraintの除去です。同程度のアイテム数を無作為に訪問してもRPGへ届きません。通常のbeam searchは全データセットで`0.0000`でした。これはOPQ tokenに左から右の依存がないのに、prefixを順番に伸ばすsearchを適用したためです。つまり、並列生成によって失ったトークン間の整合性を、valid item グラフが推論時に補っています。

## 長さは64が常に最適ではない

Semantic ID長`m`を4、8、16、32、64で変えた結果、概ね長いほどNDCG@10は改善しますが、小さいデータセットでは早く頭打ちになります。最適な長さはSportsが16、Beautyが32、Toysが16、最大のCDsが64でした。

同じ4-トークン条件ではRPGはTIGERより悪く、たとえばSportsで`0.0152`対`0.0225`です。RPGの利点は短いIDで自己回帰モデルに勝つことではなく、自己回帰では扱いづらい長いIDまで規模できる点にあります。

Semantic encoderを`sentence-t5-base`から`text-embedding-3-large`へ変えた実験では、4-token TIGERは全データセットで一貫して改善しませんでした。一方、長いIDのRPGは4データセットすべてで改善しています。強いエンコーダーが出す豊富な情報を、短い4-token IDでは量子化し切れない可能性を示す結果です。

Cold-start分析ではSportsのtest itemを学習出現回数`[0,5]`、`[6,10]`、`[11,15]`、`[16,20]`へ分け、RPGが全体として最も高いNDCG@10を示しました。ただしFigureには棒の正確な値が掲載されていないため、ここでは順位と傾向だけを扱います。

## Graph探索の品質とコスト

Sportsでのハイパーパラメータ分析では、beam size `b`はtop-Kより大きい10でほぼ飽和しました。Edge数`k`は100程度まで増やすと改善し、その先は限界効果が小さくなります。Iteration `q`は0から2で大きく改善し、2以降はほぼ飽和しました。

これは本番環境設計にも直結します。

- `b`を増やすと保持候補とscore計算が増える
- `k`を増やすと探索範囲は広がるが、グラフ 保存領域と1 ステップの計算が増える
- `q`を増やすと遠いnodeへ届くが、遅延が線形に増える
- 初期beamが無作為なので、品質の分散と再現性をseed別に測る必要がある

論文のbest設定はデータセットごとに異なり、訪問アイテム率も約11〜25%です。`b=10, k=100, q=2`のような値を別カタログへそのまま移すのではなく、Recall／NDCGとp95／p99 latencyを同時に測って決める必要があります。

## 公開コードで試すときの注意

著者らは[facebookresearch/RPG_KDD2025](https://github.com/facebookresearch/RPG_KDD2025)でコードを公開しています。2026年9月9日に確認したmain revisionは`7dcf95c`で、Python versionは指定されていません。`requirements.txt`にはPyTorchのCUDA 12.9インデックス、`transformers==4.56.1`、`datasets==4.0.0`、`faiss-cpu==1.12.0`などが記載されています。

READMEによれば、カテゴリを指定するとデータセットを自動downloadし、次の形で学習を開始できます。

```sh
git clone https://github.com/facebookresearch/RPG_KDD2025.git
cd RPG_KDD2025
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python main.py --category=Sports_and_Outdoors
```

このcommandは公開リポジトリの手順を整理したもので、本記事ではGPU環境とdata downloadを伴うため実行していません。再現時はリポジトリをrevisionでpinし、isolated environmentを使ってください。

### 論文とREADMEの設定が一致しない

注意すべきなのは、現行READMEの「Reproduction」commandと論文Appendix Table 6のbest hyperparameterに差があることです。たとえばSportsでは次のように異なります。

| パラメータ | 論文Table 6 | README reproduction command |
| --- | ---: | ---: |
| Semantic ID length | 16 | 16 |
| Beam size | 10 | 100 |
| Edges per node | 100 | 30 |
| Propagation steps | 2 | 5 |

Beauty、Toys、CDsにもbeam size、edge数、ステップ数の差があります。公開コードは論文投稿後に更新されているため、README側が後続調整を反映した可能性はありますが、変更理由やどちらがTable 2を再現する設定かは明記されていません。再現結果を報告するときは「paper config」「README config」を分け、両方を試すのが安全です。

また、code licenseはCC BY-NC 4.0で、商用利用は許諾範囲外です。本番環境へ組み込む前にlicenseを確認し、必要なら著作権者へ別途相談してください。

## 自社システムへ適用するなら

以下は論文の再現手順ではなく、公開情報から導いた導入案です。

### 1. Exact retrievalを上限として測る

まず全アイテムをscoreするexact retrievalでNDCG／Recallを測り、グラフ decodeによる近似誤差を分離します。RPGのmodel qualityとグラフ search qualityを同時に変えると、改善・悪化の原因が分かりません。

### 2. Semantic IDの情報量と重複を確認する

`m`を増やしながらquantization error、トークン利用率、コードブックごとのentropy、同一token setを持つアイテム数を記録します。論文でも異なるアイテムが同じtoken setを持つことは可能で、グラフ nodeはunique token setではなくアイテム単位です。

### 3. Graphをoffline artifactとしてバージョン管理する

Graphはモデル再学習またはアイテム再tokenizeまで再利用できますが、カタログへアイテムが追加・削除されれば古くなります。少なくとも次のバージョンを一緒に保存します。

```text
semantic_encoder_version
opq_codebook_version
token_embedding_checkpoint
catalog_snapshot
graph_build_version
graph_built_at
```

新グラフをshadow loadし、node coverage、degree、connected component、検索品質、メモリを検証してから切り替えます。失敗時に旧グラフと旧tokenizerへ戻せるよう、モデルだけでなくインデックスも切り戻し単位にします。

### 4. Random initializationをmonitorする

同一リクエストを複数seedでdecodeし、top-K overlap、NDCGの分散、探索が到達する構成要素を確認します。人気アイテムから始める、ユーザー表現に近い粗い候補をseedにする、複数startをまとめるなどは改善案になり得ますが、いずれも原論文で検証された手法ではありません。

### 5. コストを処理全体で比較する

Graph decodeだけでなく、semantic encoding、OPQ再構築、グラフ 開発、保存領域、network fetchを含めます。Offline NDCG、visited item率、GPU memory、p50／p95／p99 latency、スループット、index freshnessをguardrailにし、exact retrievalや既存ANN、autoregressive modelと比較します。

外部embedding APIへitem 内容を送る場合は、費用、呼び出し頻度の制限、data retention、機密情報・個人情報の扱いも確認が必要です。論文が`text-embedding-3-large`で示した改善は研究データセット上の結果であり、特定サービスの採用を一般に推奨するものではありません。

## Limitation

RPGの結果を読むうえでは、次の制約があります。

評価はAmazon Reviews 2014の4カテゴリに限られ、最大でも64,443アイテム、1,022,334 interactionです。Scalability実験はdummy itemを含む最大50万アイテムで、本番トラフィックやbillion-scale catalogではありません。

レビューをpositive interactionとして扱うleave-last-out評価であり、非クリックや曝光、時刻、価格、在庫などの実運用シグナルを扱わない。NDCG@10平均12.6%はstrongest baselineに対する4データセットの相対改善で、絶対差とデータセット別の効果は大きく異なる。

Runtime memory約25分の1、速度約15倍はSports、固定ハイパーパラメータ、RTX 3090での1エポック推論で、model parameter memoryを含まない。MTPはユーザー履歴を条件に桁間の独立性を仮定し、失われた整合性を推論時のグラフへ移している。

Graphの総保存領域と更新コストはアイテム数に依存し、アイテム追加が頻繁なカタログでのincremental更新は示されていない。Random initial beamによるrun間のばらつき、グラフの分断、局所解への感度が十分に報告されていない。

公開コードのREADMEと論文で一部ハイパーパラメータが一致せず、完全な再現には追加確認が必要です。Fairness、プライバシー、悪意あるコンテンツによるSemantic ID操作、online user impactは評価対象外です。


## まとめ

RPGが示したのは、Semantic IDの長さを伸ばすにはモデルを高速化するだけでなく、IDの構造、training objective、search algorithmを同時に設計し直す必要があるということです。

OPQで各semantic subspaceを独立トークンへ変え、最大64トークンのIDを作ります。MTPで全桁を1回のsequence forwardから並列予測します。

Cached token logitの和でアイテムを効率よくscoreします。Valid item グラフを探索し、独立予測では生じる無効な組み合わせを避ける。

4つのpublic datasetでstrongest baselineをNDCG@10平均12.6%上回る。Sportsの効率評価でTIGER比メモリ約25分の1、推論約15倍を報告します。


一方で、効率化の代償として、トークン間の依存はグラフという外部インデックスへ移ります。本番環境で重要なのは「sequence modelのforwardが1回」という一点ではなく、グラフの開発、更新、保存領域、探索品質まで含めたシステム全体です。RPGは、generative recommendationを自己回帰だけで捉えず、並列分類とグラフ searchの組み合わせとして再構成した研究だと言えます。

## 参照

- Yupeng Hou et al., [Generating Long Semantic IDs in Parallel for Recommendation](https://arxiv.org/abs/2506.05781), KDD 2025, arXiv:2506.05781v1, 2025-06-06（[PDF](https://arxiv.org/pdf/2506.05781)、[ACM DOI](https://doi.org/10.1145/3711896.3736979)）。
- Meta Research, [facebookresearch/RPG_KDD2025](https://github.com/facebookresearch/RPG_KDD2025), revision `7dcf95c`（2025-09-08 UTC）。

論文の手法、数値、実験条件は原論文に基づき、公開手順とdependencyはrepositoryを確認しました。「自社systemへ適用するなら」は公開情報をもとにした記事側の提案です。
