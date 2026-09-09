---
title: RPG――64 tokenのSemantic IDを並列生成する推薦モデル
description: KDD 2025のRPGを、OPQによる長いSemantic ID、multi-token prediction、graph-constrained decoding、評価結果と再現時の注意点から解説します。
publishedAt: 2026-09-09
category: AI
tags:
  - Recommendation System
  - Generative Recommendation
  - Semantic ID
  - Vector Quantization
  - KDD
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値や主張は原論文と公開codeを確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Yupeng HouらによるKDD 2025論文「[Generating Long Semantic IDs in Parallel for Recommendation](https://arxiv.org/abs/2506.05781)」です。UC San DiegoとMeta AIの研究者が、generative recommendationのSemantic IDを自己回帰せず、一度に並列予測する**RPG（Recommendation with Parallel semantic ID Generation）**を提案しています。

この論文の価値を一文でまとめると、**Semantic IDを順序付きの短いcode列として1 tokenずつ生成する前提を外し、最大64 tokenを並列予測したうえで、有効なitemだけを結ぶgraph上を探索することで、表現力と推論効率を両立した**点にあります。

## 課題：Semantic IDを長くすると自己回帰推論が重くなる

Semantic IDは、itemのtextやimageから得た特徴を複数の離散tokenへ量子化したIDです。item数だけ増える巨大なembedding tableを直接持つ代わりに、小さなtoken vocabularyを共有できます。また、意味の近いitemがtokenを共有するため、cold startにも有利です。

近年のgenerative recommendationは、userのinteraction historyを入力し、次に選ばれるitemのSemantic IDをtoken列として生成します。代表的なTIGERは、RQ-VAEで作った4 tokenのIDを左から右へ自己回帰生成し、top-K候補を得るためbeam searchを使います。

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

Semantic IDを長くすればitemの細かな意味を保持しやすくなります。しかし、長さを `m`、beam sizeを `b` とすると、自己回帰方式は概ね `O(bm)` 回のsequence encoder forwardを必要とします。4 tokenなら現実的でも、32や64 tokenではlatencyとmemoryが大きくなります。

論文の追試では、TIGERのSemantic IDを4から8、16 tokenへ増やすと、Sports datasetの1 epoch推論時間は259秒、788秒、2,544秒へ増え、NDCG@10は逆に `0.0243`、`0.0219`、`0.0054`へ低下しました。32 tokenでは24GBのRTX 3090でout-of-memoryになっています。既存の自己回帰方式では「長いほど表現力が高い」と単純にはいきません。

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

| Component | 役割 | 従来との違い |
| --- | --- | --- |
| Optimized Product Quantization（OPQ） | dense item表現を最大64 tokenへ分割・量子化する | RQ-VAEのような前tokenから残差を引く逐次構造を使わない |
| Multi-Token Prediction（MTP） | 次itemの全tokenを独立なtargetとして同時に学習する | next-token predictionで左から右へ生成しない |
| Graph-Constrained Decoding | 実在するitemだけをnodeにし、類似nodeをたどって高score候補を探す | 全item列挙や、無効なtoken組み合わせの生成を避ける |

## 仕組み1：OPQで「順に読む必要のない」長いIDを作る

RPGはSemantic IDの生成にResidual Quantization（RQ）ではなくOPQを使います。まずsemantic encoderでitemをdense vectorへ変換し、回転したvectorを `m` 個のsubvectorへ分割します。各subvectorを別々のcodebookで量子化し、`(c1, c2, ..., cm)`を得ます。

RQは前段で近似し切れなかった残差を次段が量子化するため、code間にcoarse-to-fineな依存があります。OPQでは各codeが元vectorの異なるsubspaceを担当するため、前tokenの生成結果を待たず並列予測しやすくなります。

ここで論文のいう「unordered」は、tokenの所属先まで失ったbag-of-tokensという意味ではありません。`c1`は第1 codebook、`c2`は第2 codebookから選ばれ、それぞれ別のtoken embedding tableを持ちます。**生成順序への依存が不要**という意味です。

history側でitemをsequence modelへ入力するときは、最大64個のtokenをそのまま連結しません。各token embeddingをmeanまたはmax poolingし、itemごとに1 vectorへ集約します。そのため、Semantic IDを長くしてもuser historyのsequence長は `history長 × m` には増えません。

## 仕組み2：Multi-Token Predictionで全digitを同時に学ぶ

Transformer decoderがuser historyからsequence表現 `s` を作った後、RPGはSemantic IDのdigitごとに別のprojection head `g_j`を置きます。各headは対応するcodebook上の確率分布を出し、正解itemのtokenに対するnegative log-likelihoodを全digitで合計します。

```text
sequence representation s
  ├─ g1(s) → P(c1 | s)
  ├─ g2(s) → P(c2 | s)
  ├─ ...
  └─ gm(s) → P(cm | s)

loss = -Σ log P(cj | s)
```

これは、user history `s` が与えられた条件下で各tokenが独立だと仮定し、joint probabilityを各digitの確率の積へfactorizeしたものです。sequence modelのforwardは1回で済み、各headの計算は並列化できます。

推論時は、各codebookの全 `M` tokenについてlog probabilityを一度計算してcacheします。candidate item `(c1, ..., cm)`のscoreは、対応するcached log probabilityを `m` 個取り出して足すだけです。同じtokenを持つ多数のitemでdot productを繰り返す必要がありません。

ただし、このまま各digitのargmaxを組み合わせると問題が起きます。たとえばcodebook sizeが256、ID長が32なら、理論上の組み合わせは `256^32 ≈ 10^77`です。itemが10億個あっても、有効なIDは空間全体のごく一部です。独立予測したtokenの組み合わせが実在itemを指す確率は極めて低くなります。

## 仕組み3：有効itemを結ぶgraphでdecodeする

RPGは、無効なtoken組み合わせを直接生成せず、item poolに存在するSemantic IDだけをnodeにしたdecoding graphを事前構築します。node間のsimilarityは、対応するtoken embedding同士の内積を全digitで合計して求め、各nodeから上位 `k` 個の類似nodeへのedgeだけを残します。

requestごとのdecodeは次の4段階です。

1. Item poolから `b` 個のSemantic IDをrandomに選び、初期beamとする
2. Beam中の各nodeから最大 `k` neighborへ展開する
3. Cached token logitの和で候補をscoreし、上位 `b` nodeを残す
4. これを `q` step繰り返し、最終beamからtop-Kを返す

```text
random initial beam（b nodes）
  ↓ graph neighborへ展開（最大 b × k）
cached token logitsでscore
  ↓ top-bだけ保持
q回反復
  ↓
top-K recommendations
```

各nodeにはself-edgeもあるため、論文の定義ではbeamの平均scoreは反復で低下しません。ただし、global optimumへ到達する保証ではありません。randomな初期beamの近傍に良いcandidateがなければ、局所的な領域に留まる可能性があります。

推論時間のcomplexityは `O(Mmd + bqkm)`です。最初の項が全codebook tokenのlogit計算、後半がgraph上で訪問したnodeのscore計算に対応し、item総数 `N` を含みません。TIGERの約 `O(bm)` 回に対して、sequence encoder forwardは `O(1)` 回です。

ただし「item数に依存しない」のはrequest時にfetchする計算量です。token table、item-to-token mapping、decoding graphの総storageは `O(Mmd + N(m+k))`で、`N`とともに増えます。論文のbest設定で実際に訪問するitemは全体の約10.90〜24.79%でした。RPGは全catalogをmemoryから消すのではなく、request時に触る範囲を限定しています。

## 評価設定

実験にはAmazon Reviews 2014の4 categoryを使い、reviewをinteractionと見なして時系列に並べています。各userの最後のitemをtest、最後から2番目をvalidationにするleave-last-outです。

| Dataset | Users | Items | Interactions | 平均history長 |
| --- | ---: | ---: | ---: | ---: |
| Sports and Outdoors | 18,357 | 35,598 | 260,739 | 8.32 |
| Beauty | 22,363 | 12,101 | 176,139 | 8.87 |
| Toys and Games | 19,412 | 11,924 | 148,185 | 8.63 |
| CDs and Vinyl | 75,258 | 64,443 | 1,022,334 | 14.58 |

metricはRecall@5／10とNDCG@5／10です。RPGはTIGERとparameter数を近づけるため、2-layer Transformer decoder、embedding dimension 448、feed-forward dimension 1,024、4 attention headsを使います。実装したmodelはbatch size 256で最大150 epoch学習し、validationが20 epoch改善しなければearly stoppingします。

全実験は単一のNVIDIA RTX 3090 24GBで実行されています。Sports、Beauty、Toysでは1 hyperparameter設定あたり2 GPU時間未満、3種類のparameterを45設定探索してdatasetごとに90 GPU時間未満、CDsでは全checkpoint合計約180 GPU時間と報告されています。

## 結果：NDCG@10でstrongest baselineを平均12.6%上回る

RPGは比較されたItem ID系・Semantic ID系baselineに対し、著者らの集計で12 metric中11 metricの首位となりました。NDCG@10を各datasetのstrongest baselineと比べると次の通りです。

| Dataset | Strongest baseline | Baseline NDCG@10 | RPG NDCG@10 | 相対改善 |
| --- | --- | ---: | ---: | ---: |
| Sports | TIGER | 0.0225 | 0.0263 | +16.9% |
| Beauty | HSTU | 0.0389 | 0.0464 | +19.3% |
| Toys | TIGER | 0.0432 | 0.0490 | +13.4% |
| CDs | TIGER | 0.0411 | 0.0415 | +1.0% |

4 dataset平均の相対改善が論文abstractの**12.6%**です。絶対差はSports `+0.0038`、Beauty `+0.0075`、Toys `+0.0058`、CDs `+0.0004`で、datasetにより大きく異なります。Table 2ではbest baselineに対するpaired t-testで `p < 0.05`と報告されていますが、run数やeffect sizeのconfidence intervalは本文に記載されていません。

また、これは公開review dataset上のoffline next-item predictionです。CTR、conversion、retentionなどのonline効果を測ったproduction experimentではありません。

## 推論効率：TIGER比でmemory約25分の1、速度約15倍

Sportsのitem poolへdummy itemを追加し、2万から50万itemまで増やした実験では、retrieval型のSASRecとVQ-Recはruntime memoryと推論時間がitem数に伴って増えました。TIGERとRPGはほぼ一定で、RPGはTIGERに対してruntime memoryを**約25分の1**へ減らし、推論を**約15倍**高速化したと報告されています。

このruntime memoryは主にlogit計算とnext-token生成に使うGPU memoryで、model parameter全体やsequence encoderのmemoryは含みません。また、全modelを同じ固定hyperparameterで測っており、各modelのbest-quality設定同士の比較ではありません。約25倍・15倍を一般的なserving環境の改善率として使うことはできません。

item pool上限も50万です。計算量が `N` に依存しないことは式から説明できますが、billion-scale catalogでgraph storage、cache locality、distributed servingまで実証した結果ではありません。

## Ablation：OPQ、digit別head、graphの3つすべてが必要

Table 3のNDCG@10を見ると、単にTransformerから64 tokenを出すだけでは性能が出ないことが分かります。

| Variant | Sports | Beauty | Toys | CDs |
| --- | ---: | ---: | ---: | ---: |
| OPQをrandom tokenへ置換 | 0.0179 | 0.0359 | 0.0288 | 0.0078 |
| OPQをRQへ置換 | 0.0242 | 0.0421 | 0.0458 | 0.0406 |
| Projection headなし | 0.0252 | 0.0423 | 0.0430 | 0.0361 |
| 全digitでhead共有 | 0.0256 | 0.0424 | 0.0438 | 0.0368 |
| Graph constraintなし | 0.0082 | 0.0214 | 0.0205 | 0.0183 |
| RPG | 0.0263 | 0.0464 | 0.0490 | 0.0415 |

Random tokenで大きく悪化するため、MTPが任意のcodeを暗記しているだけではなく、Semantic IDに含まれる意味を利用していると著者らは解釈しています。RQへの置換でも一貫して低下し、並列予測にはsubspaceを独立に量子化するOPQが合っています。

Digitごとに別projection headを持つ方が、headなしや共有headより良い結果です。各codebookが異なるsemantic subspaceを担当するため、同じuser表現を別空間へ写す必要がある、という設計と整合します。

最も大きいのはgraph constraintの除去です。同程度のitem数をrandomに訪問してもRPGへ届きません。通常のbeam searchは全datasetで `0.0000`でした。これはOPQ tokenに左から右の依存がないのに、prefixを順番に伸ばすsearchを適用したためです。つまり、並列生成によって失ったtoken間の整合性を、valid item graphが推論時に補っています。

## 長さは64が常に最適ではない

Semantic ID長 `m` を4、8、16、32、64で変えた結果、概ね長いほどNDCG@10は改善しますが、小さいdatasetでは早く頭打ちになります。best lengthはSportsが16、Beautyが32、Toysが16、最大のCDsが64でした。

同じ4-token条件ではRPGはTIGERより悪く、たとえばSportsで `0.0152`対`0.0225`です。RPGの利点は短いIDで自己回帰modelに勝つことではなく、自己回帰では扱いづらい長いIDまでscaleできる点にあります。

Semantic encoderを`sentence-t5-base`から`text-embedding-3-large`へ変えた実験では、4-token TIGERは全datasetで一貫して改善しませんでした。一方、長いIDのRPGは4 datasetすべてで改善しています。強いencoderが出す豊富な情報を、短い4-token IDでは量子化し切れない可能性を示す結果です。

Cold-start分析ではSportsのtest itemをtraining出現回数 `[0,5]`、`[6,10]`、`[11,15]`、`[16,20]`へ分け、RPGが全体として最も高いNDCG@10を示しました。ただしFigureには棒の正確な値が掲載されていないため、ここでは順位と傾向だけを扱います。

## Graph探索のqualityとcost

Sportsでのhyperparameter分析では、beam size `b` はtop-Kより大きい10でほぼ飽和しました。Edge数 `k` は100程度まで増やすと改善し、その先は限界効果が小さくなります。Iteration `q` は0から2で大きく改善し、2以降はほぼ飽和しました。

これはproduction設計にも直結します。

- `b`を増やすと保持candidateとscore計算が増える
- `k`を増やすと探索範囲は広がるが、graph storageと1 stepの計算が増える
- `q`を増やすと遠いnodeへ届くが、latencyが線形に増える
- 初期beamがrandomなので、qualityの分散と再現性をseed別に測る必要がある

論文のbest設定はdatasetごとに異なり、訪問item率も約11〜25%です。`b=10, k=100, q=2`のような値を別catalogへそのまま移すのではなく、Recall／NDCGとp95／p99 latencyを同時に測って決める必要があります。

## 公開codeで試すときの注意

著者らは[facebookresearch/RPG_KDD2025](https://github.com/facebookresearch/RPG_KDD2025)でcodeを公開しています。2026年9月9日に確認したmain revisionは`7dcf95c`で、Python versionは指定されていません。`requirements.txt`にはPyTorchのCUDA 12.9 index、`transformers==4.56.1`、`datasets==4.0.0`、`faiss-cpu==1.12.0`などが記載されています。

READMEによれば、categoryを指定するとdatasetを自動downloadし、次の形で学習を開始できます。

```sh
git clone https://github.com/facebookresearch/RPG_KDD2025.git
cd RPG_KDD2025
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python main.py --category=Sports_and_Outdoors
```

このcommandは公開repositoryの手順を整理したもので、本記事ではGPU環境とdata downloadを伴うため実行していません。再現時はrepositoryをrevisionでpinし、isolated environmentを使ってください。

### 論文とREADMEの設定が一致しない

注意すべきなのは、現行READMEの「Reproduction」commandと論文Appendix Table 6のbest hyperparameterに差があることです。たとえばSportsでは次のように異なります。

| Parameter | 論文Table 6 | README reproduction command |
| --- | ---: | ---: |
| Semantic ID length | 16 | 16 |
| Beam size | 10 | 100 |
| Edges per node | 100 | 30 |
| Propagation steps | 2 | 5 |

Beauty、Toys、CDsにもbeam size、edge数、step数の差があります。公開codeは論文投稿後に更新されているため、README側が後続調整を反映した可能性はありますが、変更理由やどちらがTable 2を再現する設定かは明記されていません。再現結果を報告するときは「paper config」「README config」を分け、両方を試すのが安全です。

また、code licenseは**CC BY-NC 4.0**で、商用利用は許諾範囲外です。productionへ組み込む前にlicenseを確認し、必要なら著作権者へ別途相談してください。

## 自社systemへ適用するなら

以下は論文の再現手順ではなく、公開情報から導いた導入案です。

### 1. Exact retrievalを上限として測る

まず全itemをscoreするexact retrievalでNDCG／Recallを測り、graph decodeによる近似誤差を分離します。RPGのmodel qualityとgraph search qualityを同時に変えると、改善・悪化の原因が分かりません。

### 2. Semantic IDの情報量と重複を確認する

`m`を増やしながらquantization error、token利用率、codebookごとのentropy、同一token setを持つitem数を記録します。論文でも異なるitemが同じtoken setを持つことは可能で、graph nodeはunique token setではなくitem単位です。

### 3. Graphをoffline artifactとしてversion管理する

Graphはmodel再学習またはitem再tokenizeまで再利用できますが、catalogへitemが追加・削除されれば古くなります。少なくとも次のversionを一緒に保存します。

```text
semantic_encoder_version
opq_codebook_version
token_embedding_checkpoint
catalog_snapshot
graph_build_version
graph_built_at
```

新graphをshadow loadし、node coverage、degree、connected component、検索quality、memoryを検証してから切り替えます。失敗時に旧graphと旧tokenizerへ戻せるよう、modelだけでなくindexもrollback単位にします。

### 4. Random initializationをmonitorする

同一requestを複数seedでdecodeし、top-K overlap、NDCGの分散、探索が到達するcomponentを確認します。人気itemから始める、user表現に近い粗いcandidateをseedにする、複数startをまとめるなどは改善案になり得ますが、いずれも原論文で検証された手法ではありません。

### 5. Costをend-to-endで比較する

Graph decodeだけでなく、semantic encoding、OPQ再構築、graph build、storage、network fetchを含めます。Offline NDCG、visited item率、GPU memory、p50／p95／p99 latency、throughput、index freshnessをguardrailにし、exact retrievalや既存ANN、autoregressive modelと比較します。

外部embedding APIへitem contentを送る場合は、費用、rate limit、data retention、機密情報・個人情報の扱いも確認が必要です。論文が`text-embedding-3-large`で示した改善は研究dataset上の結果であり、特定serviceの採用を一般に推奨するものではありません。

## Limitation

RPGの結果を読むうえでは、次の制約があります。

- 評価はAmazon Reviews 2014の4 categoryに限られ、最大でも64,443 item、1,022,334 interactionである
- Scalability実験はdummy itemを含む最大50万itemで、production trafficやbillion-scale catalogではない
- Reviewをpositive interactionとして扱うleave-last-out評価であり、非クリックや曝光、時刻、価格、在庫などの実運用signalを扱わない
- NDCG@10平均12.6%はstrongest baselineに対する4 datasetの相対改善で、絶対差とdataset別の効果は大きく異なる
- Runtime memory約25分の1、速度約15倍はSports、固定hyperparameter、RTX 3090での1 epoch推論で、model parameter memoryを含まない
- MTPはuser historyを条件にdigit間の独立性を仮定し、失われた整合性を推論時のgraphへ移している
- Graphの総storageと更新costはitem数に依存し、item追加が頻繁なcatalogでのincremental更新は示されていない
- Random initial beamによるrun間のばらつき、graphの分断、局所解への感度が十分に報告されていない
- 公開codeのREADMEと論文で一部hyperparameterが一致せず、完全な再現には追加確認が必要である
- Fairness、privacy、悪意あるcontentによるSemantic ID操作、online user impactは評価対象外である

## まとめ

RPGが示したのは、Semantic IDの長さを伸ばすにはmodelを高速化するだけでなく、**IDの構造、training objective、search algorithmを同時に設計し直す必要がある**ということです。

- OPQで各semantic subspaceを独立tokenへ変え、最大64 tokenのIDを作る
- MTPで全digitを1回のsequence forwardから並列予測する
- Cached token logitの和でitemを効率よくscoreする
- Valid item graphを探索し、独立予測では生じる無効な組み合わせを避ける
- 4つのpublic datasetでstrongest baselineをNDCG@10平均12.6%上回る
- Sportsの効率評価でTIGER比memory約25分の1、推論約15倍を報告する

一方で、効率化の代償として、token間の依存はgraphという外部indexへ移ります。Productionで重要なのは「sequence modelのforwardが1回」という一点ではなく、graphのbuild、更新、storage、探索品質まで含めたsystem全体です。RPGは、generative recommendationを自己回帰だけで捉えず、並列分類とgraph searchの組み合わせとして再構成した研究だと言えます。

## 参照

- Yupeng Hou et al., [Generating Long Semantic IDs in Parallel for Recommendation](https://arxiv.org/abs/2506.05781), KDD 2025, arXiv:2506.05781v1, 2025-06-06（[PDF](https://arxiv.org/pdf/2506.05781)、[ACM DOI](https://doi.org/10.1145/3711896.3736979)）。
- Meta Research, [facebookresearch/RPG_KDD2025](https://github.com/facebookresearch/RPG_KDD2025), revision `7dcf95c`（2025-09-08 UTC）。

論文の手法、数値、実験条件は原論文に基づき、公開手順とdependencyはrepositoryを確認しました。「自社systemへ適用するなら」は公開情報をもとにした記事側の提案です。
