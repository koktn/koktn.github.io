---
title: DV365――Instagramは7万件の行動履歴をどう推薦に使ったか
description: MetaのDV365を、Multi-slicing、Funnel Summarization、オフライン埋め込み、15モデルへの展開、精度・鮮度・コストの検証から解説します。
publishedAt: 2026-09-29
category: AI
tags:
  - Recommendation System
  - User Modeling
  - Representation Learning
  - Meta
  - Instagram
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Wenhan LyuらがKDD 2025で発表した「[DV365: Extremely Long User History Modeling at Instagram](https://arxiv.org/abs/2506.00450)」です。arXivには2025年5月31日にv1が投稿され、InstagramとThreadsのproduction systemで1年以上運用した結果が報告されています。

この論文の価値を一文でまとめると、**最大7万件の行動履歴をrequestごとに直接処理するのではなく、長期的で安定した興味を表す埋め込みへオフラインで圧縮し、1つの上流modelから15個の推薦modelへ共有できることをproduction規模で示した**点にあります。

## 課題：長い履歴は有用だが、onlineで読むには高すぎる

推薦modelにとって、userが過去に何を見て、likeし、shareしたかは重要なsignalです。しかし履歴を長くするほど、次の3種類のcostが増えます。

- 低latencyなstorageへ長い履歴を保持するcost
- requestごとに履歴を取得・加工するCPUとnetworkのcost
- attentionなどでsequenceをencodeする学習・推論GPUのcost

論文執筆時のInstagramでは、production modelが参照できる行動履歴は最大約2,000件でした。その中でも、HSTUを使う高度なsequence encoderが直接処理するのは最大500件です。既存modelは何年も最適化されており、sequence長をそのまま伸ばすだけではROIが合わない状態でした。

一般的なlong-sequence手法には、長い履歴からtarget itemに関連する候補を検索し、絞り込んだ後にattentionをかける構成があります。このend-to-end方式は新しい興味を細かく捉えやすい一方、各ranking modelがtrainingとservingの両方で長い履歴を扱わなければなりません。

InstagramにはReels、Feed、Explore、Storiesなど複数のsurfaceがあり、retrieval、early-stage ranking、late-stage rankingにも別modelがあります。論文では20以上のmodelが存在するとしています。各modelが同じ長期履歴を個別に処理すると、ほぼ同じ高価な計算を何度も繰り返すことになります。

DV365はここで、userの興味を2つの時間軸へ分けて考えます。

| 興味 | 性質 | 適した処理 |
| --- | --- | --- |
| emerging interest | 直近の出来事で素早く変わる | 各downstream modelの新鮮なsequence encoder |
| stable interest | 数時間・数日では大きく変わりにくい | 長い履歴から定期的に事前計算する共有埋め込み |

つまりDV365は既存のonline sequence modelを置き換えるものではありません。直近の履歴を読むHSTUなどは残し、そのmodelだけでは届かない長期signalを追加featureとして渡します。

## 全体像：7万件を200個に集約し、58個へ圧縮する

DV365のdata flowを単純化すると次のようになります。

```text
最大70,000件のuser行動履歴
  ↓ explicit／implicit timelineへ整理
action・視聴時間・期間・ID種別などでmulti-slicing
  ↓ 各sliceをmean／weighted mean pooling
200個 × 256次元の埋め込み
  ↓ Funnel Summarization Arch（FSA）
58個 × 256次元のuser埋め込み
  ↓ 4-bit quantization
user IDをkeyにonline key-value storeへ保存
  ↓
ranking／retrieval modelが軽量なadaptation moduleで利用
```

上流のfoundation modelは定期的に学習され、user towerの中間出力を全active userについて生成します。論文では30億user分の `user ID → embedding` をkey-value storeへ保存し、埋め込みを6時間ごとに生成するとしています。

downstream modelはrequest時に圧縮済みのDV365 embeddingを取得するだけです。長いraw sequenceの取得、200種類のslicing、上流encoderの推論をrequestのcritical pathから外し、複数modelで計算結果を共有します。

## UniTi：異なる行動を1本のtimelineとして持つ

入力となるUnified User Timeline（UniTi）は、userの行動を構造化した共通形式です。各interactionには、media ID、author ID、event timestamp、action type、video duration、watch time、surface type、media typeなどを持たせます。

timelineは大きく2種類です。

- **explicit timeline**：like、share、commentなど、明示的な行動
- **implicit timeline**：表示されたcontentとdwell timeなど、視聴から得る暗黙的な行動

DV365が扱う長さは、implicit timelineが平均3万件・最大3万5,000件、explicit timelineが平均1万件・最大3万5,000件です。合計で平均4万件、最大7万件となります。最大値はsystem上の絶対限界ではなく、著者らがROIを考えて選んだcapです。

raw timelineのまま保持する利点は、feature engineeringをdata生成時に固定しないことです。model側のpreprocessingで条件を変え、同じtimelineから異なるsliceを作れます。

## Multi-slicing：順序を精密に読む代わりに、複数の見方で数える

7万件すべてへself-attentionを適用するのではなく、DV365は履歴を意味のある条件で分割し、各subsetをpoolingします。

explicit timelineでは、like、comment、shareといったaction type、videoやphotoなどのsourceを分けます。implicit timelineでは、dwell timeの範囲や、video durationに対するwatch ratioでsliceを作ります。

単純なwatch ratioには、長いvideoほどdwell timeも長くなりやすいbiasがあります。そこで論文は、`log(dwell_time + 1)` を `log(γ × video_duration + 1)` で割るscoreも用意し、thresholdを超えたinteractionを重み付きで集約します。`γ` やthresholdの値は公開されていません。

さらに、一部のfeatureを全期間、直近3日、直近7日のtime bucketへ分けます。長期履歴を使いながら、最近の傾向も粗い時間粒度で残すためです。

```text
例：同じuser履歴から作るslice

liked media IDs（全期間）
liked media IDs（直近3日）
liked author IDs（直近7日）
watch time 3〜15秒のtopic IDs
短い視聴をdurationで補正したmedia IDs + weight
```

著者らはオフライン評価を繰り返し、最終的に200個の派生featureを選びました。通常のcategorical featureにはmean pooling、重み付きfeatureにはweighted mean poolingを適用します。

この設計はinteractionの厳密な順序を捨てる代わりに、「どの種類の行動を、どの対象へ、どの期間で行ったか」を多数の集計として残します。論文のAppendix Cでは、履歴長を3万5,000件まで伸ばすと、position-weighted poolingやHSTUを追加してもmulti-slicing baselineに対するgainが消えたと報告しています。HSTU自体はscalabilityの都合で1,000件までの比較です。

著者らの解釈は、細かな順序はemerging interestには有用でも、stable interestにはcountに近い集約が強いbaselineになる、というものです。これはInstagram内部data上の観測であり、順序が重要なすべてのdomainへ一般化できる結論ではありません。

## 24時間のgapで「長期的な興味」を学ばせる

上流modelは、生成したuser embeddingとtarget itemなどを使い、likeやshareのようなbinary taskをcross entropyで、watch timeをMSEで予測するmulti-task DLRMです。

重要なのは、予測対象のinteraction直前24時間をuser timelineから意図的に除くことです。

```text
過去の長い履歴 ────────── 24時間のgap ── 予測するinteraction
       ↑ user encoderが読む範囲
```

直前の行動へ頼れない難しい予測問題にすることで、user encoderへ長期的で変化しにくいinterestを学ばせます。結果として、data pipelineやembedding更新が遅れても壊れにくい表現を狙います。

user encoderだけを単独でself-supervised学習するのではなく、productionのranking modelに近いDLRMの中へ組み込むのも特徴です。論文はこれを**Backbone Simulation Network**と呼びます。上流modelの目的をdownstreamの予測目的へ近づけ、切り出したuser embeddingが実際のrankingで使いやすくなるよう監督します。

DLRM backboneとSiamese backboneを比較したところ、retrievalへの転送性能は同程度でしたが、DLRM backboneはdownstream rankingでRelative NEを0.2%改善しました。一方、backboneを大きくすることやtaskを増やすことの追加効果は小さく、sparse・denseの両featureをsimulationへ入れることは重要だったと報告されています。

## Funnel Summarization Arch：意味を壊さず50分の1へ圧縮する

Multi-slicing後の入力は `N = 200` 個、各 `D = 256` 次元の埋め込みです。Funnel Summarization Arch（FSA）はこれを通常のTransformerとは異なる向きで扱います。

一般的には `[N, D]` の `N` をsequenceとしてattentionへ入れます。FSAはtensorを `[D, N]` へ転置し、最後のtoken軸 `N` に同じnetworkを適用します。これにより、raw embeddingの各次元でparameterを共有し、出力でもembedding次元の意味を保つことを狙います。

Funnel Transformerのblock間ではtoken軸をmean poolingし、token数を段階的に減らします。並列にLinear Compression Encoder（LCE）も置き、両方の出力を合わせます。最終出力は58個の256次元embeddingです。

比較実験では、同じ58出力でもdim-wiseなMLP／Transformerよりtoken-wise版が一貫して良く、token-wise TransformerとFSAが上位でした。FSAは通常のTransformerよりparameterが少なく、training QPSが10%高かったため採用されています。

最後に4-bit quantizationを適用します。論文の表現では、`200 × 256` 個のFP32値を `58 × 17` 個の64-bit整数へpackし、約50倍に圧縮します。1 userあたりでは、58個の出力がそれぞれ136 bytes、合計約7.9 KBです。

## Downstream modelへは目的別に接続する

保存されたembeddingはdequantizeした後、downstream modelに合わせて軽量なadaptation moduleを通します。

| Downstream model | 接続方法 |
| --- | --- |
| DLRM ranking | 線形projection後、既存のsparse embeddingと連結 |
| HSTU | 直近行動sequenceの先頭へDV365 embeddingを追加 |
| Retrieval | Siamese／Mixture-of-Logits（MoL）でGateNetを利用 |

1つの上流embeddingをそのまま押し付けるのではなく、利用先ごとにprojectionやgateを学習するのがポイントです。stable interestという共通signalを共有しながら、Reels、Feed、notification、ranking、retrievalなどの目的差をdownstream側で吸収します。

## 評価結果：強いproduction baselineへ追加しても改善した

評価dataはInstagramの非公開recommendation logです。比較対象のReels ranking modelはDLRM、retrieval modelはMoLで、どちらもHSTUをsequence encoderに持つ当時のproduction modelでした。すでに約1,000件規模の履歴とattentionを使う強いbaselineへ、DV365を追加した効果を測っています。

rankingではNormalized Entropy（NE）を使います。binary cross entropyをlabelの平均頻度から計算したentropyで正規化した指標で、低いほど良いmetricです。論文のRelative NE Deltaは同じ期間のcontrolに対するpercent deltaで、負値が改善を表します。

線形projectionを使ったdownstream rankingの結果は次の通りです。

| Task | Relative NE／MSE delta |
| --- | ---: |
| video complete | -0.637% |
| skip | -0.508% |
| reshare | -0.179% |
| profile visit | -0.394% |
| follow | -0.592% |
| liked | -0.590% |
| save | -0.356% |
| comment | -1.151% |
| watch time MSE | -0.835% |

全taskで改善し、著者らは平均0.4%超のNE改善とまとめています。MetaではRelative NE `0.1%`超をonline A/B testのgainにつながる実務上の基準として扱う、と論文は説明しています。ただし、これはp-valueなどに基づく統計的有意性の定義ではありません。

retrievalではhit rate@1／@10を測定しています。たとえばGateNetを使うと、`follow@1` は0.413から0.437へ相対5.8%、`profile_visit@1` は0.374から0.402へ7.5%、`watch_time@10` は0.710から0.738へ3.9%改善しました。

論文本文はretrievalの改善を2〜8%と要約していますが、表全体にはそれより小さい値もあります。全taskを含めると、単純なDV365追加は相対0.5〜7.9%、GateNet版は1.3〜7.5%です。またGateNetが常に最大ではなく、reshareでは単純な接続の方が高い結果でした。

最終的にDV365はInstagramとThreadsの15個のproduction modelへ導入され、各launchのA/B testを累積してInstagram appのtime spentを0.7%改善したと報告されています。これはoffline NEとは別のonline成果です。一方、control、期間、traffic量、confidence interval、個別launchの寄与は公開されていません。

## Staleness：更新が遅れても本当に使えるのか

offline embeddingには、常に鮮度の問題があります。論文のproduction pipelineは定期更新ですが、上流の学習dataはonline trainingが使うdataより約2日遅れます。

Appendix Bでは、次の3モデルを7日分のdataで比較しています。

- DV365を使わないbaseline
- 毎日更新し、平均約2日古いDV365を使うmodel
- 実験中に固定し、日ごとに古くなるDV365を使うmodel

更新版と固定版はどちらもbaselineに対して約0.35%のNE gainを保ちました。また7日離れたembedding version間でもcosine similarityは90%を超えています。

この結果は、24時間gapと長期履歴によってstable interestを学習する設計と整合します。ただし検証期間は7日間です。「更新を止めても長期にわたり安全」という意味ではなく、数日単位のpipeline遅延に耐える根拠と捉えるべきです。急に変わる興味は、引き続きdownstream modelの新鮮なfeatureが担当します。

## Cost：offline化の価値は、1回作って15回使うことにある

論文は、同等の長期履歴処理を各modelへend-to-endで組み込む場合と、DV365を共有する場合のcostを試算しています。

| 項目 | 長い履歴をonline／各modelで処理する想定 | DV365 |
| --- | ---: | ---: |
| feature取得・加工 | 約100 MW | embedding servingの実測21 kW |
| 30億user分の低latency storage | 11 PB | 24 TB |
| Reels 1 modelのserving前処理 | 6 MW | offline側へ移動 |
| Reels 1 modelの追加推論GPU | H100 78基相当 | downstreamでは圧縮embeddingを利用 |

storageは、raw timelineの各attributeをint64で保持する想定と、4-bit量子化した58個のembeddingを比較しています。11 PBから24 TBは約458分の1です。

ただし、これらを同じ確度の実測値として読んではいけません。100 MWは、平均60件の履歴をonline取得したload testの150 kWを、平均4万件へ長さに比例して外挿した値です。6 MWも、長さ60で測った900 kWへ長さ比を掛け、さらに100倍の最適化余地を仮定した試算です。

trainingでは、1 consumerしかなければend-to-endとofflineのmodel計算costは同程度だと論文自身が認めています。利点は1つの上流modelを15個へ共有できることです。fleet全体の比較では、15個すべてが同じ重さではないため10倍相当と仮定し、追加前処理500 kW、A100 pool 3,380基相当を節約できると見積もっています。serving GPUもReelsの78 H100を10倍し、fleetで780 H100相当としています。

したがってDV365のROIは、単にmean poolingが安いことではなく、**高価な長期履歴処理をofflineへ移し、圧縮し、十分に多くのdownstream modelで再利用できる組織規模**に依存します。

## 自社systemで試すなら

Instagramのdata、training code、feature定義、threshold、FSAの全hyperparameter、serving実装は公開されていません。DV365そのものを完全再現することはできません。以下は原論文の再現手順ではなく、公開情報から導いた小規模な検証案です。

### 1. まず二つの時間軸が存在するか確かめる

直近履歴だけのbaselineに対し、古い履歴をactionや期間で集計したfeatureを追加します。離脱・復帰、seasonality、急な興味変化などのcohortを分け、長期signalが本当にincrementalか確認します。

```text
baseline: recent sequence encoder
treatment: recent sequence encoder + long-term aggregate

評価:
  ranking loss / retrieval hit rate / calibration
  active・inactive・new user別のquality
  feature取得cost / p95・p99 latency / storage
```

### 2. Timeline schemaを先に安定させる

eventを後から別のsliceへ再利用できるよう、source、action、item属性、event time、dwell timeをversion付きで保存します。削除要求やretention policyを反映できるよう、生成済みembeddingまで追跡できるlineageも必要です。

### 3. Multi-slicingをattentionより先に試す

action、期間、content type、duration、dwell timeで少数のsliceを作り、mean poolingをbaselineにします。slice数を増やすとfeature leakage、疎なbucket、計算量も増えるため、各追加featureをablationで選びます。

```text
for user in active_users:
  events = load_history(user, retention_window)
  pooled = []

  for rule in versioned_slice_rules:
    selected = rule.filter(events)
    pooled.append(weighted_mean(embed(selected), rule.weights))

  user_vector = summarizer(pooled)
  store(user, quantize(user_vector), model_version, generated_at)
```

### 4. Stalenessを意図的に作って評価する

毎日更新、3日前に固定、7日前に固定といったembeddingを同じdownstream modelへ入力し、quality curveを測ります。平均cosine similarityだけでなく、prediction、cohort、個々のuserでのdriftを確認します。急変したuserだけ更新頻度を上げる方法も比較対象になります。

### 5. Downstreamごとに小さなadapterを持つ

共有embeddingを直接連結するだけでなく、linear projection、gate、feature-wise modulationを同じparameter budgetで比較します。upstream metricだけで採用を決めず、rankingとretrievalそれぞれでknowledge transferを検証します。

### 6. Productionではcoverageとfallbackを設計する

embedding欠損、生成遅延、schema不一致、quantization version不一致が起きても、recent sequenceだけで推薦を継続できるようにします。監視対象には少なくとも次を含めます。

- embedding coverage、age、生成jobの遅延
- sliceごとのevent数と分布drift
- model version／feature schema versionの不一致
- dequantization後のnorm、NaN、cosine drift
- offline metric、online KPI、new／inactive user別のquality
- key-value storeのhit rate、latency、traffic、cost

## Limitationと適用しにくい条件

この研究には、解釈と再現の両面で制約があります。

- dataset、code、30億user分のpipeline、feature thresholdが非公開で、第三者が同条件を再現できない
- offline評価のdate range、sample数、複数runの分散、confidence interval、統計的検定が示されていない
- onlineのtime spent `+0.7%`は15 launchの累積値だが、実験設計と個別効果が公開されていない
- costの重要な値は、短いsequenceのload testからの線形外挿やfleet倍率を含む
- staleness実験は7日間で、季節変化やlife eventにまたがる長期安定性は分からない
- multi-slicingは人手で設計したruleへ依存し、200 featureの選択過程や全ablationは公開されていない
- offline embeddingの欠損率、inactive／new userのfallback、更新jobのfailure時設計は詳述されていない
- 長期間の行動を集約・共有することによるprivacy、retention、公平性への影響は評価されていない

また、consumer modelが少ないsystemでは、共通foundation model、定期batch、feature storeを増やす方が複雑になる可能性があります。userの意図が短時間で大きく変わるdomain、actionの順序そのものが重要なdomain、長期履歴が十分にないproductにも、そのままは適用できません。

## まとめ

DV365の本質は「Transformerで7万件を読んだ」ことではありません。**長期的な興味には毎requestの精密なsequence処理が本当に必要かを問い直し、offlineで一度だけ要約した表現を複数modelへ配ることで、qualityとROIを両立した**ことです。

- explicit／implicitな履歴を平均4万件、最大7万件まで集める
- action、期間、視聴時間などで200個へmulti-slicingし、poolingする
- 24時間のgapを置く予測taskでstable interestを学ばせる
- FSAと4-bit quantizationで50分の1へ圧縮する
- adapterを通して15個のranking／retrieval modelへ共有する
- 既存のHSTUを残し、recent interestとlong-term interestを分担する

production推薦では、最も表現力の高いmodelをすべてのdataへ適用することが最適とは限りません。signalが変わる速度に合わせて計算場所と更新頻度を分け、再利用できる部分をsystem全体で共有する。DV365は、その設計原則を大規模な実運用で示した事例です。

## 参照

- Wenhan Lyu et al., [DV365: Extremely Long User History Modeling at Instagram（arXiv abstract）](https://arxiv.org/abs/2506.00450)
- Wenhan Lyu et al., [DV365: Extremely Long User History Modeling at Instagram（HTML全文）](https://arxiv.org/html/2506.00450v1)
- Wenhan Lyu et al., [DV365: Extremely Long User History Modeling at Instagram（PDF）](https://arxiv.org/pdf/2506.00450)
- KDD 2025 proceedings, [DOI: 10.1145/3711896.3737209](https://doi.org/10.1145/3711896.3737209)
