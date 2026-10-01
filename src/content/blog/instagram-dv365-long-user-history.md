---
title: DV365：Instagramは7万件の行動履歴をどう推薦に使ったか
description: MetaのDV365を、Multi-slicing、Funnel Summarization、オフライン埋め込み、15モデルへの展開、精度・鮮度・コストの検証から解説します。
publishedAt: 2026-09-29
updatedAt: 2026-10-01
category: AI
tags:
  - Recommendation System
  - User Modeling
  - Representation Learning
  - Meta
  - Instagram
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Wenhan LyuらがKDD 2025で発表した「[DV365: Extremely Long User History Modeling at Instagram](https://arxiv.org/abs/2506.00450)」です。arXivには2025年5月31日にv1が投稿され、InstagramとThreadsの本番システムで1年以上運用した結果が報告されています。

この論文の価値は、最大7万件の行動履歴をリクエストごとに直接処理するのではなく、長期的で安定した興味を表す埋め込みへオフラインで圧縮し、1つの上流モデルから15個の推薦モデルへ共有できることを本番規模で示した点にあります。

## 課題：長い履歴は有用だが、オンラインで読むには高すぎる

推薦モデルにとって、ユーザーが過去に何を見て、likeし、shareしたかは重要なシグナルです。しかし履歴を長くするほど、次の3種類のコストが増えます。

- 低遅延な保存領域へ長い履歴を保持するコスト
- リクエストごとに履歴を取得・加工するCPUとネットワークのコスト
- attentionなどで系列をencodeする学習・推論GPUのコスト

論文執筆時のInstagramでは、本番モデルが参照できる行動履歴は最大約2,000件でした。その中でも、HSTUを使う高度なsequence encoderが直接処理するのは最大500件です。既存モデルは何年も最適化されており、系列長をそのまま伸ばすだけではROIが合わない状態でした。

一般的なlong-sequence手法には、長い履歴からtarget itemに関連する候補を検索し、絞り込んだ後にattentionをかける構成があります。この処理全体方式は新しい興味を細かく捉えやすい一方、各ranking modelが学習とservingの両方で長い履歴を扱わなければなりません。

InstagramにはReels、Feed、Explore、Storiesなど複数のsurfaceがあり、retrieval、early-stage ranking、late-stage rankingにも別モデルがあります。論文では20以上のモデルが存在するとしています。各モデルが同じ長期履歴を個別に処理すると、ほぼ同じ高価な計算を何度も繰り返すことになります。

DV365はここで、ユーザーの興味を2つの時間軸へ分けて考えます。

| 興味 | 性質 | 適した処理 |
| --- | --- | --- |
| emerging interest | 直近の出来事で素早く変わる | 各downstream modelの新鮮なsequence encoder |
| stable interest | 数時間・数日では大きく変わりにくい | 長い履歴から定期的に事前計算する共有埋め込み |

つまりDV365は既存のonline sequence modelを置き換えるものではありません。直近の履歴を読むHSTUなどは残し、そのモデルだけでは届かない長期シグナルを追加特徴量として渡します。

## 全体像：7万件を200個に集約し、58個へ圧縮する

DV365のdata flowを、オフラインとオンラインの責務に分けて整理すると次のようになります。

<picture>
  <source media="(max-width: 760px)" srcset="/img/posts/dv365-offline-online-architecture-mobile.svg">
  <img src="/img/posts/dv365-offline-online-architecture.svg" alt="DV365の構成。offlineでは最大7万件の長期履歴を200個へmulti-slicingし、FSAで58個に圧縮して共有する。onlineでは直近履歴を処理するHSTUとDV365 embeddingを合流させ、adapter経由で15個のproduction modelへ渡す">
</picture>

*図1：Wenhan Lyu et al.「[DV365: Extremely Long User History Modeling at Instagram](https://arxiv.org/html/2506.00450v1)」のFigure 1・2と本文の記述に基づき、本記事でオフライン／オンラインの責務へ再構成した図。原図の転載ではありません。*

上流のfoundation modelは定期的に学習され、user towerの中間出力を全活発なユーザーについて生成します。論文では30億ユーザー分の`user ID → embedding`をキーバリューストアへ保存し、埋め込みを6時間ごとに生成するとしています。

downstream modelはリクエスト時に圧縮済みのDV365 embeddingを取得するだけです。長いraw sequenceの取得、200種類のslicing、上流エンコーダーの推論をリクエストのcritical pathから外し、複数モデルで計算結果を共有します。

## UniTi：異なる行動を1本のtimelineとして持つ

入力となるUnified User Timeline（UniTi）は、ユーザーの行動を構造化した共通形式です。各interactionには、media ID、author ID、event timestamp、行動 type、video duration、watch time、surface type、media typeなどを持たせます。

timelineは大きく2種類です。

- explicit timeline：like、share、コメントなど、明示的な行動
- implicit timeline：表示されたコンテンツとdwell timeなど、視聴から得る暗黙的な行動

DV365が扱う長さは、implicit timelineが平均3万件・最大3万5,000件、explicit timelineが平均1万件・最大3万5,000件です。合計で平均4万件、最大7万件となります。最大値はシステム上の絶対限界ではなく、著者らがROIを考えて選んだcapです。

raw timelineのまま保持する利点は、特徴量設計をデータ生成時に固定しないことです。モデル側のpreprocessingで条件を変え、同じtimelineから異なるsliceを作れます。

## Multi-slicing：順序を精密に読む代わりに、複数の見方で数える

7万件すべてへself-attentionを適用するのではなく、DV365は履歴を意味のある条件で分割し、各subsetをpoolingします。

explicit timelineでは、like、コメント、shareといった行動 type、videoやphotoなどの参照元を分けます。implicit timelineでは、dwell timeの範囲や、video durationに対するwatch ratioでsliceを作ります。

単純なwatch ratioには、長いvideoほどdwell timeも長くなりやすいバイアスがあります。そこで論文は、`log(dwell_time + 1)`を`log(γ × video_duration + 1)`で割るscoreも用意し、しきい値を超えたinteractionを重み付きで集約します。`γ`やしきい値の値は公開されていません。

さらに、一部の特徴量を全期間、直近3日、直近7日のtime bucketへ分けます。長期履歴を使いながら、最近の傾向も粗い時間粒度で残すためです。

```text
例：同じuser履歴から作るslice

liked media IDs（全期間）
liked media IDs（直近3日）
liked author IDs（直近7日）
watch time 3〜15秒のtopic IDs
短い視聴をdurationで補正したmedia IDs + weight
```

著者らはオフライン評価を繰り返し、最終的に200個の派生特徴量を選びました。通常のcategorical featureにはmean pooling、重み付き特徴量にはweighted mean poolingを適用します。

この設計はinteractionの厳密な順序を捨てる代わりに、「どの種類の行動を、どの対象へ、どの期間で行ったか」を多数の集計として残します。論文のAppendix Cでは、履歴長を3万5,000件まで伸ばすと、position-weighted poolingやHSTUを追加してもmulti-slicing baselineに対するgainが消えたと報告しています。HSTU自体はscalabilityの都合で1,000件までの比較です。

著者らの解釈は、細かな順序はemerging interestには有用でも、stable interestには件数に近い集約が強い比較手法になる、というものです。これはInstagram内部データ上の観測であり、順序が重要なすべてのドメインへ一般化できる結論ではありません。

## 24時間のgapで「長期的な興味」を学ばせる

上流モデルは、生成したユーザー埋め込みとtarget itemなどを使い、likeやshareのようなbinary taskをcross entropyで、watch timeをMSEで予測するmulti-task DLRMです。

予測対象のinteraction直前24時間を、ユーザーの時系列履歴から意図的に除くことが重要です。

```text
過去の長い履歴 ────────── 24時間のgap ── 予測するinteraction
       ↑ user encoderが読む範囲
```

直前の行動へ頼れない難しい予測問題にすることで、ユーザーエンコーダーへ長期的で変化しにくいinterestを学ばせます。結果として、データパイプラインや埋め込み更新が遅れても性能が低下しにくい表現を狙います。

ユーザーエンコーダーだけを単独でself-supervised学習するのではなく、本番環境のranking modelに近いDLRMの中へ組み込むのも特徴です。論文はこれをBackbone Simulation Networkと呼びます。上流モデルの目的をdownstreamの予測目的へ近づけ、切り出したユーザー埋め込みが実際のランキングで使いやすくなるよう監督します。

DLRM backboneとSiamese backboneを比較したところ、retrievalへの転送性能は同程度でしたが、DLRM backboneはdownstream rankingでRelative NEを0.2%改善しました。一方、backboneを大きくすることやタスクを増やすことの追加効果は小さく、sparse・denseの両特徴量をシミュレーションへ入れることは重要だったと報告されています。

## Funnel Summarization Arch：意味を保って50分の1へ圧縮する

Multi-slicing後の入力は`N = 200`個、各`D = 256`次元の埋め込みです。Funnel Summarization Arch（FSA）はこれを通常のTransformerとは異なる向きで扱います。

一般的には`[N, D]`の`N`を系列としてattentionへ入れます。FSAはtensorを`[D, N]`へ転置し、最後のトークン軸`N`に同じネットワークを適用します。これにより、raw embeddingの各次元でパラメータを共有し、出力でも埋め込み次元の意味を保つことを狙います。

Funnel Transformerのblock間ではトークン軸をmean poolingし、トークン数を段階的に減らします。並列にLinear Compression Encoder（LCE）も置き、両方の出力を合わせます。最終出力は58個の256次元埋め込みです。

<picture>
  <source media="(max-width: 760px)" srcset="/img/posts/dv365-fsa-compression-flow-mobile.svg">
  <img src="/img/posts/dv365-fsa-compression-flow.svg" alt="Funnel Summarization Archの圧縮flow。200個×256次元を256×200へ転置し、Funnel TransformerとLCEの並列経路で58個×256次元へ要約した後、4-bit量子化する">
</picture>

*図2：同論文のSection 2.3.2、アルゴリズム1、Figure 1の記述に基づき、本記事でFSAのtensor形状と圧縮方向を新規に図解。原図の転載ではありません。*

比較実験では、同じ58出力でもdim-wiseなMLP／Transformerよりtoken-wise版が一貫して良く、token-wise TransformerとFSAが上位でした。FSAは通常のTransformerよりパラメータが少なく、training QPSが10%高かったため採用されています。

最後に4-bit quantizationを適用します。論文の表現では、`200 × 256`個のFP32値を`58 × 17`個の64-bit整数へpackし、約50倍に圧縮します。1ユーザーあたりでは、58個の出力がそれぞれ136 bytes、合計約7.9 KBです。

## Downstream modelへは目的別に接続する

保存された埋め込みはdequantizeした後、downstream modelに合わせて軽量なadaptation moduleを通します。

| Downstream model | 接続方法 |
| --- | --- |
| DLRM ranking | 線形projection後、既存のsparse embeddingと連結 |
| HSTU | 直近行動系列の先頭へDV365 embeddingを追加 |
| Retrieval | Siamese／Mixture-of-Logits（MoL）でGateNetを利用 |

1つの上流埋め込みをそのまま押し付けるのではなく、利用先ごとにprojectionやgateを学習するのがポイントです。stable interestという共通シグナルを共有しながら、Reels、Feed、notification、ランキング、retrievalなどの目的差をdownstream側で吸収します。

## 評価結果：強いproduction baselineへ追加しても改善した

評価データはInstagramの非公開recommendation logです。比較対象のReels ranking modelはDLRM、retrieval modelはMoLで、どちらもHSTUをsequence encoderに持つ当時の本番モデルでした。すでに約1,000件規模の履歴とattentionを使う強い比較手法へ、DV365を追加した効果を測っています。

ランキングではNormalized Entropy（NE）を使います。binary cross entropyをラベルの平均頻度から計算したentropyで正規化した指標で、低いほど良い指標です。論文のRelative NE Deltaは同じ期間の制御に対するpercent deltaで、負値が改善を表します。

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

全タスクで改善し、著者らは平均0.4%超のNE改善とまとめています。MetaではRelative NE `0.1%`超をオンラインA/Bテストのgainにつながる実務上の基準として扱う、と論文は説明しています。ただし、これはp-valueなどに基づく統計的有意性の定義ではありません。

retrievalではhit rate@1／@10を測定しています。たとえばGateNetを使うと、`follow@1`は0.413から0.437へ相対5.8%、`profile_visit@1`は0.374から0.402へ7.5%、`watch_time@10`は0.710から0.738へ3.9%改善しました。

論文本文はretrievalの改善を2〜8%と要約していますが、表全体にはそれより小さい値もあります。全タスクを含めると、単純なDV365追加は相対0.5〜7.9%、GateNet版は1.3〜7.5%です。またGateNetが常に最大ではなく、reshareでは単純な接続の方が高い結果でした。

最終的にDV365はInstagramとThreadsの15個の本番モデルへ導入され、各launchのA/B testを累積してInstagram appのtime spentを0.7%改善したと報告されています。これはoffline NEとは別のオンライン成果です。一方、制御、期間、トラフィック量、信頼区間、個別launchの寄与は公開されていません。

評価の層を混ぜないよう、結果をまとめると次のようになります。

| Evidenceの層 | 比較対象 | 論文の結果 | 読み方 |
| --- | --- | --- | --- |
| FSA ablation | 圧縮方式・エンコーダー | token-wiseがdim-wiseを上回り、FSAは通常Transformer比でtraining QPS +10% | 上流ユーザーエンコーダーの設計比較 |
| Downstream ranking | HSTUを含むReels production baseline | 全タスク改善、平均Relative NE -0.4%超 | 非公開データ上のオフライン評価 |
| Downstream retrieval | MoL baseline | hit rateの相対改善は全表で0.5〜7.9%、GateNet版1.3〜7.5% | タスク・接続方法で効果が異なる |
| Production rollout | Instagram／Threadsの15モデル | 累積A/B testでInstagram time spent +0.7% | 実験期間や個別寄与は非公開 |

## Staleness：更新が遅れても本当に使えるのか

offline embeddingには、常に鮮度の問題があります。論文のproduction pipelineは定期更新ですが、上流の学習データはonline trainingが使うデータより約2日遅れます。

Appendix Bでは、次の3モデルを7日分のデータで比較しています。

- DV365を使わない比較手法
- 毎日更新し、平均約2日古いDV365を使うモデル
- 実験中に固定し、日ごとに古くなるDV365を使うモデル

更新版と固定版はどちらも比較手法に対して約0.35%のNE gainを保ちました。また7日離れたembedding version間でもcosine similarityは90%を超えています。

この結果は、24時間gapと長期履歴によってstable interestを学習する設計と整合します。ただし検証期間は7日間です。「更新を止めても長期にわたり安全」という意味ではなく、数日単位のパイプライン遅延に耐える根拠と捉えるべきです。急に変わる興味は、引き続きdownstream modelの新鮮な特徴量が担当します。

## コスト：オフライン化の価値は、1回作って15回使うことにある

論文は、同等の長期履歴処理を各モデルへ処理全体で組み込む場合と、DV365を共有する場合のコストを試算しています。

| 項目 | 長い履歴をオンライン／各モデルで処理する想定 | DV365 | 数値の根拠 |
| --- | ---: | ---: | --- |
| 特徴量取得・加工 | 約100 MW | embedding serving 21 kW | 左は長さ60のload testから40Kへ線形外挿、右はproduction load testの実測 |
| 30億ユーザー分の低latency 保存領域 | 11 PB | 24 TB | raw attributeと量子化埋め込みのsizeから算出 |
| Reels 1モデルのserving前処理 | 6 MW | オフライン側へ移動 | 900 kWのload test、長さ比、`0.01`の最適化係数による試算 |
| Reels 1モデルの追加推論GPU | H100 78基相当 | downstreamでは圧縮埋め込みを利用 | ユーザーエンコーダー追加時のinference QPS低下から換算 |

保存領域は、raw timelineの各attributeをint64で保持する想定と、4-bit量子化した58個の埋め込みを比較しています。11 PBから24 TBは約458分の1です。

ただし、これらを同じ確度の実測値として読んではいけません。100 MWは、平均60件の履歴をオンライン取得したload testの150 kWを、平均4万件へ長さに比例して外挿した値です。6 MWも、長さ60で測った900 kWへ長さ比を掛け、さらに100倍の最適化余地を仮定した試算です。

学習では、1 consumerしかなければ処理全体とオフラインのモデル計算コストは同程度だと論文自身が認めています。利点は1つの上流モデルを15個へ共有できることです。fleet全体の比較では、15個すべてが同じ重さではないため10倍相当と仮定し、追加前処理500 kW、A100 pool 3,380基相当を節約できると見積もっています。serving GPUもReelsの78 H100を10倍し、fleetで780 H100相当としています。

したがってDV365のROIは、単にmean poolingが安いことではなく、高価な長期履歴処理をオフラインへ移し、圧縮し、十分に多くのdownstream modelで再利用できる組織規模に依存します。

## 自社システムで試すなら

Instagramのデータ、training code、特徴量定義、しきい値、FSAの全ハイパーパラメータ、serving実装は公開されていません。DV365そのものを完全再現することはできません。以下は原論文の再現手順ではなく、公開情報から導いた小規模な検証案です。

### 1. まず二つの時間軸が存在するか確かめる

直近履歴だけの比較手法に対し、古い履歴を行動や期間で集計した特徴量を追加します。離脱・復帰、seasonality、急な興味変化などのcohortを分け、長期シグナルが本当にincrementalか確認します。

```text
baseline: recent sequence encoder
treatment: recent sequence encoder + long-term aggregate

評価:
  ranking loss / retrieval hit rate / calibration
  active・inactive・new user別のquality
  feature取得cost / p95・p99 latency / storage
```

### 2. Timeline schemaを先に安定させる

イベントを後から別のsliceへ再利用できるよう、参照元、行動、アイテム属性、event time、dwell timeをバージョン付きで保存します。削除要求やretention policyを反映できるよう、生成済み埋め込みまで追跡できるlineageも必要です。

### 3. Multi-slicingをattentionより先に試す

行動、期間、コンテンツの種類、duration、dwell timeで少数のsliceを作り、mean poolingを比較手法にします。slice数を増やすとfeature leakage、疎なbucket、計算量も増えるため、各追加特徴量をablationで選びます。

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

毎日更新、3日前に固定、7日前に固定といった埋め込みを同じdownstream modelへ入力し、quality curveを測ります。平均cosine similarityだけでなく、prediction、cohort、個々のユーザーでのdriftを確認します。急変したユーザーだけ更新頻度を上げる方法も比較対象になります。

### 5. Downstreamごとに小さなadapterを持つ

共有埋め込みを直接連結するだけでなく、linear projection、gate、feature-wise modulationを同じparameter 予算で比較します。upstream metricだけで採用を決めず、ランキングとretrievalそれぞれでknowledge transferを検証します。

### 6. 本番環境ではcoverageとfallbackを設計する

埋め込み欠損、生成遅延、schema不一致、quantization version不一致が起きても、recent sequenceだけで推薦を継続できるようにします。監視対象には少なくとも次を含めます。

- embedding coverage、age、生成ジョブの遅延
- sliceごとのイベント数と分布drift
- モデルのバージョン／feature schema versionの不一致
- dequantization後のnorm、NaN、cosine drift
- offline metric、online KPI、new／inactive user別の品質
- キーバリューストアのhit rate、遅延、トラフィック、コスト

## 制約と適用しにくい条件

この研究には、解釈と再現の両面で制約があります。

データセット、コード、30億ユーザー分のパイプライン、feature thresholdが非公開で、第三者が同条件を再現できません。オフライン評価のdate range、サンプル数、複数runの分散、信頼区間、統計的検定が示されていない。

オンラインのtime spent `+0.7%`は15 launchの累積値だが、実験設計と個別効果が公開されていない。コストの重要な値は、短い系列のload testからの線形外挿やfleet倍率を含みます。

staleness実験は7日間で、季節変化やlife eventにまたがる長期安定性は分からない。multi-slicingは人手で設計した規則へ依存し、200特徴量の選択過程や全ablationは公開されていない。

offline embeddingの欠損率、inactive／new userのfallback、更新ジョブのfailure時設計は詳述されていない。長期間の行動を集約・共有することによるプライバシー、retention、公平性への影響は評価されていない。


また、consumer modelが少ないシステムでは、共通foundation model、定期バッチ、feature storeを増やす方が複雑になる可能性があります。ユーザーの意図が短時間で大きく変わるドメイン、行動の順序そのものが重要なドメイン、長期履歴が十分にないproductにも、そのままは適用できません。

## まとめ

DV365の本質は「Transformerで7万件を読んだ」ことではありません。長期的な興味には毎リクエストの精密な系列処理が本当に必要かを問い直し、オフラインで一度だけ要約した表現を複数モデルへ配ることで、品質とROIを両立したことです。

explicit／implicitな履歴を平均4万件、最大7万件まで集める。行動、期間、視聴時間などで200個へmulti-slicingし、poolingします。

24時間のgapを置く予測タスクでstable interestを学ばせる。FSAと4-bit quantizationで50分の1へ圧縮します。

adapterを通して15個のランキング／retrieval modelへ共有します。既存のHSTUを残し、recent interestとlong-term interestを分担します。


本番環境推薦では、最も表現力の高いモデルをすべてのデータへ適用することが最適とは限りません。シグナルが変わる速度に合わせて計算場所と更新頻度を分け、再利用できる部分をシステム全体で共有する。DV365は、その設計原則を大規模な実運用で示した事例です。

## 参照

- Wenhan Lyu et al., [DV365: Extremely Long User History Modeling at Instagram（arXiv abstract）](https://arxiv.org/abs/2506.00450)
- Wenhan Lyu et al., [DV365: Extremely Long User History Modeling at Instagram（HTML全文）](https://arxiv.org/html/2506.00450v1)
- Wenhan Lyu et al., [DV365: Extremely Long User History Modeling at Instagram（PDF）](https://arxiv.org/pdf/2506.00450)
- KDD 2025 proceedings, [DOI: 10.1145/3711896.3737209](https://doi.org/10.1145/3711896.3737209)
