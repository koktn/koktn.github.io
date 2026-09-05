---
title: Meta Adaptive Ranking Model――LLM級の広告rankingを100ms台で配信する設計
description: MetaがInstagram広告へ導入したAdaptive Ranking Modelを、request単位の計算共有、Wukong Turbo、FP8、multi-GPU embeddingと公開情報の限界から解説します。
publishedAt: 2026-09-06
category: AI
tags:
  - Recommendation System
  - Ads Ranking
  - Inference Optimization
  - GPU
  - Meta
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値や主張は参照元を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Meta Engineeringが2026年3月31日に公開し、4月21日（UTC）に更新した「[Meta Adaptive Ranking Model: Bending the Inference Scaling Curve to Serve LLM-Scale Models for Ads](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/)」です。

この記事の価値を一文でまとめると、**広告候補ごとに重複していたuser側の重い計算をrequest単位へまとめ、model・kernel・serving infrastructureを一体で最適化することで、LLM級の計算量とtrillion規模の疎なparameterを100ms台の制約へ収めた**点にあります。

## 課題：広告rankingにはLLMと違う厳しさがある

広告rankingは、userやcontextと多数の広告候補を組み合わせ、clickやconversionなどの確率を推定して表示対象を決めます。modelを大きくすれば、長い行動履歴や複雑なfeature interactionを扱いやすくなります。しかし、Metaが示すproduction要件には3者の緊張関係があります。

- **model quality**：より複雑な興味や意図を捉えたい
- **latency**：広告選択をsub-secondで完了しなければならない
- **cost**：traffic量が大きいため、hardwareを足すだけではROIが合わない

chatbotなら応答に数秒を使える場面がありますが、広告rankingはページやfeedを返すcritical pathにあります。しかも、1 requestで多数の広告候補をscoreするため、候補ごとに同じuser履歴を処理するとmodelの高度化ほど重複計算が膨らみます。

Meta Adaptive Ranking Model（以下、Adaptive Ranking Model）は、この「inference trilemma」を単一の高速化手法ではなく、計算flowからhardware配置までをまたぐco-designで解いています。

## 全体像：4層を同時に変える

公開記事の構成を、data flowとして整理すると次のようになります。

```text
request-level data
  ├─ userの長い行動sequenceを1回だけ処理
  └─ request-level embeddingを生成
             ↓ In-Kernel Broadcast
ad candidate群 ── 各candidate固有featureと結合
             ↓
Wukong Turboで高次のfeature interactionを計算
             ↓
selective FP8・operator fusion・Grouped GEMMで実行
             ↓
multi-GPUに分割した大規模embeddingを参照
             ↓
候補ごとのscore
```

対応する最適化は次の4層です。

| 層 | 中核となる変更 | 狙い |
| --- | --- | --- |
| 計算flow | Request-Oriented Optimization | candidate間の重複計算を除く |
| model | Wukong Turbo | 深くしても安定するfeature interactionを作る |
| graph／kernel | selective FP8、fusion、GPU前処理 | memory転送と小さなkernelのoverheadを減らす |
| serving | embedding sharding、remote cache、autoscaling | single GPUのmemory上限を越えて安定運用する |

Metaはさらに「contextとintentに応じて最も効果的かつ効率的なmodelへrequestをroutingする」と説明しています。ただし、routingの判定方法、modelの段数、学習方法、fallback条件は公開されていません。したがって、以下では詳細が示された計算共有とserving最適化を中心に見ます。

## 仕組み1：candidate中心からrequest中心へ変える

従来型の処理を単純化すると、userと広告候補のpairを独立にmodelへ入れます。

```text
for ad in candidates:
  user_state = encode(long_user_history)  # 同じrequest内で重複
  score[ad] = rank(user_state, ad)
```

候補数を `N`、user履歴の重い処理costを `U`、広告固有の処理costを `A` とすれば、概念的なcostは `N × (U + A)` です。Adaptive Ranking Modelはuser側の計算をrequestにつき1回へまとめます。

```text
user_state = encode(long_user_history)
for ad in candidates:
  score[ad] = rank(shared(user_state), ad)
```

概念上は `U + N × A` となり、`U`が大きいほど共有の効果が出ます。Metaはこれを**Request-Oriented Computation Sharing**と呼び、request-level embeddingをGPU kernel内でcandidateへ共有する**In-Kernel Broadcast**も組み合わせています。中間値をcandidate数だけ複製せず、memory bandwidthへの圧力を抑える狙いです。

この変更により、従来はcomputeとstorageの制約で扱いにくかった長いuser behavior sequenceも、requestごとに一度だけ処理できるようになります。training data側でもuser logを各exampleへ複製せず、中央のkey-value storeに置き、学習時にjoinする構成へ変えています。

元記事は、この設計によりmodel scaling costがlinearからsub-linearへ変わると表現しています。ただし、これは候補数やmodel sizeに対する厳密な計算量証明ではありません。candidate共通部分を増やして限界costを抑える、system上のscaling特性を指す表現として読むのが妥当です。

## 仕組み2：Wukongをproduction向けのTurboへ進化させる

modelの土台は、Metaが2024年に公開した「[Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545)」です。Wukongは、categorical featureをembeddingへ変換した後、Factorization Machine Block（FMB）とLinear Compress Block（LCB）を並列に置いた層をstackします。

```text
categorical／dense features
  → embeddings
  → [FMB: feature同士をinteraction]
    [LCB: 低次の情報を線形圧縮して保持]
  → concatenate + residual + normalization
  → 次のinteraction layer
  → prediction MLP
```

FMBは入力間の2次interactionを作り、その出力を次の層が再び組み合わせます。原論文では、`i` 層目までに1次から最大 `2^i` 次のinteractionを表現できると説明しています。LCBは低次の情報を残し、FMB側だけへ情報を押し込めない役割を持ちます。

論文版Wukongは、6つのpublic datasetで比較対象をAUCで上回り、Metaのinternal datasetではmodel complexityを2桁の範囲で増やして100 GFLOP/exampleを超えてもquality向上が続いたと報告しています。ただし、public datasetとinternal datasetの結果を混ぜてはいけません。大規模でのscaling lawはprivate data上のvendor評価であり、第三者が同条件を再現できるものではありません。

今回の**Wukong Turbo**は、これをruntime向けに調整したarchitectureです。

- **No-Bias approach**で数値的に不安定な項を除く
- 小さなparameterをFSDPからDDPへ移す**small parameter delegation**で通信とmemory overheadを減らす
- sparsityを利用してlinear layerの冗長なcomponentを簡略化する

Metaは、これらがFLOPsやparameter数を増やさずthroughputを高め、深いmodelの安定性とsub-second latencyを守ると説明しています。一方、No-Biasで具体的にどのbias項を除くのか、delegationのthreshold、sparsity pattern、qualityのablationは公開されていません。論文版Wukongの実装をそのまま変更すればWukong Turboを再現できる、という情報量ではありません。

## 仕組み3：前処理からkernelまでGPUに合わせる

大きなmodelでも、GPUが計算を待っていれば速くなりません。元記事は、CPU側のfeature preprocessingがclient memoryを圧迫し、GPUへのdata供給を止めるbottleneckだったと説明しています。

そこで前処理をremote GPU hostへoffloadし、compactなtuple形式とGPU-native kernelを導入しました。Top-K処理も `O(N log N)` から `O(N)` へ変え、data compressionとclient flowの再構成でthread-pool contentionを除いています。

model実行側の最適化は、主に次の3つです。

### Selective FP8 quantization

すべてのlayerを一律にFP8へ落とすのではなく、micro-benchmarkで低精度化への耐性が高いlayerを選び、post-training quantizationを適用します。元記事はrecommendation qualityへの影響をnegligibleとしていますが、対象metricや差の値は示していません。

### Graphとkernelのspecialization

同じ入力を使うoperatorをfusionし、High Bandwidth Memory（HBM）とon-chip SRAM間のdata movementを減らします。また、数千個の小さなoperationをGrouped General Matrix Multiply（Grouped GEMM）やhorizontal fusionでcompute-denseなkernelへまとめ、kernel launch overheadを抑えます。

### Hardwareごとのco-design

model graphを抽象的に最適化するだけでなく、異なるhardwareとsiliconの得意なprecision、memory階層、通信特性へ合わせています。Metaは複数hardware typeでModel FLOPs Utilization（MFU）35%を達成したと報告しています。ただし、hardware名、MFUの定義式、baseline、traffic条件は公開されていないため、他systemとの横比較には使えません。

## 仕組み4：1兆parameterは主に疎なembeddingとして扱う

元記事は、Adaptive Ranking Modelが`O(1T)`、すなわちtrillion規模へparameterを拡大できると述べています。ここでLLMのparameter数と同じ感覚で捉えると誤解します。

recommendation modelでは、user、item、categoryなどのcategorical IDを高次元vectorへ写す巨大なembedding tableがparameterの大部分を占めます。1 requestですべてのparameterをdenseに計算するLLMとは異なり、lookupされた一部のrowだけが使われる疎な構造です。したがって、**1T parameterは1T個を毎回演算するという意味ではありません**。

embedding tableを大きくしすぎればrare IDを覚えてoverfittingしやすくなり、小さすぎればhash collisionで別IDが同じ表現を共有してqualityが落ちます。Adaptive Ranking Modelは、次のmemory最適化でこのtrade-offを扱います。

- featureのsparsityに応じてembeddingのhash sizeを割り当てる
- 使われていないembeddingをpruneする
- 複数featureで1つのtableを共有するunified embeddingsを使う
- terabyte級のtableを複数GPU cardへshardingする

Metaはhardware固有の通信最適化を組み合わせ、multi-card構成でsingle-cardと同等のperformanceを達成したと述べています。ただし、ここでも「performance」がlatency、throughput、qualityのどれをどの条件で比較した値かは掲載されていません。

## Productionで壊さないための設計

大規模modelは、通常時の速さだけでなくdeploymentとtraffic変動にも耐える必要があります。Adaptive Ranking Modelは、multi-stream downloadとremote cacheによってmodelを10分未満でloadし、Streaming Multiprocessor（SM）utilizationをsignalにautoscaleします。

この設計から読み取れるのは、autoscalingを単純なrequest数だけで判断していないことです。同じrequest数でも、candidate数、sequence長、routing先のmodelによってGPU負荷は変わり得ます。実際のbottleneckに近いSM utilizationを見ることで、過剰provisioningを抑えながらtraffic変動へ追従する狙いがあります。

ただし、failure時のrouting、shard欠損時のfallback、model load中のtraffic切替、availability SLOは公開されていません。「production-grade reliability」という主張を、障害設計一式が開示されたとまでは解釈できません。

## 効果：Instagramでconversion +3%、CTR +5%

MetaはAdaptive Ranking Modelを2025年第4四半期にInstagramへ導入し、**targeted usersでad conversionが3%、ad click-through rate（CTR）が5%増加した**と報告しています。また、次のsystem規模も示しています。

| 指標 | Metaの報告値 |
| --- | ---: |
| model complexity | top-tier LLM相当の `O(10 GFLOPs)` per token |
| bounded latency | `O(100 ms)` |
| Model FLOPs Utilization | 複数hardware typeで35% |
| parameter scale | `O(1T)` |
| model load | 10分未満 |
| Instagram導入後のad conversion | +3% |
| Instagram導入後のCTR | +5% |

これらはMeta自身のproduction報告で、重要な成果です。一方、解釈に必要な次の条件は元記事にありません。

- conversionとCTRがA/B testによる因果効果か、導入前後の比較か
- control／treatmentのsample数、期間、confidence interval、p-value
- `targeted users`の定義と全trafficに占める割合
- baseline model、hardware、batch size、candidate数、sequence長
- `per token`でいうtokenの定義と、LLMとのcomplexity比較方法
- cost、power consumption、latencyの絶対分布やtail latency

したがって、+3%と+5%を別環境でも期待できる一般的な改善率とは扱えません。また、`O(10 GFLOPs)`、`O(100 ms)`、`O(1T)`は桁を表す概数であり、algorithmの漸近計算量ではありません。

## 自社systemへ応用するときの進め方

Wukong Turboのcode、training data、routing policy、kernel、hardware構成は公開されていないため、Adaptive Ranking Modelそのものは再現できません。ただし、設計原則は段階的に検証できます。以下は元記事の再現手順ではなく、公開情報から導いた実装案です。

### 1. Request内の重複をprofileする

まずcandidateごとのlatencyだけでなく、同じrequestで何度も生成・転送しているuser embedding、sequence表現、featureを測ります。request ID単位でCPU preprocessing、host-to-device transfer、kernel time、memory allocation、network communicationをtraceします。

### 2. 共通計算を1つだけ切り出す

最も重いuser-side encoderをrequest単位で一度だけ実行し、candidate batchへbroadcastします。最初はmodel architectureを変えず、出力が旧実装と許容誤差内で一致することを確認します。

```text
baseline: candidateごとにuser encoderを実行
treatment: requestごとに1回実行してcandidateへ共有

確認項目:
  prediction差 / p50・p95・p99 latency / throughput
  GPU memory / HBM bandwidth / network traffic / cost per request
```

### 3. 前処理をGPUへ移す前に計算密度を見る

CPU処理をそのままGPUへ移しても、小さなkernelが大量に起動すれば遅くなります。batch化、tuple形式、operator fusionを一緒に設計し、GPUがdata待ちになっている時間が実際に減るか確認します。Top-Kを線形時間へ変えられても、対象の `N` が小さければ転送costの方が大きい場合があります。

### 4. Quantizationはlayerごとに評価する

FP8対応hardwareだけを前提にせず、layer単位でlatency、quality、calibrationへの影響を測ります。offline metricが同じでも、scoreの微小な変化がauction順位やbidへ増幅される可能性があるため、最終的にはshadow trafficとonline experimentが必要です。

### 5. Dense scalingとsparse scalingを分けて管理する

interaction layerのdepth／widthを増やすdense scalingと、embedding tableを増やすsparse scalingでは、消費するcompute・memory・networkが異なります。parameter総数だけでmodelを比較せず、active parameter、FLOPs/request、embedding lookup量、通信量を別々に記録します。

### 6. 段階的にrolloutする

offline replay、shadow serving、少量trafficのA/B test、段階拡大の順に進めます。model qualityだけでなく、p99 latency、timeout率、load時間、GPU utilization、cost、conversion、CTRをguardrailとして持ち、旧modelへrequestを戻せるroutingを用意します。

## Limitationと今後の方向性

Adaptive Ranking Modelの公開情報には、次の限界があります。

- architecture名と最適化の方向は分かるが、実装詳細とablationが少ない
- production効果の実験設計と統計的不確実性が公開されていない
- latency、MFU、multi-card parityの測定条件がなく、第三者比較ができない
- 1T parameterのうちembeddingとdense componentが占める割合が不明
- personalizationの入力data、privacy、安全性、公平性への影響は扱われていない
- 広告主規模別の効果や、conversionとuser experienceのtrade-offは示されていない

Metaが今後の方向として挙げるのは、より高度なmodel compressionとultra-low precision、agentによるkernel optimization、incrementalなin-place weight updateによるmodel freshnessです。これらはroadmapであり、今回のproduction成果として検証済みではありません。

特にin-placeでweightを継続更新するなら、学習とservingの境界が薄くなります。更新途中の整合性、bad updateの検出、rollback、実験群の再現性、監査可能性が新たな課題になります。

## まとめ

Adaptive Ranking Modelから学べるのは、「巨大modelをservingするにはGPUを増やせばよい」のではなく、**重複を生む計算単位そのものを変える必要がある**という点です。

- userの重い処理をcandidateごとではなくrequestごとに1回実行する
- Wukong Turboで高次interactionを拡張しつつ、安定性と通信costを守る
- selective FP8、fusion、Grouped GEMMでhardwareの実効利用率を上げる
- 疎なembeddingをprune・共有・shardingし、single GPUのmemory上限を越える
- model loadとautoscalingまで含めてproduction systemとして設計する

そして、LLM級という言葉だけでなく、何がdense computeで、何がsparse parameterかを見ることが重要です。Adaptive Ranking Modelの本質はparameter数の記録ではなく、request内で共有できる情報を見つけ、modelとsystemを同じ境界で設計し直したことにあります。

## 参照

- Xi Chen et al., [Meta Adaptive Ranking Model: Bending the Inference Scaling Curve to Serve LLM-Scale Models for Ads](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/), Engineering at Meta, 2026-03-31（2026-04-21 UTC更新）。
- Buyun Zhang et al., [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545), arXiv:2403.02545v4, 2024-06-04（[HTML全文](https://arxiv.org/html/2403.02545)）。

本記事のAdaptive Ranking Modelに関する数値とproduction上の主張はMeta Engineeringの記事に基づきます。Wukongのarchitectureと評価条件は論文を参照しました。自社systemへの適用手順は、公開情報をもとにした記事側の提案です。
