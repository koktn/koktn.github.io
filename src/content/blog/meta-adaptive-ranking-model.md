---
title: Meta Adaptive Ranking Model：LLM級の広告ランキングを100ms台で配信する設計
description: MetaがInstagram広告へ導入したAdaptive Ranking Modelを、リクエスト単位の計算共有、Wukong Turbo、FP8、multi-GPU embeddingと公開情報の限界から解説します。
publishedAt: 2026-09-06
updatedAt: 2026-10-01
category: AI
tags:
  - Recommendation System
  - Ads Ranking
  - Inference Optimization
  - GPU
  - Meta
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は参照元を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Meta Engineeringが2026年3月31日に公開し、4月21日（UTC）に更新した「[Meta Adaptive Ranking Model: Bending the Inference Scaling Curve to Serve LLM-Scale Models for Ads](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/)」です。

この記事の価値は、広告候補ごとに重複していたユーザー側の重い計算をリクエスト単位へまとめ、モデル・kernel・serving infrastructureを一体で最適化することで、LLM級の計算量とtrillion規模の疎なパラメータを100ms台の制約へ収めた点にあります。

## 課題：広告ランキングにはLLMと違う厳しさがある

広告ランキングは、ユーザーや文脈と多数の広告候補を組み合わせ、クリックやconversionなどの確率を推定して表示対象を決めます。モデルを大きくすれば、長い行動履歴や複雑な特徴量の相互作用を扱いやすくなります。しかし、Metaが示す本番環境要件には3者の緊張関係があります。

- model quality：より複雑な興味や意図を捉えたい
- 遅延：広告選択をsub-secondで完了しなければならない
- コスト：トラフィック量が大きいため、ハードウェアを足すだけではROIが合わない

chatbotなら応答に数秒を使える場面がありますが、広告ランキングはページやfeedを返すcritical pathにあります。しかも、1リクエストで多数の広告候補をscoreするため、候補ごとに同じユーザー履歴を処理するとモデルの高度化ほど重複計算が膨らみます。

Meta Adaptive Ranking Model（以下、Adaptive Ranking Model）は、この「inference trilemma」を単一の高速化手法ではなく、計算流れからハードウェア配置までをまたぐco-designで解いています。

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
| 計算流れ | Request-Oriented Optimization | 候補間の重複計算を除く |
| モデル | Wukong Turbo | 深くしても安定する特徴量の相互作用を作る |
| グラフ／kernel | selective FP8、fusion、GPU前処理 | メモリ転送と小さなkernelのoverheadを減らす |
| serving | embedding sharding、remote cache、autoscaling | single GPUのメモリ上限を越えて安定運用する |

Metaはさらに「文脈とintentに応じて最も効果的かつ効率的なモデルへリクエストをroutingする」と説明しています。ただし、routingの判定方法、モデルの段数、学習方法、fallback条件は公開されていません。したがって、以下では詳細が示された計算共有とserving最適化を中心に見ます。

## 仕組み1：候補中心からリクエスト中心へ変える

従来型の処理を単純化すると、ユーザーと広告候補のペアを独立にモデルへ入れます。

```text
for ad in candidates:
  user_state = encode(long_user_history)  # 同じrequest内で重複
  score[ad] = rank(user_state, ad)
```

候補数を`N`、ユーザー履歴の重い処理コストを`U`、広告固有の処理コストを`A`とすれば、概念的なコストは`N × (U + A)`です。Adaptive Ranking Modelはユーザー側の計算をリクエストにつき1回へまとめます。

```text
user_state = encode(long_user_history)
for ad in candidates:
  score[ad] = rank(shared(user_state), ad)
```

概念上は`U + N × A`となり、`U`が大きいほど共有の効果が出ます。MetaはこれをRequest-Oriented Computation Sharingと呼び、request-level embeddingをGPU kernel内で候補へ共有するIn-Kernel Broadcastも組み合わせています。中間値を候補数だけ複製せず、memory bandwidthへの圧力を抑える狙いです。

この変更により、従来は計算量と保存領域の制約で扱いにくかった長いuser behavior sequenceも、リクエストごとに一度だけ処理できるようになります。学習データ側でもuser logを各exampleへ複製せず、中央のキーバリューストアに置き、学習時にjoinする構成へ変えています。

元記事は、この設計によりmodel scaling costがlinearからsub-linearへ変わると表現しています。ただし、これは候補数やmodel sizeに対する厳密な計算量証明ではありません。候補共通部分を増やして限界コストを抑える、システム上のscaling特性を指す表現として読むのが妥当です。

## 仕組み2：Wukongを本番向けのTurboへ進化させる

モデルの基礎となったのは、Metaが2024年に公開した「[Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545)」です。Wukongは、categorical featureを埋め込みへ変換した後、Factorization Machine Block（FMB）とLinear Compress Block（LCB）を並列に置いた層をstackします。

```text
categorical／dense features
  → embeddings
  → [FMB: feature同士をinteraction]
    [LCB: 低次の情報を線形圧縮して保持]
  → concatenate + residual + normalization
  → 次のinteraction layer
  → prediction MLP
```

FMBは入力間の2次interactionを作り、その出力を次の層が再び組み合わせます。原論文では、`i`層目までに1次から最大`2^i`次のinteractionを表現できると説明しています。LCBは低次の情報を残し、FMB側だけへ情報を押し込めない役割を持ちます。

論文版Wukongは、6つのpublic datasetで比較対象をAUCで上回り、Metaのinternal datasetではmodel complexityを2桁の範囲で増やして100 GFLOP/exampleを超えても品質向上が続いたと報告しています。ただし、public datasetとinternal datasetの結果を混ぜてはいけません。大規模でのスケーリング則はprivate data上のvendor評価であり、第三者が同条件を再現できるものではありません。

今回のWukong Turboは、これをruntime向けに調整したアーキテクチャです。

No-Bias approachで数値的に不安定な項を除く。小さなパラメータをFSDPからDDPへ移すsmall parameter delegationで通信とmemory overheadを減らす。

sparsityを利用してlinear layerの冗長な構成要素を簡略化します。


Metaは、これらがFLOPsやパラメータ数を増やさずスループットを高め、深いモデルの安定性とsub-second latencyを守ると説明しています。一方、No-Biasで具体的にどのバイアス項を除くのか、delegationのしきい値、sparsity pattern、品質のablationは公開されていません。論文版Wukongの実装をそのまま変更すればWukong Turboを再現できる、という情報量ではありません。

## 仕組み3：前処理からkernelまでGPUに合わせる

大きなモデルでも、GPUが計算を待っていれば速くなりません。元記事は、CPU側のfeature preprocessingがclient memoryを圧迫し、GPUへのデータ供給を止める制約になる箇所だったと説明しています。

そこで前処理をremote GPU hostへoffloadし、compactなtuple形式とGPU-native kernelを導入しました。Top-K処理も`O(N log N)`から`O(N)`へ変え、data compressionとclient flowの再構成でthread-pool contentionを除いています。

モデル実行側の最適化は、主に次の3つです。

### Selective FP8 quantization

すべての層を一律にFP8へ落とすのではなく、micro-benchmarkで低精度化への耐性が高い層を選び、post-training quantizationを適用します。元記事は推薦の品質への影響をnegligibleとしていますが、対象指標や差の値は示していません。

### Graphとkernelのspecialization

同じ入力を使うoperatorをfusionし、High Bandwidth Memory（HBM）とon-chip SRAM間のdata movementを減らします。また、数千個の小さなoperationをGrouped General Matrix Multiply（Grouped GEMM）やhorizontal fusionでcompute-denseなkernelへまとめ、kernel launch overheadを抑えます。

### ハードウェアごとのco-design

model グラフを抽象的に最適化するだけでなく、異なるハードウェアとsiliconの得意なprecision、メモリ階層、通信特性へ合わせています。Metaは複数hardware typeでModel FLOPs Utilization（MFU）35%を達成したと報告しています。ただし、ハードウェア名、MFUの定義式、比較手法、トラフィック条件は公開されていないため、他システムとの横比較には使えません。

## 仕組み4：1兆パラメータは主に疎な埋め込みとして扱う

元記事は、Adaptive Ranking Modelが`O(1T)`、すなわちtrillion規模へパラメータを拡大できると述べています。ここでLLMのパラメータ数と同じ感覚で捉えると誤解します。

recommendation modelでは、ユーザー、アイテム、カテゴリなどのcategorical IDを高次元ベクトルへ写す巨大な埋め込みテーブルがパラメータの大部分を占めます。1リクエストですべてのパラメータをdenseに計算するLLMとは異なり、参照された一部のrowだけが使われる疎な構造です。したがって、1T parameterは1T個を毎回演算するという意味ではありません。

埋め込みテーブルを大きくしすぎればrare IDを覚えてoverfittingしやすくなり、小さすぎればhash collisionで別IDが同じ表現を共有して品質が落ちます。Adaptive Ranking Modelは、次のメモリ最適化でこのトレードオフを扱います。

- 特徴量のsparsityに応じて埋め込みのhash sizeを割り当てる
- 使われていない埋め込みをpruneする
- 複数特徴量で1つのtableを共有するunified embeddingsを使う
- terabyte級のtableを複数GPU cardへshardingする

Metaはハードウェア固有の通信最適化を組み合わせ、multi-card構成でsingle-cardと同等のperformanceを達成したと述べています。ただし、ここでも「performance」が遅延、スループット、品質のどれをどの条件で比較した値かは掲載されていません。

## 本番環境で安定して運用するための設計

大規模モデルは、通常時の速さだけでなく本番導入とトラフィック変動にも耐える必要があります。Adaptive Ranking Modelは、multi-stream downloadとremote cacheによってモデルを10分未満でloadし、Streaming Multiprocessor（SM）utilizationをシグナルにautoscaleします。

この設計から読み取れるのは、autoscalingを単純なリクエスト数だけで判断していないことです。同じリクエスト数でも、候補数、系列長、routing先のモデルによってGPU負荷は変わり得ます。実際の制約になる箇所に近いSM utilizationを見ることで、過剰provisioningを抑えながらトラフィック変動へ追従する狙いがあります。

ただし、failure時のrouting、shard欠損時のfallback、model load中のトラフィック切替、availability SLOは公開されていません。「production-grade reliability」という主張を、障害設計一式が開示されたとまでは解釈できません。

## 効果：Instagramでconversion +3%、CTR +5%

MetaはAdaptive Ranking Modelを2025年第4四半期にInstagramへ導入し、targeted usersでad conversionが3%、ad click-through rate（CTR）が5%増加したと報告しています。また、次のシステム規模も示しています。

| 指標 | Metaの報告値 |
| --- | ---: |
| model complexity | top-tier LLM相当の`O(10 GFLOPs)` per token |
| bounded latency | `O(100 ms)` |
| Model FLOPs Utilization | 複数hardware typeで35% |
| parameter 規模 | `O(1T)` |
| model load | 10分未満 |
| Instagram導入後のad conversion | +3% |
| Instagram導入後のCTR | +5% |

これらはMeta自身の本番環境報告で、重要な成果です。一方、解釈に必要な次の条件は元記事にありません。

- conversionとCTRがA/B testによる因果効果か、導入前後の比較か
- 対照群／介入群のサンプル数、期間、信頼区間、p-value
- `targeted users`の定義と全トラフィックに占める割合
- baseline model、ハードウェア、バッチサイズ、候補数、系列長
- `per token`でいうトークンの定義と、LLMとのcomplexity比較方法
- コスト、power consumption、遅延の絶対分布や遅延の裾部分

したがって、+3%と+5%を別環境でも期待できる一般的な改善率とは扱えません。また、`O(10 GFLOPs)`、`O(100 ms)`、`O(1T)`は桁を表す概数であり、アルゴリズムの漸近計算量ではありません。

## 自社システムへ応用するときの進め方

Wukong Turboのコード、学習データ、routing policy、kernel、ハードウェア構成は公開されていないため、Adaptive Ranking Modelそのものは再現できません。ただし、設計原則は段階的に検証できます。以下は元記事の再現手順ではなく、公開情報から導いた実装案です。

### 1. リクエスト内の重複をprofileする

まず候補ごとの遅延だけでなく、同じリクエストで何度も生成・転送しているユーザー埋め込み、系列表現、特徴量を測ります。request ID単位でCPU preprocessing、host-to-device transfer、kernel time、memory allocation、network communicationをtraceします。

### 2. 共通計算を1つだけ切り出す

最も重いuser-side encoderをリクエスト単位で一度だけ実行し、候補 batchへbroadcastします。最初はmodel architectureを変えず、出力が旧実装と許容誤差内で一致することを確認します。

```text
baseline: candidateごとにuser encoderを実行
treatment: requestごとに1回実行してcandidateへ共有

確認項目:
  prediction差 / p50・p95・p99 latency / throughput
  GPU memory / HBM bandwidth / network traffic / cost per request
```

### 3. 前処理をGPUへ移す前に計算密度を見る

CPU処理をそのままGPUへ移しても、小さなkernelが大量に起動すれば遅くなります。バッチ化、tuple形式、operator fusionを一緒に設計し、GPUがデータ待ちになっている時間が実際に減るか確認します。Top-Kを線形時間へ変えられても、対象の`N`が小さければ転送コストの方が大きい場合があります。

### 4. Quantizationは層ごとに評価する

FP8対応ハードウェアだけを前提にせず、層単位で遅延、品質、calibrationへの影響を測ります。offline metricが同じでも、scoreの微小な変化がauction順位やbidへ増幅される可能性があるため、最終的にはshadow trafficとonline experimentが必要です。

### 5. Dense scalingとsparse scalingを分けて管理する

interaction layerのdepth／widthを増やすdense scalingと、埋め込みテーブルを増やすsparse scalingでは、消費する計算量・メモリ・ネットワークが異なります。パラメータ総数だけでモデルを比較せず、active parameter、FLOPs/request、embedding lookup量、通信量を別々に記録します。

### 6. 段階的にrolloutする

offline replay、shadow serving、少量トラフィックのA/B test、段階拡大の順に進めます。model qualityだけでなく、p99 latency、タイムアウト率、load時間、GPU utilization、コスト、conversion、CTRをguardrailとして持ち、旧モデルへリクエストを戻せるroutingを用意します。

## 制約と今後の方向性

Adaptive Ranking Modelの公開情報には、次の限界があります。

アーキテクチャ名と最適化の方向は分かるが、実装詳細とablationが少ない。本番環境効果の実験設計と統計的不確実性が公開されていない。

遅延、MFU、multi-card parityの測定条件がなく、第三者比較ができません。1T parameterのうち埋め込みとdense componentが占める割合が不明。

personalizationの入力データ、プライバシー、安全性、公平性への影響は扱われていない。広告主規模別の効果や、conversionとuser experienceのトレードオフは示されていない。


Metaが今後の方向として挙げるのは、より高度なmodel compressionとultra-low precision、エージェントによるkernel optimization、incrementalなin-place weight updateによるmodel freshnessです。これらはroadmapであり、今回の本番環境成果として検証済みではありません。

特にin-placeでweightを継続更新するなら、学習とservingの境界が薄くなります。更新途中の整合性、bad updateの検出、切り戻し、実験群の再現性、監査可能性が新たな課題になります。

## まとめ

Adaptive Ranking Modelから学べるのは、「巨大モデルをservingするにはGPUを増やせばよい」のではなく、重複を生む計算単位そのものを変える必要があるという点です。

ユーザーの重い処理を候補ごとではなくリクエストごとに1回実行します。Wukong Turboで高次interactionを拡張しつつ、安定性と通信コストを守る。

selective FP8、fusion、Grouped GEMMでハードウェアの実効利用率を上げる。疎な埋め込みをprune・共有・shardingし、single GPUのメモリ上限を越える。

model loadとautoscalingまで含めて本番システムとして設計します。


そして、LLM級という言葉だけでなく、何がdense computeで、何がsparse parameterかを見ることが重要です。Adaptive Ranking Modelの本質はパラメータ数の記録ではなく、リクエスト内で共有できる情報を見つけ、モデルとシステムを同じ境界で設計し直したことにあります。

## 参照

- Xi Chen et al., [Meta Adaptive Ranking Model: Bending the Inference Scaling Curve to Serve LLM-Scale Models for Ads](https://engineering.fb.com/2026/03/31/ml-applications/meta-adaptive-ranking-model-bending-the-inference-scaling-curve-to-serve-llm-scale-models-for-ads/), Engineering at Meta, 2026-03-31（2026-04-21 UTC更新）。
- Buyun Zhang et al., [Wukong: Towards a Scaling Law for Large-Scale Recommendation](https://arxiv.org/abs/2403.02545), arXiv:2403.02545v4, 2024-06-04（[HTML全文](https://arxiv.org/html/2403.02545)）。

本記事のAdaptive Ranking Modelに関する数値とproduction上の主張はMeta Engineeringの記事に基づきます。Wukongのarchitectureと評価条件は論文を参照しました。自社systemへの適用手順は、公開情報をもとにした記事側の提案です。
