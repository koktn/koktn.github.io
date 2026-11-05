---
title: Slice Finderで平均指標に隠れたモデルの失敗を見つける
description: 検証データから解釈可能で問題の大きい部分集合を自動探索するSlice Finderを、統計検定、探索戦略、実験結果、実装上の注意点まで解説します。
publishedAt: 2026-11-05
category: AI
tags:
  - 機械学習
  - モデル評価
  - データスライス
  - 論文解説
draft: false
---

> AI利用の明示<br>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文のv3を確認して記載しています。利用時は原文も確認してください。

今回取り上げるのは、Yeounoh Chungらの論文「[Automated Data Slicing for Model Validation: A Big data - AI Integration Approach](https://arxiv.org/abs/1807.06068)」です（[PDF](https://arxiv.org/pdf/1807.06068)、2019年1月7日改訂のv3）。この研究はモデル全体の平均指標では見えない失敗を、`sex = Male ∧ education = Bachelors`のように人が読めるデータ条件として自動発見するSlice Finderを提案しています。

本論文の短い版は2019 IEEE International Conference on Data Engineering（ICDE 2019）に「Slice Finder: Automated Data Slicing for Model Validation」として採録されています。この記事では手法と実験を詳しく収めたarXiv v3を主な出典にします。

## 課題：全体では良くても、特定の集団では悪い

モデルの平均accuracyや平均log lossが十分でも、特定の属性や属性の組み合わせに限ると性能が低いことがあります。論文では共通の特徴量条件を満たすデータの部分集合を「slice」と呼びます。

UCI Census IncomeデータにRandom Forestを適用した論文の例では、全3万件のlog lossは`0.35`です。しかし、`Sex = Male`の2万件では`0.41`、`Sex = Female`の1万件では`0.22`でした。学歴で分けると、`Bachelors`は`0.44`、`Masters`は`0.49`、`Doctorate`は`0.56`と悪化しています。全体の数値だけではこうした偏りを捉えられません。

かといって、考えられる条件をすべて列挙すればよいわけでもありません。特徴量と値の組み合わせに対して候補数は指数的に増えます。小さなsliceは損失が偶然高く見えやすく、多数の候補を検定すると偽陽性も増えます。クラスタリングで似た行をまとめても、そのクラスタを人が短い条件で説明できるとは限りません。

Slice Finderが探すのは、次の3条件を同時に満たすsliceです。

1. problematic：残りのデータよりモデルの損失が明確に高い
2. large：モデル全体の品質に影響し得るだけの件数がある
3. interpretable：少数の特徴量条件で説明できる

これは個々の予測を説明するLIMEやAnchors、モデル学習前のfeature selectionとは目的が違います。すでに学習したモデルを検証データ上で調べ、「どのデータ領域で失敗が集中するか」を見つける仕組みです。

## Slice Finderの全体像

Slice Finderは検証データ、正解ラベル、モデル予測から各行の損失を求め、Decision TreeまたはLattice Searchで候補sliceを探索します。各候補ではslice内とそれ以外の損失を比べ、効果量と統計的有意性を検査したうえで、条件が短く、件数と効果量が大きい上位`k`件を可視化します。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/slice-finder-pipeline-mobile.svg">
    <img src="/img/posts/slice-finder-pipeline.svg" alt="検証データから候補探索、効果量と統計検定、多重比較制御を経て上位スライスを返す流れ" loading="lazy">
  </picture>
  <figcaption>原論文の<a href="https://arxiv.org/pdf/1807.06068">Figure 1と§2〜3</a>を基に、処理の意味を本記事用に再構成。各候補は残りのデータと比較される。</figcaption>
</figure>

### sliceを条件式として表す

sliceは異なる特徴量に対する条件のANDで表します。

```text
country = DE ∧ gender = Male
age = [20, 30) ∧ education = Masters
```

数値特徴は分位点や等頻度binなどで区間へ離散化します。値をそのまま使うと、ほとんど行を含まないsliceが大量にできるためです。カテゴリ特徴の値が多すぎる場合、論文の実装は頻出上位`N`個を残し、それ以外を`other values`へまとめます。

条件数が少ないsliceほど解釈しやすいとみなし、次に件数が多いもの、効果量が大きいものを優先します。たとえば`country = DE`が十分に問題なら、そこへ年齢や郵便番号を追加した小さく複雑な子sliceは優先しません。

### 「差がある」と「差が大きい」を分ける

候補sliceを`S`、それ以外を`S′ = D − S`とします。論文は、`S`の平均損失が`S′`より高いかを片側のWelchのt検定で調べます。Welchのt検定を使うのは、`S`と`S′`で件数と損失の分散が異なる状況に適しているためです。

```text
H₀: ψ(S, h) ≤ ψ(S′, h)
Hₐ: ψ(S, h) > ψ(S′, h)
```

ここで`ψ(S, h)`はモデル`h`のslice `S`における平均損失です。ただし、サンプル数が多ければ小さな差でも統計的に有意になり得ます。そこで論文は標準偏差で正規化した効果量`φ`も使います。

```text
φ = √2 × (ψ(S, h) − ψ(S′, h)) / √(σ²S + σ²S′)
```

論文ではCohenの目安に従い、`φ = 0.2`を小、`0.5`を中、`0.8`を大、`1.3`を非常に大きい効果としています。ユーザーは効果量の下限`T`を決め、`φ ≥ T`かつ統計的に有意なsliceだけを残します。有意性は「差が存在しそうか」、効果量は「差がどれほど大きいか」を担い、片方だけではproblematicの判定として不十分です。

### 多重比較による偽陽性をα-investingで抑える

有意水準を`0.05`にして8候補を独立に検定するだけでも、少なくとも1個の偽陽性が出る確率は`1 − (1 − 0.05)^8 ≈ 0.34`です。候補が数百、数千へ増えるslice探索では、このmultiple comparisons problemを無視できません。

Slice Finderは検定回数を事前に固定しなくてよいα-investingを使い、検定に使える「α-wealth」を逐次更新します。効果量と件数が大きい候補を先に検定し、真の発見が早い段階に集まるという探索順を利用したBest-foot-forward policyです。論文が制御対象にするのは、期待偽発見数と期待発見数の比で定義するmarginal false discovery rate（mFDR）です。

Bonferroni補正やBenjamini–Hochberg法も比較されています。論文の実験ではα-investingは検出力を保ちやすい一方、設定によってFDRとのトレードオフがあります。したがって、α-investingを使えば誤検出がなくなるわけではありません。

## 2つの探索方法

Slice FinderはDecision Tree（DT）とLattice Search（LS）を補完的に用意しています。両者の違いは候補空間をどこまで探すかです。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/slice-finder-search-comparison-mobile.svg">
    <img src="/img/posts/slice-finder-search-comparison.svg" alt="Decision Treeが非重複領域へ分割し、Lattice Searchが重複する条件組み合わせも探索する違い" loading="lazy">
  </picture>
  <figcaption>原論文の<a href="https://arxiv.org/pdf/1807.06068">Figure 2と§3.1</a>を基に、仮想的な特徴A・B・Cで本記事用に再構成。LSは候補が広い分、計算量も大きい。</figcaption>
</figure>

### Decision Tree：速いが、重なるsliceを見落とし得る

DTは誤分類を分けるCARTの木を学習し、幅優先で葉を調べます。浅い段階で上位`k`件が見つかれば、探索する候補が少なく高速です。葉の経路をそのまま条件式として読める利点もあります。

一方、決定木はデータを重ならない領域へ分割します。`education = Masters`と`country = DE`がどちらも問題で、一部の行が両方に属するとしても、木の分岐順によって片方を見つけにくくなります。深い木では条件数が増え、解釈もしにくくなります。

### Lattice Search：重複も探せるが、候補数が増える

LSは、条件が1個のsliceから始め、条件を1個ずつ加えながら幅優先探索します。異なる親を持つ組み合わせも探索するため、重なるproblematic sliceを見つけられます。

各levelではまず効果量`T`を満たす候補を優先queueへ入れ、決められた順序で有意性を検定します。ある親sliceがすでにproblematicなら、その子は件数が減り条件も長くなるため、それ以上展開しません。反対に、親だけではproblematicでなかったsliceを細分化し、組み合わせで初めて現れる失敗を探します。

それでも高次元データでは候補数が指数的に増えます。論文は同じ条件数のsliceに対する効果量計算を複数workerへ分配する方法と、データをsamplingして探索する方法を示しています。

## 実験で何が分かったか

論文はLS、DT、k-means clustering（CL）を、合成データと2つの実データで比較しています。

| タスク | データとモデル | 評価上の注意 |
| --- | --- | --- |
| Census Income分類 | 15特徴・3万件、Random Forestで年収が5万ドルを超えるか予測 | 既存の未知な問題sliceに加え、選んだsliceのlabelを50%の確率で反転して既知の失敗を注入 |
| Credit Card Fraud検出 | 29特徴・28.4万取引中492件がfraud。非fraudをundersamplingし984件へ均衡化 | 特徴名の多くが匿名化され、見つかった条件の意味を解釈しにくい |
| 合成分類 | 離散化した2特徴、元は完全分類可能 | 単一条件または2条件のsliceへ50%のlabel反転を注入 |

### Lattice Searchは重なる問題を見つけやすい

合成データでは、推薦するslice数を増やしてもLSのaccuracyはDTを上回りました。論文のaccuracyは注入したproblematic sliceに属する行をどれだけ正しく覆うかについてprecisionとrecallを求め、その調和平均を取ったものです。LSは重複sliceを探索でき、DTは非重複の葉に制約されるため、この差が出ます。CLは効果量`T`以上のclusterだけを評価しても、LSとDTより低い結果でした。

Census Incomeでも同様の傾向ですが、評価は注入したsliceだけを正解とします。元データに最初から存在した本当のproblematic sliceを手法が見つけても誤りとして数えるため、論文自身がaccuracyを過小評価し得ると説明しています。

### 「似ている集団」と「失敗が集中する集団」は違う

CLは大きなclusterを作れても、平均効果量が`0`付近、場合によっては負になる結果でした。特徴量が似ていることは、モデルの失敗傾向が似ていることを保証しません。LSとDTは`T = 0.4`以上の効果量を持つsliceを直接探すため、よりproblematicな候補を返しました。

Census Incomeの上位例では、LSが`Marital Status = Married-civ-spouse`（14,065件、効果量`0.58`）、`Relationship = Husband`（12,463件、`0.52`）、`Relationship = Wife`（1,406件、`0.46`）などを発見しています。DTも最上位では同じ大きなsliceを見つけましたが、その後は4〜6条件の小さな葉が並びます。たとえば効果量`1.07`のsliceはわずか5件です。効果量だけを最大化すると、影響範囲の小さい候補へ寄ることが分かります。

### samplingで速くできるが、完全一致ではない

Census Incomeで全データ探索とsamplingを比較した実験では、データを`1/128`にしても、LSとDTはいずれもrelative accuracy `0.88`を維持し、実行時間はおよそ2桁短縮されました。ここでのrelative accuracyは、全データで見つけたsliceとの一致度です。絶対的な正解率ではありません。

並列化ではLSの効果量計算をworkerへ分配すると高速化しましたが、workerを増やすほど追加改善は小さくなりました。論文のDT実装は並列学習に未対応です。また、返す件数`k`が5未満ならDTが速く、その後はLSが速くなる領域がある一方、LSが次のlattice levelへ進む`k ≥ 70`付近では再びDTが相対的に速くなりました。どちらが速いかは`k`とproblematic sliceの密度に依存します。

## 公平性監査では「入口」であって結論ではない

論文はmodel fairnessを用途の一つに挙げ、sensitive attributeを事前指定せず、性能差の大きいsliceを発見できると説明します。これは監査候補を絞るうえで有用です。

ただし、Slice Finderが直接示すのは、選んだ損失関数においてsliceと残りのデータに差があることです。差の原因が差別なのか、データ収集の偏り、ラベル品質、サンプル不足、model specificationの問題なのかは追加調査が必要です。公平性の定義も一つではなく、論文はequalized oddsとの関係を例示しているにすぎません。発見されたsliceをそのまま「不公平な集団」と断定してはいけません。

さらに、条件へ使う特徴量が人にとって意味を持たなければ、条件式が短くてもactionableとは限りません。Credit Card Fraudデータの`V14`のような匿名特徴がその例です。解釈可能性は条件数だけでなく、特徴量の意味と業務上の対応可能性にも左右されます。

## 実装するときの最小構成

著者の[公開repository](https://github.com/yeounoh/slicefinder)にはPython実装、demo notebook、testがあります。ただし、release、依存version、再現用environmentは明示されていません。現在の環境でそのまま動くことを前提にせず、論文のアルゴリズムを確認する参照実装として扱うのが安全です。

独自実装なら、まず次のschemaを用意します。

```ts
type ValidationRow = {
  features: Record<string, string | number>;
  label: number;
  prediction: number;
  loss: number;
};

type SliceResult = {
  predicates: Array<{ feature: string; op: string; value: unknown }>;
  size: number;
  meanLoss: number;
  counterpartMeanLoss: number;
  effectSize: number;
  pValue: number;
};
```

Lattice Searchの最小処理は次のように整理できます。これは論文のAlgorithm 1を読みやすくした疑似codeであり、そのまま実行できる実装ではありません。

```text
input: 検証行、上限k、効果量threshold T、有意水準α
候補 = 1条件の全slice
発見済み = []

while 候補がある:
  1. 各候補と「それ以外」の損失分布から効果量を計算
  2. 効果量がT未満の候補は、次levelへ展開する親として保持
  3. T以上の候補を、条件数が少ない → 件数が多い → 効果量が大きい順に並べる
  4. α-investingの現在wealthでWelchのt検定を実行
  5. 有意なら結果へ追加し、その子sliceは展開しない
  6. 非有意な親へ条件を1個追加し、次levelの候補を生成
  7. k件見つかったら終了
```

### 導入時に追加したい安全策

論文の定義だけでは、本番の監視設計として足りない部分があります。実装時は次を追加すると扱いやすくなります。

- 検証データは学習データと分け、時間、地域、model versionなどの比較条件を記録する
- `T`、`k`、binning、最小件数、最大条件数を設定としてversion管理する
- 平均log lossだけでなく、タスクに応じてrecall、false positive rate、calibration errorなどを選べるようにする
- sliceの条件、件数、損失差、効果量、補正後の判定を一緒に保存する
- 同じデータでthresholdを繰り返し調整した場合、その探索自体が追加の選択バイアスになることを監査ログへ残す
- samplingで候補を探した後、独立した全量データまたはholdoutで再検証する
- model更新前後を比べる場合は、各行の「新modelの損失 − 旧modelの損失」をscoreにして、悪化sliceを探す

発見後の対応も自動化しすぎない方が安全です。sliceに該当するデータを増やす、ラベルを点検する、featureやmodelを変える、対象範囲を制限するといった選択肢は原因によって変わります。再学習後は対象sliceの改善だけでなく、全体と他sliceのregressionも再評価します。

## 限界と、今読んでも重要な点

原論文には次の限界があります。

まず、数値特徴の離散化で結果が変わります。binの境界が悪ければ、実際の失敗領域を分断したり、別の領域を混ぜたりします。著者もより良い離散化を今後の課題に挙げています。

次に、LSの探索空間は依然として指数的です。top-down pruning、sampling、並列化で軽減しても、高次元・高cardinalityのデータへ無条件に拡張できるわけではありません。DTは速さと引き換えに、重複sliceの網羅性を失います。

実験のproblematic sliceは主にlabel反転で注入され、現実の失敗原因をすべて再現するものではありません。§5.2〜5.6では説明を単純にするため全sliceを統計的に有意と仮定し、false discovery controlは§5.7で別に評価しています。探索、検定、ユーザー操作を含むsystem全体の有効性をproductionで検証した結果ではありません。著者も将来課題としてproduction ML platformへの導入とuser studyを挙げています。

それでも、平均指標を細かく切るだけで終わらず、解釈可能性、影響範囲、効果量、統計的有意性、多重比較を一つの探索問題として扱った点は重要です。model monitoringで大量のdashboardを並べる前に、どのsliceへ人の注意を向けるべきかという問題は現在のML運用にも残っています。

実務上の要点は最も損失の高い小集団を探すことではありません。人が理解でき、十分な件数があり、残りのデータとの差が偶然だけでは説明しにくい領域を見つけ、その後の原因調査につなげることです。Slice Finderはmodel validationを1個の平均値から、actionableな失敗条件の探索へ進める考え方を示しています。

## 参照

- Yeounoh Chung, Tim Kraska, Neoklis Polyzotis, Ki Hyun Tae, Steven Euijong Whang, [Automated Data Slicing for Model Validation: A Big data - AI Integration Approach](https://arxiv.org/abs/1807.06068), arXiv:1807.06068v3, 2019.
- Yeounoh Chung et al., [Slice Finder: Automated Data Slicing for Model Validation](https://ieeexplore.ieee.org/document/8731353), ICDE 2019.
- Yeounoh Chung, [slicefinder](https://github.com/yeounoh/slicefinder), public implementation.

本記事の手法、数値、実験条件は原論文を、公開実装の有無と内容は著者repositoryを確認して記載しました。
