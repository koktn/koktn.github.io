---
title: 低カバレッジA/Bテストの効果をどう戻すか：Trigger分析と希釈推定
description: 低カバレッジ機能のA/Bテストで、trigger分析の効果を全体効果へ戻すときに比率指標の単純な希釈式が壊れる理由と、分散削減として統一する方法を解説します。
publishedAt: 2026-11-04
category: Data
tags:
  - AB Testing
  - Causal Inference
  - Variance Reduction
  - Experimentation
  - 論文解説
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数式、数値、主張は原論文を確認して記載しています。利用時は原文も確認してください。

この論文の価値は低カバレッジ機能の効果推定を「trigger分析の結果をあとから薄める計算」ではなく、「trigger情報を使って全体効果の推定分散を減らす問題」と捉え直した点にあります。

取り上げるのは、Alex DengとVictor Huによる「[Diluted Treatment Effect Estimation for Trigger Analysis in Online Controlled Experiments](https://doi.org/10.1145/2684822.2685307)」です。WSDM 2015で発表された査読済み論文で、MicrosoftとBingのオンライン実験を題材にしています。本稿では[論文PDF](https://exp-platform.com/Documents/wsdm2015-dilution.pdf)をもとに、trigger分析の二つの方式、単純な希釈式が比率指標で誤る理由、分散削減による統一手法、実験結果、実装時の前提を順に整理します。

## 低カバレッジ機能は全体分析で効果が見えにくい

A/Bテストではランダムに分けたtreatment群とcontrol群の指標差から平均処置効果を推定します。しかし、すべてのユーザーやセッションが変更箇所へ触れるとは限りません。天気検索だけに出る機能や、綴り間違いを検出したときだけ表示する修正候補は、トラフィックの一部でしか発火しません。

このような機能を全ユーザーで比較すると、影響を受けようがない観測が大量に混ざります。効果の平均は小さくなり、標準誤差に対して差が見えにくくなります。論文では、Bingの低カバレッジ機能で全体効果が典型的に`0.1%`未満になると説明しています。また、feature coverageが`20%`未満の場合、チームにtrigger分析も求めていたと記載しています。これは論文執筆時のMicrosoftにおける運用例であり、一般的なしきい値ではありません。

trigger分析は機能の影響を受ける可能性がある部分集合に絞って比較する方法です。全体分析より効果を濃く観測できるため、帰無仮説を棄却できるかという検定では有利になります。

ただし、部分集合で測った差は、そのままでは全ユーザーへの平均効果ではありません。ROIの比較やチーム間の貢献評価では、カバレッジの異なる機能を同じ母集団で比べる必要があります。そこで、triggered subsetで推定した効果を全体効果へ変換する「dilution（希釈）」が必要になります。

## User-triggerとSession-triggerは残す範囲が違う

論文はtrigger分析を二種類に分けます。違いは最初のtrigger後にどのセッションを分析へ残すかです。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/trigger-analysis-scope-mobile.svg">
    <img src="/img/posts/trigger-analysis-scope.svg" alt="全体分析、User-trigger分析、Session-trigger分析が同じユーザーのどのセッションを残すかを比較した図" loading="lazy">
  </picture>
  <figcaption>図1：<a href="https://exp-platform.com/Documents/wsdm2015-dilution.pdf">原論文Figure 2・§2</a>の定義をもとに、本記事用の仮想例で再構成した独自図。色付きのTはtriggerを含むセッション、白いUはtriggerを含まないセッションです。矢印は時間の向きを示します。</figcaption>
</figure>

**User-trigger分析**は、ユーザーが初めてtriggerした時点より前を除き、その後のセッションをすべて残します。機能が次回以降の訪問やエンゲージメントにも影響し得る場合に向いています。sessions per userやqueries per userのようなユーザー単位の指標も解釈できます。

**Session-trigger分析**は、triggerを一度でも含むセッションだけを残します。影響がtriggerしたセッション内に収まるなら、User-triggerより狭い範囲へ絞れるため、さらに高い感度が期待できます。一方、機能がtrigger後の別セッションにも影響する場合、その効果を捨ててしまいます。

セッション内のtriggerイベントだけを抜き出さず、セッション全体を残す点も重要です。たとえばスペル修正候補をクリックしたあと、次のqueryで目的を達成することがあります。triggerしたimpressionだけを見ると、セッションが成功したかを定義できません。

Session-triggerを使うには、「triggerを含まないセッションには処置効果がない」という前提が必要です。論文は、session-trigger complement、つまり除外したセッションだけでtreatment群とcontrol群を比較し、効果の証拠がないかを確認する方法を提案しています。ただし、非有意であることは効果がない証明ではありません。

## 加法的な指標なら人数比で希釈できる

まず、sessions per userやqueries per userのような加法的な指標`X`を考えます。各ユーザーについて、全体値をtrigger部分とtrigger complementへ分けられる指標です。

```text
X = TrX + UnTrX
```

`N`を全ユーザー数、`N_Tr`をtriggered user数、`Δ_Tr(X)`をtrigger分析でのtreatment-control差とします。untriggered userにもtrigger complementにも処置効果がないなら、全体効果は次の式で推定できます。

```text
Δ_overall(X) = Δ_Tr(X) × N_Tr / N
```

triggered userが全体の5%で、triggered subsetでsessions per userが`+2`なら、全体では`+0.1`です。この計算が成立するのは、ユーザーごとの全体指標をtrigger部分とcomplementの和に分解できるからです。論文はRubinのpotential outcome frameworkを使い、User-triggerとSession-triggerの両方についてこの式を導いています。

## 比率指標では「平均してから掛ける」と誤る

問題はsession success rate per user（SSR）のような二重平均の比率指標です。各ユーザーについて`成功セッション数 / セッション数`を計算し、その値をユーザー間で平均します。click-through rate per userも同じ型です。

分子と分母はtrigger部分とcomplementへ分けられても、比率そのものは足し算へ分解できません。よく使われる近似式は、ユーザーのtrigger率に加え、比率の分母がtriggerした割合`TR`も掛けます。

```text
近似：Δ_overall ≈ Δ_Tr × N_Tr / N × 平均(TR)
```

この式は、triggered subsetの効果と`TR`が独立であるときに限って正当化できます。重いユーザーほど機能をtriggerしやすく、効果も大きいといった相関があれば、平均効果と平均trigger率を別々に計算して掛けることはできません。

論文のTime to Success per userの例を追うと、ずれがはっきりします。

| ユーザー | Trigger sessionでの短縮 | 全session数 | ユーザー全体での平均短縮 |
| --- | ---: | ---: | ---: |
| User 1 | `100 ms` | 2 | `50 ms` |
| User 2 | `20 ms` | 10 | `2 ms` |

ユーザーを等しく重み付けした正しい全体平均は`(50 + 2) / 2 = 26 ms`の短縮です。ところが、triggered sessionでの平均短縮`60 ms`へ平均trigger率`(1/2 + 1/10) / 2 = 0.3`を掛けると`18 ms`になります。効果が大きいUser 1ほどtrigger率も高いため、積の平均と平均の積が一致しません。

## 厳密式はユーザーごとに効果を希釈してから平均する

ユーザー`i`のtrigger率を`TR_i`、trigger部分における潜在的な処置効果を`TrX_i^T - TrX_i^C`とします。比率指標の分母自体は処置によって変わらず、trigger complementにも効果がないと仮定すると、厳密な全体効果は次の形になります。

```text
Δ_overall(X) = (1 / N) × Σ_i∈Tr [TR_i × (TrX_i^T - TrX_i^C)]
```

順序が重要です。各ユーザーのtrigger部分の効果へ、そのユーザー自身の`TR_i`を掛けてから全ユーザーで平均します。先に`Δ_Tr`と`TR`をそれぞれ平均し、その積を取る式ではありません。

実際のA/Bテストでは、同じユーザーのtreatment時とcontrol時を同時に観測できません。論文は次のderived metricをユーザー単位で作り、treatment群とcontrol群の平均との差を取れば、厳密式と同じ全体効果を推定できると示します。untriggered userでは`TR_i = 0`とします。

```text
Y_i = TR_i × TrX_i
Δ_overall = mean(Y | treatment) - mean(Y | control)
```

これは、trigger率ごとにユーザーを細かく層別化する実装より単純です。ただし`TR_i`のばらつきを掛け込むため、点推定は不偏でも分散が大きくなる場合があります。論文の高カバレッジ実験では、この厳密式をそのまま使うと全体分析より分散が増えました。

## 希釈を分散削減として一段で解く

論文の中心的な提案はtrigger分析と希釈を別々に行わず、全体効果の推定量をtrigger情報で直接調整する方法です。

指標`X`の素朴な推定量を`Δ(X)`とします。処置の影響を受けない共変量`Y`なら、ランダム化により期待値上の群差は0です。そのため、次の推定量は任意の`θ`について元の全体効果と同じ期待値を持ちます。

```text
Δ* = Δ(X) - θ × Δ(Y)
```

`X`と`Y`が相関していれば、適切な`θ`で群間差の偶然の揺れを差し引き、分散を減らせます。単一共変量で両群の分散・共分散が同じとみなせる場合、最適な調整後の分散は`(1 - ρ²) Var(Δ)`です。ここで`ρ`は`X`と`Y`の相関です。

trigger分析で除外していたcomplementは、「処置の影響を受けないが、ユーザーの基礎的な行動を反映し、全体指標と相関するデータ」です。捨てるべきノイズではなく、分散削減に使える共変量になります。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/unified-trigger-variance-reduction-mobile.svg">
    <img src="/img/posts/unified-trigger-variance-reduction.svg" alt="全体指標とtrigger complement、trigger率、100% triggerの指示変数から調整済みの全体効果を直接推定する流れ" loading="lazy">
  </picture>
  <figcaption>図2：<a href="https://exp-platform.com/Documents/wsdm2015-dilution.pdf">原論文§4.2〜4.4のModel (6)</a>を基に本記事用に再構成した独自図。共変量の群差を回帰係数で調整し、全体指標のtreatment-control差を保ったまま分散を減らします。</figcaption>
</figure>

一般の指標へ使えるModel (6)は、全体指標`X`を次の共変量で調整します。

```text
X ~ UnTrX + TR + I(TR = 1)
```

- `UnTrX`：trigger complementだけで計算した同じ指標
- `TR`：ユーザー単位のtrigger率
- `I(TR = 1)`：complementが一つもないユーザーを表す指示変数

`TR = 1`のユーザーは`UnTrX`を定義できないため、値を0などへ置き、欠損の意味を指示変数で分けます。論文の`~`は確率モデルの完全な形を指定する記号ではなく、右辺を分散削減の共変量に使うという表記です。

比率指標では、先ほどのderived metric`Y = TR × TrX`を左辺へ置き、同じ共変量でさらに分散を減らすModel (7)も使えます。

```text
Y := TR × TrX ~ UnTrX + TR + I(TR = 1)
```

Model (7)は比率指標に対する厳密式に基づくため、「比率の分母に処置効果がない」という追加前提が必要です。Model (6)は指標の形を問わず、実装もしやすい一方、Model (7)は前提が成立する比率指標でより大きな分散削減を得られる可能性があります。

## 2,800万〜4,000万ユーザーの3実験で比較した

評価にはBingのA/Bテストを3件使っています。各実験には2,800万〜4,000万ユーザーと数百万件以上のquery・click情報が含まれます。指標はユーザー単位のSSRです。論文は同程度のtrigger率を持つ別の3実験でも似た結果を得たこと、さらに数十件の実験を分析したことを述べていますが、表で個別結果が公開されているのはExpA〜Cです。

| 実験 | 変更内容 | Triggered user | Triggered session |
| --- | --- | ---: | ---: |
| ExpA | Related Search blockの位置変更 | `5.26%` | `1.27%` |
| ExpB | task paneのvideo answer | `33.46%` | `20.83%` |
| ExpC | 検索結果の動的なサイズ変更 | `65.17%` | `60.35%` |

評価指標はvariance reduction rateです。`95%`なら、調整後の推定分散が全体分析の`5%`になったことを表します。論文は、`x%`の分散削減を同じ検出力へ到達するためのトラフィック・サンプルサイズを`x%`減らせることと対応づけています。この対応は同じ実験設計と分散構造を前提とします。

### Model (6)は全実験で分散を減らした

汎用のModel (6)について、User-trigger由来とSession-trigger由来の共変量を使ったときの残存分散は次の通りです。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/trigger-model6-remaining-variance-mobile.svg">
    <img src="/img/posts/trigger-model6-remaining-variance.svg" alt="ExpA、ExpB、ExpCでModel 6を使ったときのUser-triggerとSession-triggerの残存分散を全体分析比で比較した棒グラフ" loading="lazy">
  </picture>
  <figcaption>図3：<a href="https://exp-platform.com/Documents/wsdm2015-dilution.pdf">原論文Table 5</a>を本記事でchart化。縦軸は全体分析の分散を100%とした残存分散で、低いほど推定精度が高いことを示します。</figcaption>
</figure>

| 実験 | User-trigger共変量 | Session-trigger共変量 | Session-triggerの分散削減率 |
| --- | ---: | ---: | ---: |
| ExpA | `4.40%` | `1.75%` | `98.25%` |
| ExpB | `21.43%` | `14.01%` | `85.99%` |
| ExpC | `63.97%` | `46.03%` | `53.97%` |

3件すべてでSession-trigger由来の共変量がUser-triggerより低い分散になりました。ExpAではSession-triggerの残存分散`1.75%`が、User-triggerの`4.40%`の約40%です。Session-triggerは影響範囲がセッション内に収まるという強い前提を置く代わりに、より狭いcomplementを使って感度を高めています。

### 厳密式だけでは分散が増える場合がある

User-trigger分析で厳密式`Y = TR × TrSSR`の群差をそのまま使った場合、variance reduction rateはExpAで`88.60%`でしたが、ExpBで`-17.14%`、ExpCで`-49.47%`でした。負の値は全体分析より分散が増えたことを表します。

同じ`Y`をModel (7)で調整すると、User-triggerでは`98.42%`、`84.80%`、`61.45%`、Session-triggerでは`99.44%`、`89.85%`、`69.10%`の分散削減となり、3件すべてで比較した方式中の最大値でした。`TR`を掛けることで生じるばらつきを、`TR`自体も共変量に含めて取り除いた結果です。

この結果は「比率指標なら常にModel (7)がよい」という保証ではありません。3件のBing実験における結果であり、Model (7)には分母不変の前提があります。論文も一般の指標には実装しやすいModel (6)を推奨し、比率指標ではModel (7)を強い代替案と位置づけています。

### 検定だけならtrigger分析が勝つ場合もある

Session-triggerにおけるt-statisticは、Model (6)と通常のtrigger分析で次の値でした。

| 実験 | Model (6) | Trigger分析 |
| --- | ---: | ---: |
| ExpA | `6.72` | `3.32` |
| ExpB | `3.97` | `5.26` |
| ExpC | `3.50` | `3.39` |

3件とも両方式で統計的に有意でしたが、ExpAではModel (6)、ExpBではtrigger分析のt-statisticが高くなりました。分散削減方式の主な利点はtrigger分析より必ず高いt-statisticを出すことではありません。triggered subsetの効果ではなく、全体効果を直接推定できる点です。

## 実装するときの最小手順

以下は論文のModel (6)を、ユーザー単位の集計処理へ落とすための疑似コードです。論文は特定のライブラリや公開実装を提供していないため、実行可能な再現コードではありません。

```text
for each user i:
  X_i       = 全期間の指標
  TR_i      = triggered denominator / total denominator
  UnTrX_i   = trigger complementだけの指標
  IsTR1_i   = 1 if TR_i == 1 else 0

  if TR_i == 1:
    UnTrX_i = 0

control群で X ~ UnTrX + TR + IsTR1 を回帰し、係数 theta を推定

各ユーザーについて:
  X_adj_i = X_i - theta_1 * UnTrX_i
                  - theta_2 * TR_i
                  - theta_3 * IsTR1_i

overall_effect = mean(X_adj | treatment) - mean(X_adj | control)
standard_error = treatment群とcontrol群の標本分散から計算
```

論文は`θ`をcontrol群だけ、treatment群だけ、または両方から推定できるとしています。処置効果が大きくなければ差は小さく、最適値を正確に当てなくても近い係数で分散削減の多くを得られるという説明です。ただし、同じデータから係数を推定する有限標本での扱い、ロバスト標準誤差、cluster、欠損、winsorizationなどの実装詳細は論文にまとまっていません。

ここからは論文で実証された結果ではなく、公開された手法を運用へ移す際の本記事側の提案です。導入時は少なくとも次の四つを別々に確認する必要があります。

1. Counterfactual triggerを記録し、control群でもtreatmentだったならtriggerしたかを判定できるようにします。treatment群だけの事後的な条件で絞ると、ランダム化を壊す可能性があります。
2. Session-triggerを使うなら、session-trigger complementの群差から効果範囲を検証します。非有意でも「効果ゼロが証明された」とは扱いません。
3. Model (7)では、session数やpage-view数など比率の分母に処置効果がないかを検定します。分母が動くなら、その比率だけでproduct impactを評価すること自体を見直します。
4. 点推定が全体分析と整合するか、標準誤差とconfidence intervalが期待どおり狭くなるかをA/Aテストと既知のA/Bテストでbacktestします。

## 論文からそのまま一般化できない点

実験は検索サービスの3件を中心とする社内データです。数千万ユーザー規模ですが、生データ、コード、詳細なmetric定義、実験期間、群ごとの標本数は公開されていません。第三者が表の数値を完全再現することはできません。

主結果のvariance reduction rateは大きいものの、選ばれた実験は著者らが高品質にlabelされたpositiveまたはnegativeな実験と説明する事例です。すべての実験における分布や、near-zero effectでの性能は分かりません。

Session-trigger complementのp-valueはExpA `0.23`、ExpB `0.38`、ExpC `0.55`で、有意な差は観測されませんでした。しかし、論文の脚注も指摘するとおり、統計検定にはfalse negativeがあり、影響がないことは証明できません。長期効果、spillover、network effectがあり得る機能では、Session-triggerの前提が特に危うくなります。

論文の分散削減は大標本で中心極限定理を使える状況を想定しています。Appendixの8ユーザー例は計算手順を示すためのもので、小標本での妥当性を示す実験ではありません。

また、論文は2015年の研究です。現在の実験基盤へ導入する際は、metricの比率推定、sequential testing、multiple testing、clustered randomization、sample ratio mismatch、ログ品質など、この論文が扱っていない論点も別途検討する必要があります。

## まとめ

低カバレッジ機能では、影響を受け得る部分集合へ絞るtrigger分析が検定感度を高めます。しかし、そこで得た効果は全体母集団への平均効果ではありません。

sessions per userのような加法的指標なら、triggered user比を掛ける単純な希釈式が成り立ちます。SSRやCTR per userのような比率指標では、ユーザーごとのtrigger率と効果が相関し得るため、平均効果と平均trigger率の積では誤ります。厳密には、ユーザーごとに`TR_i × TrX_i`を作ってから群差を取ります。

論文の重要な転換は、その厳密式を個別に実装するだけで終わらないことです。全体指標を目的変数とし、trigger complement、trigger率、complement欠損の指示変数を共変量に使えば、指標の形を問わず全体効果を直接推定しながら分散を減らせます。

3件のBing実験では、汎用のModel (6)が全条件で分散を削減し、Session-trigger由来の共変量を使った残存分散は全体分析の`1.75%`、`14.01%`、`46.03%`でした。一方、Session-triggerには影響がセッション内に収まるという前提があります。感度を上げるために観測範囲を狭めるほど、除外部分に効果がないという仮定の検証が重要になります。

trigger情報は分析対象を切り捨てるためだけのfilterではありません。全体効果を保ったまま推定精度を上げる共変量として使えます。この見方がカバレッジの異なる機能を同じ尺度で比較するための要点です。

## 参照資料

- Alex Deng and Victor Hu, [Diluted Treatment Effect Estimation for Trigger Analysis in Online Controlled Experiments](https://doi.org/10.1145/2684822.2685307), WSDM 2015, DOI: `10.1145/2684822.2685307`.
- [論文PDF](https://exp-platform.com/Documents/wsdm2015-dilution.pdf) — trigger分析は§2、厳密な希釈式は§3、分散削減による統一は§4、実験結果は§5、計算例はAppendix。
