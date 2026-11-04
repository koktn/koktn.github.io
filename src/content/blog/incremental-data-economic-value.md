---
title: 追加データはいくらまで買えるか：予測性能を経済価値へ変換する方法
description: 追加データによるprecision、recall、AUCの変化を金額へ換算する方法を、広告と寄付募集の事例、数式、導入手順、限界から解説します。
publishedAt: 2026-11-04
category: Data Science
tags:
  - Machine Learning
  - Data Valuation
  - Predictive Modeling
  - ROI
  - 論文解説
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値と数式は原論文を確認して記載しています。利用時は原文も確認してください。

この論文の価値は追加データの価格を「レコード数」や「AUCが何ポイント上がったか」だけで決めず、**その改善が意思決定を変えて生む増分利益**から逆算した点にあります。

取り上げるのは、Brian Dalessandro、Claudia Perlich、Troy Raederによる「[Bigger is Better, but at What Cost? Estimating the Economic Value of Incremental Data Assets](https://journals.sagepub.com/doi/10.1089/big.2014.0010)」です（[PDF](https://journals.sagepub.com/doi/pdf/10.1089/big.2014.0010)）。2014年6月に学術誌 *Big Data*、Volume 2、Issue 2、pp. 87–96へ掲載されたOriginal Articleで、CC BY 3.0 USの下で公開されています。

論文が対象とするのは、連続スコアを返す二値分類です。追加データを入れたモデルと入れないモデルを同じholdout dataで比較し、precision、recall、AUCの差をtrue positiveやfalse positiveに伴う利益・費用へ結び付けます。広告配信と寄付募集の2事例では、外部データが単独では役立っても、有用な自社データをすでに持つ場合、その増分価値が大きく下がりました。

## モデル精度だけでは購入価格を決められない

外部データを買う判断で知りたいのは、「モデルが良くなるか」だけではありません。「いくらまで払えば投資として成り立つか」です。

たとえば、AUCが`0.01`上がる2種類のデータがあったとします。一方は1件の見逃しが数百万円の損失につながる不正検知に使われ、もう一方は単価の低い広告配信に使われます。同じAUC差でも、1件の判定が持つ経済的な重みは異なります。

反対に、利益が大きい業務でも、追加データが現在のモデルとほぼ同じ情報しか持たなければ、購入による改善は小さくなります。したがって、データの価値はデータ単体の性質では決まりません。少なくとも、現在のbaseline、行動方針、予測性能の増分、判定結果ごとの損益を組み合わせる必要があります。

## 予測から金額までの流れ

論文の方法は追加データを加えたときの反実仮想を、2つのモデルのout-of-sample評価で近似します。

<figure class="article-figure">
  <img src="/img/posts/incremental-data-value-flow.svg" alt="現在の特徴量だけのモデルと追加データを含むモデルを同じholdout dataで比較し、指標差と損益から追加データの経済価値を求める流れ" loading="lazy">
  <figcaption>図1：<a href="https://journals.sagepub.com/doi/10.1089/big.2014.0010">原論文§Estimating the Value of DataとFigure 1</a>を基に、本記事で処理順を整理した独自図。矢印は評価結果から次の計算へ進む順序を示します。</figcaption>
</figure>

まず、データを次のように分けます。

```text
D = [X_BL, X_Inc, Y]
```

`X_BL`は現在使えるbaseline features、`X_Inc`は購入を検討する追加features、`Y`は正解labelです。`X_BL`だけで学習したモデルと、`X_BL + X_Inc`で学習したモデルを用意します。

次に、同じout-of-sample dataで評価指標`m`を計算します。追加データによる指標の変化は次の差です。

```text
Δm = m(D_Inc, scores_Inc, predictions_Inc)
   - m(D_BL,  scores_BL,  predictions_BL)
```

最後に、TP、FP、FN、TNへ利益または費用を割り当てたcost–confusion matrixを使い、`Δm`を1件あたりの期待価値の変化`ΔE[V_i]`へ変換します。全対象へデータを適用するなら、1件あたりの値に対象件数を掛けることで、購入前に見込める価値の上限を計算できます。

この比較には、追加データを最終購入前に試用できるという前提があります。また、学習と評価の双方で同じ種類のfeaturesを利用でき、評価値は学習に使っていないdataから得る必要があります。

## 行動方針に合わせて指標を選ぶ

precision、recall、AUCは交換可能な指標ではありません。論文はモデルをどのように行動へ変えるかによって指標を選び分けています。

| 業務上の行動 | 適した指標 | 経済的な意味 |
| --- | --- | --- |
| 予算が固定され、上位`T`件だけへ行動する | precision@`k`またはlift@`k` | 固定費用の範囲で成果件数を増やす |
| 期待利益が基準を超える対象へ行動する | AUCまたはlog-likelihood | 費用または利益を基準にthresholdを選ぶ |
| 所定数のtrue positiveを獲得する | recall@`k` | 目標件数を得る費用を下げる |
| 予算やthresholdが事前に決まらない | AUCまたはpartial AUC | 複数thresholdでの平均的な順位性能を扱う |

広告を100万人へ出せる場合と、期待利益が正の利用者全員へ出せる場合では、同じranking modelでも使う範囲が異なります。固定された上位層だけを対象にするなら、その層でのprecisionが価値へ直結します。購入後にどのthresholdを使うか分からず、全件分のデータを一括購入する場合は、平均的なranking能力を表すAUCを使います。

## precision、recall、AUCを金額へ変える

論文の記号ではtrue positiveから得る価値を`V_TP`、true positiveへ行動する費用を`C_TP`、false positiveの費用を`C_FP`、false negativeの費用を`C_FN`とします。各結果の価値や費用は一定で、既知であると仮定します。

### precisionが変わる場合

threshold `k`より上の対象へ行動するとき、precisionの差`ΔPRE_k`による1件あたりの期待価値の変化は次の式です。

```text
ΔE_k[V_i] = ΔPRE_k × (V_TP - C_TP + C_FP)
```

precisionが上がると、行動対象のうちTPが増え、FPが減ります。そのため、TPの純価値`V_TP - C_TP`だけでなく、避けられたFP費用`C_FP`も改善分へ入ります。

### recallが変わる場合

false positive rateを固定した場合、recall、つまりtrue positive rateの差`ΔTPR_k`は次の式で金額へ変わります。

```text
ΔE_k[V_i]
  = p(Y) × ΔTPR_k × (V_TP - C_TP + C_FN)
```

`p(Y)`はpositive classの比率です。TPを増やすことで得る純価値に加え、FNを避けた価値が含まれます。FPRが固定されない場合は、`ΔFPR_k × C_FP`も考慮しなければならず、recallの差だけでは金額を決められません。

### AUCが変わる場合

thresholdが未確定で、論文の仮定に従ってthreshold全体の期待値を扱うと、AUC差は次の式になります。

```text
ΔE[E_k[V_i]]
  = (V_TP - C_TP + C_FN) × p(Y) × ΔAUC
```

実際の運用で使うthresholdの範囲を絞れるなら、全域のAUCではなくpartial AUCを使えます。

この式を使うときに注意したいのは、AUCの差がそのまま利益ではないことです。class prevalenceと1件あたりの価値を掛けて初めて金額になります。また、運用thresholdが固定されているのにAUCだけで値付けすると、実際には使わないscore領域の改善まで評価へ含める可能性があります。

## 広告事例：単独では価値があっても追加価値は消える

1つ目の事例はDstilleryのdisplay advertisingです。4種類のthird-party audience segmentを10 campaignへ加え、上位1%をtargetingしたときのprecisionを比較しました。conversion 1件の価値はcampaignによって異なり、cruise lineの`$200`からrestaurantの`$1.50`まで幅があります。図表の価値はtargetingした1,000人あたりで示されています。

比較は2つのbaselineで行われました。最初はrandom targetingです。この条件では、segmentごとの価値がcampaignによって大きく異なり、自動車部品へ関心があるsegmentは自動車用品のcampaignだけで価値を持つといった違いが出ました。

次に、自社がすでに持つweb usage dataをbaselineにします。L2正則化logistic regressionで、自社segmentだけのモデルと、自社segmentに購入候補のsegmentを加えたモデルをout-of-sample dataで比較しました。その結果、対象となったcampaignとsegmentの大半で追加価値がゼロ近くまで下がりました。

ここから分かるのは、vendorが示す「このdataだけでどれだけ予測できるか」では購入判断に足りないということです。買い手がすでに持つdataとの重複を差し引き、現在のproduction baselineに対する増分で評価する必要があります。

## 寄付募集事例：`$3.69`が`$0.41`へ下がる

2つ目の事例は1998年KDD Cupで公開されたParalyzed Veterans of America（PVA）のdirect mail dataです。対象は最後の寄付から13〜24か月が経過した約19万1,000人のdonorです。寄付依頼の郵便を送る相手をrankingするため、L2正則化logistic regressionを学習しました。

この事例では購入後に一部のrecordだけを選べず、全件分のdataを一括購入する想定です。そのため、特定thresholdのprecisionではなくAUCを使います。TPの価値には平均寄付額`$15`を置いています。

| Data used | AUC | baselineからの`ΔAUC` | 1,000 recordsあたりの価値 |
| --- | ---: | ---: | ---: |
| Base | `0.5361` | — | — |
| Base + Census | `0.5402` | `+0.0041` | `$3.04` |
| Base + Third 1 | `0.5343` | `-0.0018` | `$0.00` |
| Base + Third 2 | `0.5411` | `+0.0049` | `$3.69` |
| Base + Historical | `0.6036` | `+0.0674` | `$50.58` |
| Base + Historical + Census | `0.5958` | `-0.0078` | `$0.00` |
| Base + Historical + Third 1 | `0.6000` | `-0.0036` | `$0.00` |
| Base + Historical + Third 2 | `0.6041` | `+0.0005` | `$0.41` |

<small>原論文Table 1。`Base`は約30個の世帯・人口統計features、`Historical`は約10個の過去10年間の寄付履歴featuresです。`Census`は約300個の地域集計featuresで、`Third 1`と`Third 2`は外部dataです。</small>

`Base`だけを基準にすると、`Third 2`の価値は1,000 recordsあたり`$3.69`でした。しかし、PVAがすでに持つ`Historical`をbaselineへ加えると、`Third 2`の価値は`$0.41`へ下がります。`Census`と`Third 1`はAUCを悪化させ、表では価値が`$0.00`とされています。

反対に、自社の寄付履歴は`ΔAUC = 0.0674`、1,000 recordsあたり`$50.58`でした。論文は購入候補の外部dataより10倍を超える価値だと説明しています。data sourceの列数ではなく、現在の情報へ何を追加できるかが価値を決めた例です。

## 実務で評価する手順

論文の方法を実務へ移す場合は、次の順で進めると、model metricと購入判断を混同しにくくなります。ここからは、論文の枠組みをproductionで検証するための実装案です。

### 1. 行動と評価単位を先に固定する

誰に何を行い、TP、FP、FN、TNが何を意味するかを定義します。1件あたり、1,000 recordsあたり、月間全件など、価格と比較する単位もそろえます。

固定予算で上位1%へactionするならprecision@1%が候補です。必要なTP数が決まっているならrecall、thresholdが変動するならAUCまたはpartial AUCを検討します。慣例だけでAUCを選ばず、action policyと一致する指標を使います。

### 2. 現在のproduction dataをbaselineにする

「追加dataだけ」と「dataなし」の比較は、vendorのdataが単独で予測に使えるかを示します。しかし、購入判断で必要なのは、現在のfeaturesとmodelに加えたときの差です。

baselineには実際に利用できる自社data、現在の前処理、productionで採用するmodel familyを含めます。追加data側だけ高性能なalgorithmを使うと、dataの価値とmodel改善の価値が混ざります。

### 3. 同じholdout dataで比較する

baseline modelとincremental modelは、同じsplit、同じ評価期間、同じ対象で比較します。将来の購入判断なら、random splitだけでなくtime-based holdoutも用意し、追加dataが時点を越えて漏れていないか確認します。

差分にはsampling errorがあるため、point estimateだけでなくbootstrap confidence intervalも見ると判断しやすくなります。これは本記事の実装上の提案であり、原論文の2事例ではconfidence intervalは報告されていません。

### 4. cost–confusion matrixを事業側と作る

TPの売上だけでなく、action自体の費用、FPの接触費用や顧客体験への影響、FNで失う利益を見積もります。値が不確かな場合は1つに決めず、悲観・中央・楽観のscenarioで感度分析します。

```text
scenario = {
  conservative: { V_TP, C_TP, C_FP, C_FN },
  expected:     { V_TP, C_TP, C_FP, C_FN },
  optimistic:   { V_TP, C_TP, C_FP, C_FN }
}
```

購入価格の上限は期待価値の増分そのものとは限りません。data fee以外に、join keyの整備、欠損処理、pipeline、storage、推論、monitoring、契約・privacy reviewの費用がかかるためです。目標ROIを確保するなら、これらを増分価値から差し引いて交渉可能な価格帯を決めます。

### 5. 導入後も増分を測る

trial dataと本契約dataでcoverage、鮮度、欠損率が違えば、購入前の評価は再現しません。data version、取得日時、coverage、join率を記録し、model performanceと経済指標を継続して監視します。

可能ならholdout groupを残し、追加dataを利用したactionの因果効果をonline experimentで確かめます。論文の計算はout-of-sampleの予測性能差を経済価値へ換算する方法です。dataを使うことで実際の行動結果がどれだけ変わったかを、randomized experimentで直接推定した結果とは区別する必要があります。

## この方法の限界

第一に、cost–confusion matrixの値を正しく置くことは簡単ではありません。論文もこの点を認めた上で、値は既知と仮定しています。長期的な顧客価値、苦情、ブランド毀損、機会費用を1件あたりの定数で表せない業務では、計算結果の幅が大きくなります。

第二に、価値はmodelerの選択に依存します。algorithm、hyperparameter、前処理、holdout設計が違えば`Δm`も変わります。追加dataをうまく使えないモデルで低い価値が出ても、data自体に価値がないとは限りません。反対に、leakageがあれば価値を過大評価します。

第三に、論文の対象は連続scoreを返す二値分類です。multi-class、regression、ranking、生成AIへ式をそのまま適用することはできません。利益・費用が対象ごとに違う場合や、actionが他の対象へ影響する場合も、定数のcost matrixでは足りません。

第四に、2つのcase studyはmarketingです。PVA dataは1998年のcompetition dataで、現在のprivacy、consent、data governance、規制対応を評価した研究ではありません。予測上の価値が正でも、取得・利用が適法で妥当だという結論にはなりません。

第五に、価値は時間とともに変わります。自社dataが増えれば外部dataの重複が増え、vendor dataの鮮度やcoverageが落ちれば増分も下がります。一度算定した価格を契約期間中の固定的な価値とみなさず、更新時に同じbaselineから再評価する必要があります。

## まとめ

この論文は追加dataの価値を、baseline modelに対するout-of-sampleの性能差と、判定結果ごとの利益・費用から計算します。dataは単独で評価せず、現在の自社dataへ加えたときの増分を見ることが重要です。

広告事例ではthird-party segmentの多くが自社dataをbaselineにすると価値をほぼ失いました。寄付募集事例でも、外部data `Third 2`の価値は1,000 recordsあたり`$3.69`から`$0.41`へ下がりました。既存dataと重複する情報へ、単独性能を根拠に高い価格を払うべきではないことを示しています。

一方、算出額はdataの普遍的な値段ではありません。action policy、class prevalence、cost matrix、model、評価期間、既存featuresによって変わる、買い手固有の期待値です。データ購入では「どれだけ多いか」より、「現在の意思決定をどれだけ改善し、その改善がいくらになるか」を測る必要があります。

## 参照資料

- Brian Dalessandro, Claudia Perlich, and Troy Raeder, [Bigger is Better, but at What Cost? Estimating the Economic Value of Incremental Data Assets](https://journals.sagepub.com/doi/10.1089/big.2014.0010), *Big Data*, 2(2), 87–96, 2014. [PDF](https://journals.sagepub.com/doi/pdf/10.1089/big.2014.0010)

本記事の内容上の出典は上記の原論文です。実務手順のうち、time-based holdout、bootstrap confidence interval、運用費用の分離、online experimentは、論文の枠組みを導入時に検証するための本記事側の提案です。
