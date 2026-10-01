---
title: 複数の再クロール戦略をEXP3で束ねる：Googleの商用Webクローリング研究
description: GoogleのWWW 2020論文をもとに、価格情報の鮮度を限られたクロール予算で高めるK-armed adversarial bandits方式を解説します。
publishedAt: 2026-09-07
updatedAt: 2026-10-01
category: Machine Learning
tags:
  - Webクローリング
  - Multi-Armed Bandit
  - EXP3
  - 機械学習
  - 論文解説
draft: false
---

> AI利用の明示<br>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載しています。利用時は必ず原論文も確認してください。

今回取り上げるのは、Google ResearchのShuguang HanらがWWW 2020で発表した論文「[Adversarial Bandits Policy for Crawling Commercial Web Content](https://research.google/pubs/adversarial-bandits-policy-for-crawling-commercial-web-content/)」です（[PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/5512.pdf)、[DOI](https://doi.org/10.1145/3366423.3380125)）。

この論文の価値は、将来の価格変化を完全に予測しようとするだけでなく、性質の異なる再クロール戦略をEXP3で選び分けることで、予測誤差や時間変化に強い更新方針を作ったことにあります。

## 課題：すべての商品ページを頻繁には巡回できない

ショッピング検索では、検索側のデータベースに保存した価格と、販売ページの現在価格が一致している必要があります。しかし、商品販売商品は膨大であり、すべてのページを高頻度で再クロールするのは現実的ではありません。販売者から受け取るfeedだけでも、低遅延性や品質管理の面で不十分な場合があります。

そこで必要になるのが、限られたクロール予算をどの商品へ割り当てるかというrecrawl policy（再クロール方針）です。素朴には「価格がよく変わる商品ほど頻繁に巡回する」と考えたくなります。ところが、極端に変化の激しいページは巡回直後にまた変わり得るため、予算を集中しても利用者が見る時点の価格を新鮮に保てないことがあります。

論文は次の2指標を区別しています。

| 指標 | 何を測るか | 重視するもの |
| --- | --- | --- |
| click-weighted freshness | クリックされた時点で、保存価格が実価格と一致したクリックの割合 | 利用者が実際に見る商品 |
| offer-level freshness | ある時点で、保存価格が実価格と一致する販売商品の割合 | 商品集合全体の網羅性 |

原論文での第1指標の名称は`click-weighted freshness`です。以降はこの表記を使います。主目的はclick-weighted freshnessの最大化であり、offer-level freshnessは副次的に維持したい指標です。この目的の違いが、後で重要になります。

## まず理解したい5つの再クロール戦略

論文が候補にしたのは、次の5戦略です。各戦略は、1 time ステップあたり合計`b`件という同じ予算の範囲で、販売商品ごとの再クロール率を決めます。

| 戦略 | 予算を多く配る販売商品 | 主な性質 |
| --- | --- | --- |
| Uniform | 全販売商品へ均等 | 過去に選ばれにくかった販売商品も探索できる |
| Change weighted | 価格変化率が高い | 変化が激しすぎる販売商品へ予算を使いすぎやすい |
| Click weighted | クリック率が高い | click-weighted freshnessと目的が近い |
| Impression weighted | 表示率が高い | クリックよりシグナルが密で、将来のクリックの代理になり得る |
| LambdaCrawl | クリック率と価格変化率を組み合わせる | 変化が激しすぎる販売商品を抑制する制約最適化手法 |

LambdaCrawlは、クリック率と価格変化率が一定で、正確に分かるという前提では理論的に優れた割り当てを計算できます。しかし実際の価格には季節・曜日・時間帯の変動があり、将来のクリック率と変化率も事前には分かりません。推定値が外れれば、閉形式の最適解を使っても実際の割り当ては最適になりません。

ここで論文は、1つの戦略を選んで固定するのではなく、戦略そのものをbanditのarmとして扱う方向へ進みます。

## K-armed adversarial banditsとして定式化する

提案方式KABでは、2時間ごとに次の処理を繰り返します。

```text
各戦略の重みから選択確率を計算
  ↓
1つの再クロール戦略（arm）をsample
  ↓
その戦略で全offerの再クロール率を計算
  ↓
予算内のofferをクロール
  ↓
古い保存価格を更新できた量をrewardとして重みに反映
```

販売商品ごとにarmを選ぶのではなく、1 time ステップにつき1つのarmを選び、全販売商品へ適用する点がポイントです。各戦略の中ですでに予算制約を処理しているため、複数戦略をまたぐ複雑な予算最適化を避けられます。

報酬は、再クロールによって古い価格を正しい価格へ更新できた場合だけ正になります。さらにクリック率で重み付けし、合計を`[0, 1]`へ正規化します。したがって、累積報酬を増やすことは、論文の定義ではclick-weighted freshnessを増やすことに対応します。

### なぜstochastic banditではなくEXP3なのか

価格変更やクリックの分布は、saleや季節によって変化します。論文は報酬分布の定常性を仮定しないadversarial banditを採用し、実装には標準的なEXP3を使いました。

EXP3は、過去に高い報酬を得たarmを活用しながら、確率`γ`で探索を残します。論文の評価では`γ = 0.1`が最良で、EXP3-IXとEXP3++も試したものの、標準EXP3と同程度だったため、より単純なEXP3を採用しています。

Uniformは単なる弱い比較手法ではありません。予測モデルが見落とした販売商品も対象にする探索役になる点が重要です。クリックやメタデータに基づく戦略は同じ販売商品を繰り返し選びがちですが、Uniformを混ぜると、履歴が乏しい販売商品にも更新機会が生まれます。

## banditの前に、予測値そのものを改善する

Uniform以外の戦略には、翌日の価格変更、クリック、impressionの確率が必要です。過去30日の頻度だけで推定すると、履歴のない新商品に弱く、選ばれたページだけ観測が増えるfeedback loopも起きます。

論文は各予測を二値分類として扱い、次の3種類の特徴量構成を比較しました。

1. メタデータのみ
2. 履歴のみ
3. メタデータと履歴

メタデータにはブランド、商品の状態、国、曜日、言語、販売者、商品カテゴリを使います。履歴には直近1か月の価格変更頻度、最後の変更からの時間、過去1日・1週・2週・1か月のクリック数とimpression数を使います。

モデルはTensorFlowの`DNNClassifier`で、隠れ層は`256 → 128 → 64`、活性化関数はReLU、オプティマイザーはAdagrad、L1・L2正則化はともに`0.001`です。これは2020年の論文で採用された構成であり、現在の環境でも最適だという意味ではありません。

テスト集合を20分割して求めたAUCの平均と標準偏差は次の通りです。

| 予測タスク | メタデータ | History | メタデータ + History |
| --- | ---: | ---: | ---: |
| Price change | 0.860 ± 0.008 | 0.833 ± 0.011 | **0.882 ± 0.007** |
| Click | 0.796 ± 0.021 | 0.948 ± 0.006 | **0.949 ± 0.006** |
| Impression | 0.736 ± 0.008 | **0.896 ± 0.003** | 0.895 ± 0.003 |

価格変更ではメタデータが履歴だけのモデルを上回り、メタデータと履歴の併用が最良でした。一方、クリックとimpressionでは履歴が強く、メタデータを加える効果はほとんどありません。それでも論文は、全タスクで安定しておりコールドスタートにも対応できるメタデータ＋履歴モデルを後続の実験に使っています。

## データと評価方法

実験には、Google Shoppingにインデックスされた130万販売商品を使っています。2018年8月1日から2019年4月10日まで1時間ごとに巡回し、取得したsnapshotは合計数十億件です。ただし、リクエスト拒否などにより全snapshotの取得に成功したわけではありません。

予測モデルのデータ分割は時間で分離されています。

| 用途 | 期間 | 規模 |
| --- | --- | ---: |
| Train | 2018-08-01〜2018-12-31 | 1億example |
| Validation | 2019-01-01〜2019-01-09 | 60万example |
| テスト | 2019-01-10以降 | 800万example |

recrawl policyの評価では、2019年1月10日から3月10日をwarm-upに使い、3月11日から4月10日の31日間を評価期間にしています。実際に各方針で再クロールした結果をオンライン比較したのではなく、1時間ごとの観測がある本番データ上でローカル価格を更新するシミュレーションです。

再クロールの選択は確率的なので、全評価を100回繰り返します。31日×100回の3,100値は正規分布ではなかったため、論文はmedianを報告し、差の検定にはWilcoxon signed-rank testを使っています。

## 結果：KAB5はLambdaCrawlを上回るが、目的によって最良は違う

最終方式のKAB5は、Uniform、予測モデル版のChange weighted、Click weighted、Impression weighted、LambdaCrawlの5 armを組み合わせます。

代表例として、1 time ステップに全販売商品の10%を再クロールできる予算でのmedianを抜き出します。

| Policy | Click-weighted freshness | Offer-level freshness |
| --- | ---: | ---: |
| Uniform | 0.7460 | **0.7426** |
| Predictive Click weighted | 0.8976 | 0.4447 |
| Predictive LambdaCrawl | 0.8983 | 0.6550 |
| KAB5 | **0.9027** | 0.6604 |

KAB5は、10〜90%の9種類すべての予算で、Predictive LambdaCrawlより両指標が有意に高くなりました（`p < 0.05`）。10%予算での改善率medianは、click-weighted freshnessが`+0.93%`、offer-level freshnessが`+1.07%`です。この改善率は各評価値に対する増加率のmedianなので、表中のmedian同士を割った値とは一致しません。

さらに0.1〜9%の厳しい予算でも、KAB5はPredictive LambdaCrawlを両指標で有意に上回り、差はとくに1%未満で大きくなりました。ただしグラフから精密な値は読み取れないため、ここでは増加幅を数値化しません。

一方、offer-level freshnessだけを見れば、全販売商品を均等に探索するUniformが最良です。KAB5の強みは、すべての指標で全比較手法に勝つことではありません。主目的であるクリック時の価格鮮度を高く保ちつつ、LambdaCrawlより商品全体の鮮度も改善する点にあります。

また、履歴頻度を予測モデルへ置き換えた効果も大きく、Change weightedではclick-weighted freshnessが約2倍、各heuristicのoffer-level freshnessが最大4〜5倍になったと報告されています。つまり成果はbanditだけではなく、parameter estimationの改善との組み合わせです。

## 実装するときの最小構成

原論文はアルゴリズム1と主要なハイパーパラメータを示していますが、Googleのoffer data、学習済みモデル、実装コードは公開していません。以下は完全な再現コードではなく、公開された設計を別のデータ同期タスクへ適用するための疑似コードです。

```ts
type Arm = 'uniform' | 'change' | 'click' | 'impression' | 'lambda';

const weights: Record<Arm, number> = initializeToOne();
const gamma = 0.1; // 原論文の評価設定。環境ごとに再調整する

for (const window of twoHourWindows) {
  const probabilities = exp3Probabilities(weights, gamma);
  const arm = sample(probabilities);

  const estimates = await predictNextDaySignals(window.offers);
  const recrawlRates = allocateBudget(arm, estimates, window.budget);
  const results = await crawl(sampleOffers(recrawlRates));

  const reward = normalize(
    results.sum((offer) =>
      offer.previousStoredPrice !== offer.currentPrice
        ? offer.clickRate
        : 0,
    ),
  );

  updateExp3Weight(weights, arm, reward, probabilities[arm], gamma);
}
```

実サービスへ導入するなら、少なくとも次を追加で設計する必要があります。

- 販売者ごとの呼び出し頻度の制限、robots.txt、タイムアウトを満たすscheduler
- crawl失敗と「価格が変わらなかった」を区別する観測schema
- 特徴量・予測モデル・policy weightのバージョン管理
- click-weightedとoffer-levelの両方を監視するダッシュボード
- 新商品、国、カテゴリ、販売者別のslice評価
- 方針更新が不調なときにUniformや既存方針へ戻す切り戻し

とくに報酬設計は、そのままコピーすべきではありません。価格の誤表示による損失、人気商品への偏り、ロングテール商品の最低更新頻度を事業要件として整理し、必要なら制約や複数指標を加えるべきです。

## この研究の限界

実務へ応用する際は、次の点に注意が必要です。

評価は本番データを使った大規模シミュレーションであり、実トラフィック上のA/B testや本番導入の結果ではありません。データセットとコードが非公開で、第三者は論文の数値を完全再現できません。

対象は商品価格であり、在庫、news、イベント、IoT sensorへの一般化は将来方向として述べられた段階。KABは2時間ごとに1戦略を全販売商品へ適用するため、販売商品の文脈から直接armを選ぶ方式ではありません。

主報酬がclick-weighted freshnessなので、人気商品の鮮度とロングテール商品の網羅性にはトレードオフがあります。予算以外の本番環境制約、たとえばhost単位のpolitenessや実運用コストは、この実験で定量評価されていない。

観測期間は2018〜2019年であり、販売者構成やcrawl環境が異なる現在のシステムへ数値をそのまま移せない。


論文自身も、次の発展としてoffer contextから直接クロール率を決めるcontextual banditを挙げています。現在なら、単純なEXP3を堅牢な比較手法として置き、contextual方式がコールドスタート、計算量、方針の説明可能性を含めて本当に上回るかをoffline replayとshadow運用で比較するのが現実的でしょう。

## まとめ

この研究から得られる最も実務的な教訓は、予測モデルの精度を上げることと、予測を使う方針を堅牢にすることは別の問題だということです。

メタデータを加えた予測モデルはコールドスタートを緩和します。それでも各戦略には固有の偏りが残ります。そこでEXP3が、過去に良かった戦略を活用しながらUniformによる探索を残し、変化する状況に合わせて戦略の比率を更新します。

ただし、何を報酬にするかで「新鮮さ」の意味は変わります。クリックされた商品の正しさを優先するのか、全商品の網羅性を守るのかを先に決め、その目的と副作用を別々の指標で測ることが、bandit algorithmの選択以上に重要です。

## 参照

- Shuguang Han et al., [Adversarial Bandits Policy for Crawling Commercial Web Content](https://research.google/pubs/adversarial-bandits-policy-for-crawling-commercial-web-content/), Proceedings of The Web Conference 2020, pp. 407–417.
- [原論文PDF](https://storage.googleapis.com/gweb-research2023-media/pubtools/5512.pdf)
- [ACM Digital Library DOI: 10.1145/3366423.3380125](https://doi.org/10.1145/3366423.3380125)
