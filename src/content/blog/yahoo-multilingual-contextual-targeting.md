---
title: YahooのURLだけでWebページを分類する：多言語文脈に基づく広告配信の実装
description: YahooがKDD 2022で報告した多言語Webページ分類を、ロングテール対策、知識蒸留、URL-only推論、本番効果と限界から解説します。
publishedAt: 2026-09-05
updatedAt: 2026-10-01
category: AI
tags:
  - NLP
  - Transformer
  - Knowledge Distillation
  - 広告
  - 論文解説
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Yahooの研究チームがKDD 2022で発表した論文「[Multilingual Taxonomic Web Page Classification for Contextual Targeting at Yahoo](https://dl.acm.org/doi/10.1145/3534678.3539189)」です（[PDF](https://dl.acm.org/doi/pdf/10.1145/3534678.3539189?download=true)）。

この論文の価値は、ページ本文を読めない広告リクエストでも、本文を読める大規模な教師モデルの知識をURL-onlyの小型モデルへ移し、ロングテールかつ多言語の分類を本番環境へ導入した点にあります。

## 課題：cookieを使わず、今見ているページに合う広告を選ぶ

行動targetingは、過去の閲覧や検索といったユーザー単位の履歴を広告選択に使います。一方、文脈に基づく広告配信は、現在表示しているページの内容に合う広告を選びます。論文はGDPR、CCPA、プライバシーへの関心を背景に、user identityを追跡せず関連広告を出せる文脈に基づく広告配信の重要性が増したと説明しています。

Yahooが扱うのは、ページへ事前定義したtopicを付けるcategory-based contextual targetingです。分類先のYahoo Interest Categories（YIC）は5階層・442カテゴリからなります。1ページには複数カテゴリを付けられ、子カテゴリを付けたページはその祖先にも属します。したがって、これは通常の1-of-N分類ではなく、階層を持つマルチラベル分類です。

実用化には、次の問題がありました。

上位カテゴリや人気topicへデータが偏り、rare categoryの正例が極端に少ない。英語だけでなく、スペイン語、フランス語、ポルトガル語、繁体字中国語を1つの仕組みで扱いたい。

広告のbid requestで届くのは基本的にURLであり、本文のcrawlには時間と費用がかかる。大規模Transformerは精度が高くても、大量のURLを処理するには重い。


論文のシステムは、これらをdata sampling、loss re-weighting、多言語transfer learning、知識蒸留の組み合わせで解いています。

## 全体像：コンテンツモデルの知識をURL-only modelへ移す

処理は、学習時と本番環境時に分けると理解しやすくなります。

```text
学習時
URL + title + page body
  → XLM-RoBERTa-Largeのcontent teacher
  → categoryごとのsoft labelを大量の未label dataへ付与
  → URLだけを読むXLM-RoBERTa-Base studentを学習

production時
新しく観測したURL
  → crawled／uncrawledに応じた小型modelで事前分類
  → categoryごとのconfidence thresholdを適用
  → 予測categoryの祖先を追加
  → key-value storeへ保存
  → 広告配信時にlookup
```

広告オークションのたびに大規模モデルがページ本文を読む構成ではありません。この区別が重要です。論文のSpark Streaming pipelineは新しく現れたURLを処理し、category profileをキーバリューストアへ書きます。広告配信のlatency-sensitiveな経路では、その結果を参照します。

## 仕組み1：マルチラベル化とロングテール向け損失

一般的なmulti-class分類は、softmaxでカテゴリ間の確率を正規化します。しかし1ページに複数ラベルを付ける今回は、442個の出力それぞれへsigmoidを適用し、独立したbinary classifierとして学習します。Transformer部分の文書表現は全カテゴリで共有します。

このままbinary cross-entropyを使うと、2種類の不均衡が問題になります。

1. 1ページに付く正例はごく一部なので、損失の大半を負例が占める
2. カテゴリごとの出現数が大きく異なり、rare categoryが学習へほとんど影響しない

そこで論文は、正例だけにカテゴリ別の重みを与えます。カテゴリ`c`の出現数を`f_c`、最頻カテゴリの出現数を`max(f)`とすると、概念的には次の形です。

```text
w_c = μ × (max(f) + α) / (f_c + α)
α = γ × N
```

`μ`は正例全体を負例に対してどれだけ重くするか、`γ`はrare categoryをどれだけ強く持ち上げるかを調整します。`γ`を大きくするとカテゴリ間の重みは均一に近づき、0へ近づけると頻度の逆数に近い強い補正になります。

英語の391個のテスト可能なカテゴリを使ったRoBERTa-Largeの結果では、5エポック学習時のmAPはre-weightingなしの0.326から、正例全体への重み付けで0.435、カテゴリ別の重み付けで0.440へ上がりました。80エポックでは差が縮まり、それぞれ0.450、0.460、0.462でした。

つまり、短い学習では重み付けの調整に大きな効果がありますが、長く学習すれば差は小さくなります。それでもtail categoryでは重み付けの効果が大きく、全体mAPだけではロングテールの改善を十分に読めないことが分かります。

## 仕組み2：無作為抽出だけに頼らない

元トラフィックから無作為に集めた15,000ページでは、442カテゴリのうち27カテゴリにラベル付きページが1件もなく、114カテゴリは5件未満でした。そこで、次の順でrare categoryのデータを集めています。

### URL Collectionで最初の正例を作る

editorへカテゴリを提示し、該当する多様なsiteのURLを探してもらいます。集めたページは、対象カテゴリだけでなく該当する全カテゴリについて改めてannotationします。

これは実トラフィックを再現したsamplingではなく、意図的に偏った収集です。しかし、正例がほぼない段階では能動学習に使えるモデル自体を作れません。まず人がseed dataを集め、tailを予測できる初期モデルを作る役割があります。

### 能動学習で候補を広げる

初期モデルができた後は、rare categoryのscoreがしきい値を超えたページを選び、editorがラベルを確定します。論文では、無作為抽出、URL Collection、能動学習を複数回反復しています。

同じ25,000件で比べると、無作為だけの全category mAPは0.401、tailは0.282でした。15,000件のrandom dataにURL Collection 5,000件と能動学習5,000件を加えると、全体は0.452、tailは0.413です。絶対差では全体+0.051、tail+0.131であり、特にtailへの効果が大きい結果です。

この比較は、「母集団に忠実なデータだけを増やすこと」と「失敗しやすい領域を意図的に厚くすること」が別の目的を持つと示しています。一方、development／テスト集合はDSP trafficのstratified random sampleから作り、偏った学習データによる改善を実トラフィックに近い分布で評価しています。

## 仕組み3：5言語を1つのモデルで扱う

対象言語は英語、スペイン語、フランス語、ポルトガル語、繁体字中国語です。英語training setは56,000件、各非英語training setは48,000件で、合計248,000件です。非英語データの構築には、英語文書28,000件を各言語へGoogle Translate APIで翻訳したデータと、各言語で収集してeditorが直接annotationしたデータを使っています。

XLM-RoBERTa-Largeを5言語で追加学習した結果は次の通りです。

| Training data | en | es | fr | pt | zh-tw |
| --- | ---: | ---: | ---: | ---: | ---: |
| 各言語のeditorial data | 0.468 | 0.555 | 0.552 | 0.536 | 0.532 |
| editorial + translated data | 0.474 | 0.577 | 0.557 | 0.560 | 0.543 |

翻訳データを足した相対改善は、英語1.3%、スペイン語4.0%、フランス語0.9%、ポルトガル語4.5%、繁体字中国語2.1%です。また5言語で学習したモデルの英語mAP 0.474は、英語専用RoBERTa-Largeの0.460を相対3.0%上回りました。

ただし、これは多言語化すれば常に英語も改善するという一般則ではありません。YICという共通taxonomy、翻訳データ、各言語のeditorial dataを組み合わせた、この実験条件での結果です。

## 仕組み4：知識蒸留で小さく、URL-onlyにする

教師モデルはXLM-RoBERTa-Largeの355M parameter、生徒モデルはXLM-RoBERTa-Baseの125M parameterです。教師モデルが各カテゴリへ出した0〜1のscoreをソフトラベルとして生徒モデルを学習します。マルチラベルでは出力をカテゴリ間で正規化しないため、論文の予備実験ではtemperature scalingをせず、temperature 1のscoreをそのまま使う設定が最良でした。

コンテンツモデルのdistillationでは、editorial dataだけで学習した基盤モデルより、教師モデルがラベルを付けたrandom dataを使う方が良い結果でした。各言語600,000件、計300万件のrandom unlabeled dataだけからdistillした基盤モデルは、5言語すべてでLarge teacherと同等以上のmAPを記録しています。これは人手ラベルが不要になったという意味ではありません。教師モデルを作り、テストする基盤としてeditorial dataは依然必要です。

URL-only modelでは、さらに面白いdistillationを行います。

- 教師モデルの入力：ドメイン、path、title、page body
- 生徒モデルの入力：URLのドメインとpathだけ
- supervision：教師モデルが出した442カテゴリ分のソフトラベル

つまり生徒モデルは、学習時にも本文を直接受け取りません。それでも、本文を読んだ教師モデルの判断と大量のURLの対応から、`/sports/football`のような明示的トークンだけでなく、ドメイン固有の暗黙的なtopic対応も学びます。

最良のURL-only studentは、URL + コンテンツで学習した教師モデルからdistillし、editorial dataと各言語600,000件のrandom dataを使ったXLM-RoBERTa-Baseです。mAPは英語0.423、スペイン語0.508、フランス語0.497、ポルトガル語0.506、繁体字中国語0.418でした。同じURL-only入力でeditorial labelから通常学習したXLM-RoBERTa-Largeに対する相対改善は、順に13.1%、8.1%、10.7%、11.2%、11.2%です。従来のproduction XGBoostに対しても、mAPが相対26%改善したと報告されています。

もちろん、URL-onlyの精度は最良のコンテンツモデルには及びません。英語では0.423対0.474です。この手法の価値は本文を置き換えることではなく、crawlできず従来は分類対象外だったページを、精度を保ちながらcoverageへ加えることにあります。

## Production architectureと評価結果

YahooはAWS上にSpark Streaming pipelineを構築しました。AWS Kafkaから新しいURLを受け取り、AWS EMR上のDocker containerで小型モデルを実行します。予測にはカテゴリ別しきい値を適用し、期待precisionが0.8以上になるようfilterした後、予測カテゴリの全ancestorを追加してキーバリューストアへ保存します。

ここでhierarchyは損失へ直接組み込まれているわけではありません。442出力はsigmoidで個別に学習され、ancestor整合性は推論後の展開で保証します。シンプルで運用しやすい一方、親子間の関係そのものを学習へ活用する設計ではありません。

論文が示すproduction evidenceは2種類あり、分けて読む必要があります。

### Model launch前後の比較

3回のmodel launchについて、文脈に基づく広告配信がYahoo DSP全体へ占めるimpression、クリック、revenueの寄与率を、導入前15日と導入後15日で比較しています。

| Launch | Impression | Click | Revenue |
| --- | ---: | ---: | ---: |
| 英語コンテンツモデルをXGBoostからTransformerへ変更 | +56% | +17% | +77% |
| 英語URL-only modelを追加 | +257% | +194% | +353% |
| 4つの非英語言語へ展開 | +37% | +31% | +33% |

これは各指標の絶対値ではなく、DSP全体に対する文脈に基づく広告配信の寄与率の相対変化です。またrandomized experimentではないため、モデルだけの因果効果とは断定できません。特にURL-only model追加時の大幅増加には、分類精度だけでなく、それまでcrawlできず対象外だったURLを新たに扱えたcoverage効果が含まれます。

なお、最初のlaunchのrevenueについて、論文のTable 7は+77%ですが、直前の本文には+53%と書かれており不一致があります。本記事では表の値を掲載しつつ、原論文内で値が一致していないことを明記しておきます。

### Real-time user interest expansionのA/B test

別の実験では、現在のページから推定したカテゴリを、その時点のuser interestへ追加して広告auctionの候補を広げています。制御群とテスト群はそれぞれ全トラフィックの20%で、10日間実施されました。

CPM（1,000 impression当たりの広告プラットフォーム revenue）：+0.57%。過去のinterest categoryを持たないユーザーのCPM：+1.30%。

CPA（広告主がconversion 1件に支払う費用）：-1.27%。


こちらはA/B testですが、検証対象は分類モデル単体ではなく、予測カテゴリをaudience targetingへ追加するproduct feature全体です。サンプル数、信頼区間、p-valueは論文に記載されていないため、効果の不確実性までは評価できません。またこの機能は過去のuser interestも使うaudience targetingへの拡張であり、冒頭の「identityを追跡しない文脈に基づく広告配信」と同一視すべきではありません。

## 再現するときの最小設計

論文のデータ、YIC taxonomy、annotation guideline、学習コード、カテゴリ別しきい値、hyperparameter gridは公開されていません。そのため厳密な再現はできません。以下は、公開された設計を自社taxonomyへ適用するための記事側の実装案です。

### 1. 評価用データを先に固定する

実トラフィックからdevelopment／テスト集合を作り、URLやドメインが分割間で重複しないようグループ単位で分割します。tailを増やしたtraining setと、実トラフィックに近い評価setを混同しないことが重要です。カテゴリ別APに加え、ヘッド／torso／tail別mAP、precision threshold適用後のcoverageも測ります。

### 2. Seed収集から能動学習へ移る

正例が数件以下のカテゴリは、分野の専門家がURLを探してseedを作ります。初期モデルを学習できた後で、high-score、低confidence、モデル間の不一致などからannotation候補を選びます。意図的に集めたデータには収集方法を記録し、本番環境分布の推定には使いません。

### 3. コンテンツを読む教師モデルを作る

入力を`domain + path + title`と`body`の2 区分に分け、カテゴリ数と同じsigmoid出力を持たせます。論文の設定はコンテンツモデルが最大512トークン、URL-only modelが最大128トークンです。正例とtail categoryへの重みはdevelopment setで調整し、平均mAPだけでなく重要カテゴリごとの最低precisionを確認します。

### 4. Random URLをソフトラベル化する

大量のunlabeled URLをcrawlできる範囲で教師モデルへ通し、カテゴリごとのscoreを保存します。そのソフトラベルで小型コンテンツモデルとURL-only modelを別々に学習します。教師モデルの誤りも生徒モデルへ移るため、特定ドメイン、言語、tail categoryごとにhuman auditを入れます。

### 5. オフライン、shadow、段階展開で検証する

最初は既存モデルと新モデルを同じストリーム上で動かし、配信判断には使わないshadow modeで比較します。その後、一部トラフィックでcategory coverage、precision、遅延、コスト、広告主・publisher側の指標を確認します。model artifact、taxonomy version、しきい値を一緒にバージョン管理し、問題時に前の組み合わせへ戻せるようにします。

## 制約と現在の実装で注意したい点

この論文は本番規模の設計と事業指標まで示す貴重なindustry paperですが、次の限界があります。

private dataと非公開コードに依存し、第三者が同じ条件で再現できません。442カテゴリすべてにテスト正例がなく、言語ごとのmAPは378〜391個のテスト可能なカテゴリだけで計算されている。

model launchの結果は前後比較であり、季節性やトラフィック構成などの交絡を除いた因果効果ではありません。A/B testには統計的不確実性とサンプル数が示されていない。

対象は5言語であり、別スクリプト、低資源言語、code-switchしたページへの一般化は未検証。機械翻訳データは拡張に有効だったが、固有名詞、地域固有topic、translation artifactの分析は報告されていない。

URLは短く曖昧で、opaque ID、短縮URL、動的path、SEO spam、意図的なトークン操作に弱い可能性があります。カテゴリ別しきい値を固定すると、トラフィックや語彙の変化によるprecision driftを見逃し得る。


また、文脈に基づく広告配信はユーザー履歴を使わない設計を可能にしますが、採用するだけでGDPRやCCPAへの準拠が保証されるわけではありません。URL自体に識別子や検索語などのsensitive dataが含まれることもあるため、logging、保存期間、アクセス制御、URL正規化とredactionは別途設計が必要です。

## まとめ

この研究から最も学べるのは、強いTransformerを選んだこと以上に、制約ごとに手段を分けた設計です。

ロングテールには、URL Collectionでseedを作ってから能動学習へ進む。マルチラベルの極端な不均衡には、正例とrare categoryを分けてre-weightします。

多言語データには、直接annotationと機械翻訳を組み合わせる。serving costには、大型教師モデルから小型生徒モデルへの知識蒸留を使います。

crawlできないURLには、コンテンツを読む教師モデルからURL-only studentへ異なる入力間で知識を移す。配信時にはモデルを同期実行せず、事前計算したcategory profileを参照します。


特に、豊富な情報を読める教師モデルから、本番環境では入手できる情報が少ない生徒モデルへdistillする考え方は、広告以外にも応用できます。学習時にだけ利用できる高価なシグナルを教師モデルへ集め、本番制約に合う入力とmodel sizeへ知識を移す設計として捉えると、この論文の射程が見えやすくなります。

## 参照

- Eric Ye et al., [Multilingual Taxonomic Web Page Classification for Contextual Targeting at Yahoo](https://dl.acm.org/doi/10.1145/3534678.3539189), Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 4372–4380.
- Eric Ye et al., [著者公開PDF](https://www.cs.columbia.edu/~kapil/documents/kdd22targeting.pdf).

本記事の内容上の出典は、上記の原論文です。再現セクションの段階的な導入方法と運用上の注意は、論文の公開情報をもとにした記事側の提案です。
