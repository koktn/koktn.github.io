---
title: YahooのURLだけでWebページを分類する――多言語contextual targetingの実装
description: YahooがKDD 2022で報告した多言語Webページ分類を、long-tail対策、knowledge distillation、URL-only推論、本番効果と限界から解説します。
publishedAt: 2026-09-05
category: AI
tags:
  - NLP
  - Transformer
  - Knowledge Distillation
  - 広告
  - 論文解説
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Yahooの研究チームがKDD 2022で発表した論文「[Multilingual Taxonomic Web Page Classification for Contextual Targeting at Yahoo](https://dl.acm.org/doi/10.1145/3534678.3539189)」です（[PDF](https://dl.acm.org/doi/pdf/10.1145/3534678.3539189?download=true)）。

この論文の価値を一文でまとめると、**ページ本文を読めない広告リクエストでも、本文を読める大規模なteacher modelの知識をURL-onlyの小型modelへ移し、long-tailかつ多言語の分類をproductionへ載せた**点にあります。

## 課題：cookieを使わず、今見ているページに合う広告を選ぶ

行動targetingは、過去の閲覧や検索といったuser単位の履歴を広告選択に使います。一方、contextual targetingは、現在表示しているページの内容に合う広告を選びます。論文はGDPR、CCPA、privacyへの関心を背景に、user identityを追跡せず関連広告を出せるcontextual targetingの重要性が増したと説明しています。

Yahooが扱うのは、ページへ事前定義したtopicを付けるcategory-based contextual targetingです。分類先のYahoo Interest Categories（YIC）は5階層・442 categoryからなります。1ページには複数categoryを付けられ、子categoryを付けたページはその祖先にも属します。したがって、これは通常の1-of-N分類ではなく、**階層を持つmulti-label分類**です。

実用化には、次の問題がありました。

- 上位categoryや人気topicへdataが偏り、rare categoryの正例が極端に少ない
- 英語だけでなく、スペイン語、フランス語、ポルトガル語、繁体字中国語を1つの仕組みで扱いたい
- 広告のbid requestで届くのは基本的にURLであり、本文のcrawlには時間と費用がかかる
- 大規模Transformerは精度が高くても、大量のURLを処理するには重い

論文のsystemは、これらをdata sampling、loss re-weighting、多言語transfer learning、knowledge distillationの組み合わせで解いています。

## 全体像：content modelの知識をURL-only modelへ移す

処理は、学習時とproduction時に分けると理解しやすくなります。

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

ここで重要なのは、「広告auctionのたびに大規模modelがページ本文を読む」構成ではないことです。論文のSpark Streaming pipelineは新しく現れたURLを処理し、category profileをkey-value storeへ書きます。広告配信のlatency-sensitiveな経路では、その結果をlookupします。

## 仕組み1：multi-label化とlong-tail向けloss

一般的なmulti-class分類は、softmaxでcategory間の確率を正規化します。しかし1ページに複数labelを付ける今回は、442個の出力それぞれへsigmoidを適用し、独立したbinary classifierとして学習します。Transformer部分の文書表現は全categoryで共有します。

このままbinary cross-entropyを使うと、2種類の不均衡が問題になります。

1. 1ページに付く正例はごく一部なので、lossの大半を負例が占める
2. categoryごとの出現数が大きく異なり、rare categoryが学習へほとんど影響しない

そこで論文は、正例だけにcategory別の重みを与えます。category `c` の出現数を `f_c`、最頻categoryの出現数を `max(f)` とすると、概念的には次の形です。

```text
w_c = μ × (max(f) + α) / (f_c + α)
α = γ × N
```

`μ`は正例全体を負例に対してどれだけ重くするか、`γ`はrare categoryをどれだけ強く持ち上げるかを調整します。`γ`を大きくするとcategory間の重みは均一に近づき、0へ近づけると頻度の逆数に近い強い補正になります。

英語の391個のtest可能なcategoryを使ったRoBERTa-Largeの結果では、5 epoch学習時のmAPはre-weightingなしの0.326から、正例全体への重み付けで0.435、category別の重み付けで0.440へ上がりました。80 epochでは差が縮まり、それぞれ0.450、0.460、0.462でした。

つまり、短い学習ではre-weightingが非常に効きますが、長く学習すれば差は小さくなります。それでもtail categoryでは重み付けの効果が大きく、全体mAPだけではlong-tailの改善を十分に読めないことが分かります。

## 仕組み2：random samplingだけに頼らない

元trafficから無作為に集めた15,000ページでは、442 categoryのうち27 categoryにlabel付きページが1件もなく、114 categoryは5件未満でした。そこで、次の順でrare categoryのdataを集めています。

### URL Collectionで最初の正例を作る

editorへcategoryを提示し、該当する多様なsiteのURLを探してもらいます。集めたページは、対象categoryだけでなく該当する全categoryについて改めてannotationします。

これは実trafficを再現したsamplingではなく、意図的に偏った収集です。しかし、正例がほぼない段階ではactive learningに使えるmodel自体を作れません。まず人がseed dataを集め、tailを予測できる初期modelを作る役割があります。

### Active learningで候補を広げる

初期modelができた後は、rare categoryのscoreがthresholdを超えたページを選び、editorがlabelを確定します。論文では、random sampling、URL Collection、active learningを複数回反復しています。

同じ25,000件で比べると、randomだけの全category mAPは0.401、tailは0.282でした。15,000件のrandom dataにURL Collection 5,000件とactive learning 5,000件を加えると、全体は0.452、tailは0.413です。絶対差では全体+0.051、tail+0.131であり、特にtailへの効果が大きい結果です。

この比較は、「母集団に忠実なdataだけを増やすこと」と「失敗しやすい領域を意図的に厚くすること」が別の目的を持つと示しています。一方、development／test setはDSP trafficのstratified random sampleから作り、偏ったtraining dataによる改善を実trafficに近い分布で評価しています。

## 仕組み3：5言語を1つのmodelで扱う

対象言語は英語、スペイン語、フランス語、ポルトガル語、繁体字中国語です。英語training setは56,000件、各非英語training setは48,000件で、合計248,000件です。非英語dataの構築には、英語文書28,000件を各言語へGoogle Translate APIで翻訳したdataと、各言語で収集してeditorが直接annotationしたdataを使っています。

XLM-RoBERTa-Largeを5言語でfine-tuningした結果は次の通りです。

| Training data | en | es | fr | pt | zh-tw |
| --- | ---: | ---: | ---: | ---: | ---: |
| 各言語のeditorial data | 0.468 | 0.555 | 0.552 | 0.536 | 0.532 |
| editorial + translated data | 0.474 | 0.577 | 0.557 | 0.560 | 0.543 |

翻訳dataを足した相対改善は、英語1.3%、スペイン語4.0%、フランス語0.9%、ポルトガル語4.5%、繁体字中国語2.1%です。また5言語で学習したmodelの英語mAP 0.474は、英語専用RoBERTa-Largeの0.460を相対3.0%上回りました。

ただし、これは多言語化すれば常に英語も改善するという一般則ではありません。YICという共通taxonomy、翻訳data、各言語のeditorial dataを組み合わせた、この実験条件での結果です。

## 仕組み4：knowledge distillationで小さく、URL-onlyにする

teacherはXLM-RoBERTa-Largeの355M parameter、studentはXLM-RoBERTa-Baseの125M parameterです。teacherが各categoryへ出した0〜1のscoreをsoft labelとしてstudentを学習します。multi-labelでは出力をcategory間で正規化しないため、論文の予備実験ではtemperature scalingをせず、temperature 1のscoreをそのまま使う設定が最良でした。

content modelのdistillationでは、editorial dataだけで学習したBase modelより、teacherがlabelを付けたrandom dataを使う方が良い結果でした。各言語600,000件、計300万件のrandom unlabeled dataだけからdistillしたBase modelは、5言語すべてでLarge teacherと同等以上のmAPを記録しています。これは人手labelが不要になったという意味ではありません。teacherを作り、testする基盤としてeditorial dataは依然必要です。

URL-only modelでは、さらに面白いdistillationを行います。

- teacherの入力：domain、path、title、page body
- studentの入力：URLのdomainとpathだけ
- supervision：teacherが出した442 category分のsoft label

つまりstudentは、学習時にも本文を直接受け取りません。それでも、本文を読んだteacherの判断と大量のURLの対応から、`/sports/football`のような明示的tokenだけでなく、domain固有の暗黙的なtopic対応も学びます。

最良のURL-only studentは、URL + contentで学習したteacherからdistillし、editorial dataと各言語600,000件のrandom dataを使ったXLM-RoBERTa-Baseです。mAPは英語0.423、スペイン語0.508、フランス語0.497、ポルトガル語0.506、繁体字中国語0.418でした。同じURL-only入力でeditorial labelから通常学習したXLM-RoBERTa-Largeに対する相対改善は、順に13.1%、8.1%、10.7%、11.2%、11.2%です。従来のproduction XGBoostに対しても、mAPが相対26%改善したと報告されています。

もちろん、URL-onlyの精度は最良のcontent modelには及びません。英語では0.423対0.474です。この手法の価値は本文を置き換えることではなく、crawlできず従来は分類対象外だったページを、精度を保ちながらcoverageへ加えることにあります。

## Production architectureと評価結果

YahooはAWS上にSpark Streaming pipelineを構築しました。AWS Kafkaから新しいURLを受け取り、AWS EMR上のDocker containerで小型modelを実行します。予測にはcategory別thresholdを適用し、期待precisionが0.8以上になるようfilterした後、予測categoryの全ancestorを追加してkey-value storeへ保存します。

ここでhierarchyはlossへ直接組み込まれているわけではありません。442出力はsigmoidで個別に学習され、ancestor整合性は推論後の展開で保証します。シンプルで運用しやすい一方、親子間の関係そのものを学習へ活用する設計ではありません。

論文が示すproduction evidenceは2種類あり、分けて読む必要があります。

### Model launch前後の比較

3回のmodel launchについて、contextual targetingがYahoo DSP全体へ占めるimpression、click、revenueの寄与率を、導入前15日と導入後15日で比較しています。

| Launch | Impression | Click | Revenue |
| --- | ---: | ---: | ---: |
| 英語content modelをXGBoostからTransformerへ変更 | +56% | +17% | +77% |
| 英語URL-only modelを追加 | +257% | +194% | +353% |
| 4つの非英語言語へ展開 | +37% | +31% | +33% |

これは各metricの絶対値ではなく、DSP全体に対するcontextual targetingの**寄与率の相対変化**です。またrandomized experimentではないため、modelだけの因果効果とは断定できません。特にURL-only model追加時の大幅増加には、分類精度だけでなく、それまでcrawlできず対象外だったURLを新たに扱えたcoverage効果が含まれます。

なお、最初のlaunchのrevenueについて、論文のTable 7は+77%ですが、直前の本文には+53%と書かれており不一致があります。本記事では表の値を掲載しつつ、原論文内で値が一致していないことを明記しておきます。

### Real-time user interest expansionのA/B test

別の実験では、現在のページから推定したcategoryを、その時点のuser interestへ追加して広告auctionの候補を広げています。control群とtest群はそれぞれ全trafficの20%で、10日間実施されました。

- CPM（1,000 impression当たりの広告platform revenue）：+0.57%
- 過去のinterest categoryを持たないuserのCPM：+1.30%
- CPA（広告主がconversion 1件に支払う費用）：-1.27%

こちらはA/B testですが、検証対象は分類model単体ではなく、予測categoryをaudience targetingへ追加するproduct feature全体です。sample数、confidence interval、p-valueは論文に記載されていないため、効果の不確実性までは評価できません。またこの機能は過去のuser interestも使うaudience targetingへの拡張であり、冒頭の「identityを追跡しないcontextual targeting」と同一視すべきではありません。

## 再現するときの最小設計

論文のdata、YIC taxonomy、annotation guideline、学習code、category別threshold、hyperparameter gridは公開されていません。そのため厳密な再現はできません。以下は、公開された設計を自社taxonomyへ適用するための記事側の実装案です。

### 1. 評価用dataを先に固定する

実trafficからdevelopment／test setを作り、URLやdomainがsplit間で重複しないようgroup単位で分割します。tailを増やしたtraining setと、実trafficに近い評価setを混同しないことが重要です。category別APに加え、head／torso／tail別mAP、precision threshold適用後のcoverageも測ります。

### 2. Seed収集からactive learningへ移る

正例が数件以下のcategoryは、domain expertがURLを探してseedを作ります。初期modelを学習できた後で、high-score、低confidence、model間の不一致などからannotation候補を選びます。意図的に集めたdataには収集方法を記録し、production分布の推定には使いません。

### 3. Content teacherを作る

入力を`domain + path + title`と`body`の2 segmentに分け、category数と同じsigmoid出力を持たせます。論文の設定はcontent modelが最大512 token、URL-only modelが最大128 tokenです。正例とtail categoryへの重みはdevelopment setで調整し、平均mAPだけでなく重要categoryごとの最低precisionを確認します。

### 4. Random URLをsoft label化する

大量のunlabeled URLをcrawlできる範囲でteacherへ通し、categoryごとのscoreを保存します。そのsoft labelで小型content modelとURL-only modelを別々に学習します。teacherの誤りもstudentへ移るため、特定domain、言語、tail categoryごとにhuman auditを入れます。

### 5. Offline、shadow、段階展開で検証する

最初は既存modelと新modelを同じstream上で動かし、配信判断には使わないshadow modeで比較します。その後、一部trafficでcategory coverage、precision、latency、cost、広告主・publisher側のmetricを確認します。model artifact、taxonomy version、thresholdを一緒にversion管理し、問題時に前の組み合わせへ戻せるようにします。

## Limitationと現在の実装で注意したい点

この論文はproduction規模の設計とbusiness metricまで示す貴重なindustry paperですが、次の限界があります。

- private dataと非公開codeに依存し、第三者が同じ条件で再現できない
- 442 categoryすべてにtest正例がなく、言語ごとのmAPは378〜391個のtest可能なcategoryだけで計算されている
- model launchの結果は前後比較であり、季節性やtraffic構成などの交絡を除いた因果効果ではない
- A/B testには統計的不確実性とsample数が示されていない
- 対象は5言語であり、別script、低resource言語、code-switchしたページへの一般化は未検証
- 機械翻訳dataは拡張に有効だったが、固有名詞、地域固有topic、translation artifactの分析は報告されていない
- URLは短く曖昧で、opaque ID、短縮URL、動的path、SEO spam、意図的なtoken操作に弱い可能性がある
- category別thresholdを固定すると、trafficや語彙の変化によるprecision driftを見逃し得る

また、contextual targetingはuser historyを使わない設計を可能にしますが、採用するだけでGDPRやCCPAへの準拠が保証されるわけではありません。URL自体に識別子や検索語などのsensitive dataが含まれることもあるため、logging、保存期間、access control、URL正規化とredactionは別途設計が必要です。

## まとめ

この研究から最も学べるのは、強いTransformerを選んだこと以上に、制約ごとに手段を分けた設計です。

- long-tailには、URL Collectionでseedを作ってからactive learningへ進む
- multi-labelの極端な不均衡には、正例とrare categoryを分けてre-weightする
- 多言語dataには、直接annotationと機械翻訳を組み合わせる
- serving costには、大型teacherから小型studentへのknowledge distillationを使う
- crawlできないURLには、content teacherからURL-only studentへ異なる入力間で知識を移す
- 配信時にはmodelを同期実行せず、事前計算したcategory profileをlookupする

特に、豊富な情報を読めるteacherから、productionでは入手できる情報が少ないstudentへdistillする考え方は、広告以外にも応用できます。学習時にだけ利用できる高価なsignalをteacherへ集め、本番制約に合う入力とmodel sizeへ知識を移す設計として捉えると、この論文の射程が見えやすくなります。

## 参照

- Eric Ye et al., [Multilingual Taxonomic Web Page Classification for Contextual Targeting at Yahoo](https://dl.acm.org/doi/10.1145/3534678.3539189), Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2022, pp. 4372–4380.
- Eric Ye et al., [著者公開PDF](https://www.cs.columbia.edu/~kapil/documents/kdd22targeting.pdf).

本記事の内容上の出典は、上記の原論文です。再現セクションの段階的な導入方法と運用上の注意は、論文の公開情報をもとにした記事側の提案です。
