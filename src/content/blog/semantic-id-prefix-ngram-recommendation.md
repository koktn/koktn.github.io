---
title: Semantic ID prefix n-gram：推薦モデルのhash衝突を「意味の共有」に変える
description: Metaの広告ランキング論文をもとに、RQ-VAEによる階層Semantic ID、prefix n-gram、ロングテール・コールドスタート・予測安定性への効果と本番環境導入上の注意点を解説します。
publishedAt: 2026-09-09
updatedAt: 2026-10-01
category: AI
tags:
  - Recommendation System
  - Semantic ID
  - Representation Learning
  - Vector Quantization
  - Meta
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Carolina Zhengらが2025年4月2日にarXivへ投稿した「[Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID](https://arxiv.org/abs/2504.02137)」です。Columbia UniversityとMetaの研究者によるプレプリントで、Meta Adsの本番システムへSemantic IDを導入した経験まで報告しています。

この論文の価値は、巨大で入れ替わりの激しいitem IDをランダムに埋め込みへ割り当てる代わりに、コンテンツの意味から作った階層IDを共有させ、ロングテール、コールドスタート、時間変化に強い推薦表現を本番規模で実現した点にあります。

## 課題：random hashingは衝突相手の意味を選べない

大規模な推薦モデルは、広告や商品などのraw item IDを埋め込みテーブルのrowへ変換します。しかし、アイテム数を`I`、用意できるrow数を`H`としたとき、本番環境ではしばしば`I > H`です。すべてのアイテムへ専用rowを割り当てられないため、元のIDをhashして複数アイテムに同じ埋め込みを共有させます。

この方法はメモリを抑えられる一方、同じrowへ入るアイテムの組み合わせはランダムです。pizza広告と旅行広告のように無関係なアイテムが衝突すると、それぞれの学習例が同じ埋め込みへ矛盾したgradient updateを加えます。

論文は、Meta Adsのアイテム分布にある3つの難しさを整理しています。

| 課題 | 論文中の観測 | random hashingへの影響 |
| --- | --- | --- |
| item cardinality | アイテム数が実用的な埋め込みテーブルより大きい | 無関係なアイテム同士の衝突が避けられない |
| impression skew | 上位0.1%のアイテムが25%、次の5.5%が50%、残る94.4%が25%のimpressionを占める | tail itemは学習例が少なく、偶然の衝突相手から悪影響を受けやすい |
| ID drifting | 初期アイテム集合の半分が6日後にはシステムから退出する | 同じrowが時間とともに別のアイテムを表し、学習済みの意味が変わる |

特に厄介なのは、古い広告が終了して似た内容の新しい広告が作られても、元のIDは別物になることです。専用埋め込みなら新しいrowは未学習、random hashingなら無関係な既存rowから始まります。どちらも「似た内容についてすでに学んだこと」を自然には引き継げません。

## 全体像：コンテンツを階層的な離散トークンへ変える

Semantic IDは元のIDそのものではなく、アイテムのテキスト・image・videoから得た内容 embeddingを離散化したIDです。論文の流れを単純化すると次のようになります。

```text
itemのtext・image・video
  ↓ content understanding model
dense content embedding
  ↓ RQ-VAEによるresidual quantization
(c1, c2, ..., cL) というcoarse-to-fineなSemantic ID
  ↓ prefix n-gram parameterization
(c1), (c1,c2), ..., (c1,...,cn) をembedding rowへ変換
  ↓ lookupしてsum-pooling
ranking modelで使うitem表現
```

意味が近いアイテムほど同じコード、または同じprefixを共有しやすくなります。衝突をなくすのではなく、衝突するなら意味の近いアイテム同士にする発想です。

## RQ-VAEがcoarse-to-fineなSemantic IDを作る

まず、内容 understanding modelがアイテムの内容をdense embedding `x`へ変換します。次にResidual Quantized VAE（RQ-VAE）が`x`をlatent表現`z`へencodeし、残差を複数段のコードブックで順に量子化します。

各層`l`は、その時点までに近似し切れなかった残差`r_l`に最も近いコードブック vectorを選び、離散コード`c_l`を出します。Semantic IDは、このコードを並べた`(c1, c2, ..., cL)`です。

たとえば`L = 3`なら、論文は階層を次のように説明しています。

```text
(c1)       : foodに関する広告
(c1, c2)   : pizzaに関する広告
(c1, c2,c3): pizzaかつ英語で書かれた広告
```

これは説明用の例で、各コードに人間が直接「food」「pizza」とラベルを付けるわけではありません。前段ほど粗い情報を、後段ほど残差に含まれる細かな情報を表す、というRQ-VAEの構造を直感的に示したものです。

論文のオフライン実験では、過去3か月のtarget itemの内容 embeddingから`L = 3`、各コードブック size `K = 2048`のRQ-VAEを学習し、損失中の係数`β`を0.5に設定しています。本番環境では`L = 6`、`K = 2048`を使います。これらはMetaのデータとシステムで採用された設定であり、別環境での最適値ではありません。

## 中核：prefix n-gramが階層を埋め込みテーブルへ残す

Semantic IDを作っただけでは、ranking modelが参照するembedding rowへどう写すかが決まりません。ここで論文が提案するのがSemantic ID prefix n-gramです。

Semantic IDが`(c1, c2, c3)`のとき、prefix 3-gramは次の3トークンを作り、それぞれに対応する埋め込みをsum-poolingします。

```text
token 1: (c1)
token 2: (c1, c2)
token 3: (c1, c2, c3)

item embedding = E[(c1)] + E[(c1,c2)] + E[(c1,c2,c3)]
```

これにより、完全に同じSemantic IDを持つアイテムだけでなく、粗いprefixだけが一致するアイテムもパラメータを共有できます。新しい英語pizza広告は、同じ細粒度clusterに学習例がなくても、foodやpizzaという上位clusterの埋め込みを利用できます。

一方、単一のtrigramとして`(c1,c2,c3)`全体を1 rowへ変換すると、上位prefixの共有構造が表に現れません。隣接する全bigramも階層の全prefixを保持するわけではありません。論文のtoken parameterization比較では、階層を明示的に残すprefix n-gramが最も良いtrain NEを示しました。

| RQ-VAE設定 | parameterization | Train NE gain |
| --- | --- | ---: |
| `K=2048, L=3` | trigram | -0.028% |
| `K=2048, L=4` | all bigrams | -0.091% |
| `K=2048, L=3` | prefix 3-gram | -0.141% |
| `K=2048, L=5` | prefix 5-gram | -0.208% |
| `K=2048, L=6` | prefix 6-gram | -0.215% |

評価指標はNormalized Entropy（NE）で、モデルのcross-entropyをラベルの平均頻度だけを予測する比較手法のcross-entropyで正規化したものです。低いほど良く、表の負値はNEの改善を表します。深くするほど改善していますが、prefix 5-gramから6-gramの差は`-0.007` percentage pointです。深さとcardinalityを増やせば常に費用対効果が高い、とまでは言えません。

また、Semantic ID側のcardinalityがembedding table sizeを超える場合、論文の実装でもmodulo hashを適用します。random collisionが完全になくなる方式ではなく、階層prefixによる意味の共有と限られたtable容量を組み合わせた設計です。

## オフライン評価：効果はtailとnew itemで大きい

オフライン評価には、Metaの本番環境広告ranking modelを簡略化したDLRM系モデルを使います。dense featureとユーザーのitem interaction 履歴は残し、約100個ある他のsparse featureを外してtarget itemだけをsparse moduleへ入れています。4日分のproduction interaction dataを時系列順に1エポック学習し、翌日の最初の6時間で評価しています。

比較対象は次の3つです。

Individual Embeddings（IE）：各元のIDに専用rowを割り当てる。本番規模では非現実的で、未見アイテムは未学習のrandom embeddingになります。Random Hashing（RH）：元のIDを無作為にrowへ割り当てる。

Semantic ID（SemID）：multimodal 内容 embeddingをRQ-VAEとprefix 3-gramでrowへ割り当てる。


RHとSemIDは平均collision factorを3に揃えています。したがって、SemIDが単により大きなtableを使った比較ではありません。

| 評価区分 | RHのNE | IEのNE | SemIDのNE | SemID gain vs. RH |
| --- | ---: | ---: | ---: | ---: |
| Head | 0.80105 | 0.80101 | 0.80108 | 0.00% |
| Torso | 0.83589 | 0.83583 | 0.83580 | -0.01% |
| Tail | 0.83904 | 0.83886 | 0.83872 | -0.04% |
| 学習期間に出現したアイテム | 0.82626 | 0.82612 | 0.82600 | -0.03% |
| 評価時に初めて出現したアイテム | 0.83524 | 0.83453 | 0.83180 | -0.41% |
| 全アイテム | 0.82663 | 0.82645 | 0.82621 | -0.05% |

Headでは中立、Torsoでは小幅、Tailではより大きな改善となり、最大の差はnew itemに出ています。SemIDはnew itemに対しても、意味の近い既存アイテムが更新してきたprefix embeddingを使えます。これがRH比`-0.41%`、IE比`-0.33%`のNE gainにつながった、というのが著者らの解釈です。

ただし、これはprivateな広告データ上の結果です。区分ごとのサンプル数、run間の分散、信頼区間、統計的検定は示されていません。小さな差を他データセットへそのまま一般化はできません。

## 時間変化への強さ：長期間学習しても意味を保ちやすい

論文は、学習終盤の6時間と、その42時間前からの6時間でNEの差を取り、古い時点へのfitがどれだけ失われたかを調べています。全アイテムでの差はRHが`0.0083`、IEが`0.0074`、SemIDが`0.0073`で、SemIDはIEと同程度、RHより小さくなりました。

さらに、4日学習から20日学習へ延ばしたときのEval NE gainはRHが`-0.18%`、SemIDが`-0.23%`でした。両方ともデータ追加の恩恵を受けていますが、SemIDの方が長い履歴からやや大きな改善を得ています。

これは「Semantic IDが時間に対して不変」と証明した結果ではありません。コンテンツ上の大分類が元のIDより長く残るという仮説と、限られた期間のMeta data上の観測が整合した、と読むべきです。RQ-VAE自体をいつ再学習するか、コードブック更新時にID互換性をどう保つかは論文で詳しく扱われていません。

## ユーザー履歴ではattentionとの組み合わせが有効

Semantic IDはtarget itemだけでなく、ユーザーが過去に触れたアイテム列にも使われます。論文は長さ`O(100)`の履歴を、文脈化しないBypass、Transformer、Pooled Multihead Attention（PMA）の3方式で集約しました。

各方式でRHをSemIDへ置き換えたEval NE gainは次の通りです。

| 履歴 aggregation | Eval NE gain |
| --- | ---: |
| Bypass | -0.085% |
| Transformer | -0.110% |
| PMA | -0.100% |

文脈化するTransformerとPMAの改善がBypassより大きくなっています。1,000件の評価exampleでattentionを分析すると、SemID modelはpaddingへのattentionとentropyが低く、系列先頭の最新アイテムへのattentionが高い傾向を示しました。

ここから言えるのは、意味のある共有表現がattention moduleにとって使いやすいシグナルになった可能性です。一方、attention weightはそれ自体が因果的な説明ではなく、1,000件というsubsetでの診断指標です。「なぜ予測が改善したか」を完全に証明する結果ではありません。

## Meta Adsでのproduction pipeline

MetaはSemantic ID featureを論文執筆時点ですでに1年以上本番で運用していたと報告しています。本番環境構成は、オフライン学習とonline servingを分離しています。

```text
[offline]
過去3か月の広告content embedding
  → RQ-VAEを学習（L=6, K=2048）
  → checkpointをfreeze

[ad作成時]
text・image・video
  → content understanding model
  → frozen RQ-VAE
  → Semantic IDをEntity Data Storeへ保存

[ranking request時]
target itemとuser historyのraw ID
  → 保存済みSemantic IDでfeature enrichment
  → downstream ranking model
```

本番環境ではprefix 5-gramを使い、embedding table sizeは`O(50M)`です。テキスト・image・videoなど異なる内容 embedding 参照元から6つのsparse featureと1つのsequential featureを作っています。

flagship広告ranking modelのオフライン評価では、6 sparse featureの追加でEval NE `-0.071%`、1 sequential featureの追加で`-0.123%`でした。論文によれば、Meta Adsでは`0.02%`を超えるoffline NE gainをsignificantと見なしています。ただし、この「significant」が統計的有意性を意味するのか、社内運用上の実質的な基準なのかは明記されていません。

複数の広告ranking modelへ展開したオンラインのtop-line metricでは、全体で0.15%のperformance gainを報告しています。これはNEではなく、指標名、制御、トラフィック量、実験期間、信頼区間も非公開です。大規模で高度に最適化されたMeta Adsにおいて著者らが重要と判断した本番環境結果であり、一般的なCTR改善率として読むことはできません。

## 予測の安定性：同じ広告なのにscoreが変わる問題

random hashingでは、内容が同一の広告を別元のIDで複製すると異なるembedding rowへ入ります。そのため、同じ内容でも予測値やdeliveryが変わるA/A varianceが生じます。

論文はshadow ads実験で、A/A pairの予測差を両者の平均的な予測値で正規化したAARを測定しました。6つのSemantic ID sparse featureを加えた本番モデルは、加えない同じモデルと比べて平均AARを43%削減しています。これは0.15%のtop-line performance gainとは別の、予測安定性に関する相対改善です。

またオンラインA/Bテストでは、推薦集合のアイテムを50%の確率で同じSemantic ID prefixを持つ別アイテムへ入れ替え、CTRの変化を測っています。prefixが深くなるほどclick loss rateが単調に小さくなったため、細かいsemantic similarityほどprediction similarityと相関する、と著者らは結論づけています。Figureから精密な絶対値は読み取れないため、ここでは傾向だけを記載します。

この結果も「意味が似ていればユーザー反応が同じ」ことを保証しません。価格、ブランド、品質、在庫、creativeの微差など、内容 embeddingが十分に捉えない要因で反応は変わり得ます。論文自身もuser behaviorはsemanticsに対して単純に連続ではないと注意しています。

## 導入を検討するときの実装手順

内容 understanding model、広告データ、RQ-VAE code、ranking model、feature storeは公開されていないため、論文の完全再現はできません。以下は論文の本番環境手順そのものではなく、公開情報から導いた小規模な検証案です。

### 1. Raw ID問題を区分別に測る

まず、item cardinality、embedding table size、collision factor、アイテム寿命を記録します。全体指標だけでなく、impression数によるヘッド／torso／tail、学習時のseen／unseen、item ageで評価を分けます。SemIDの利点が全区分で均一とは限りません。

### 2. Content embeddingの品質を先に確かめる

業務上同じ意味と見なすアイテムが近く、区別すべきアイテムが離れているかをretrieval testで確認します。コンテンツエンコーダーのバイアスや欠落はRQ-VAEで直らず、そのまま意味のある「つもり」の誤衝突になります。

### 3. RQ-VAEとparameterizationを別々にablationする

同じ内容 embeddingとtable 予算で、少なくとも次を比較します。

```text
raw ID + random hashing
raw ID + individual embedding（小規模dataでの上限参考値）
Semantic ID + flat full-code token
Semantic ID + all bigrams
Semantic ID + prefix n-gram
```

`K`、`L`、prefix depth、table size、collision factorを同時に変えると原因が分からなくなります。まずtable 予算と学習データを揃え、parameterizationだけを比較します。

### 4. Random splitだけでなく時間分割を使う

ID driftingへの効果を見るには、未来の期間をテストにし、new itemを分離します。NEやlog lossに加え、AUC、calibration、区分別指標、学習期間を延ばしたときのgainを確認します。複数seedで分散と信頼区間も出します。

### 5. オフライン生成から始める

リクエストごとにコンテンツをencodeすると遅延とコストが増えます。論文のようにアイテム作成・更新時にSemantic IDを事前計算し、バージョン付きでfeature storeへ保存する構成が扱いやすいでしょう。

```text
semantic_id_version
content_encoder_version
rqvae_checkpoint_version
generated_at
prefix_tokens
fallback_raw_id_hash
```

参照失敗、unsupported 内容、encoder timeoutにはraw ID hashなどのfallbackを残します。新旧RQ-VAEのdual writeとshadow readを行えば、コードブック切り替え前にcoverageとprediction差を確認できます。

### 6. 品質とstabilityを別のguardrailにする

平均NEだけでなく、同一コンテンツの別ID、軽微なcreative変更、rare itemに対するscore差を測ります。オンラインではCTRやconversionに加え、p95／p99 latency、特徴量欠損率、Semantic ID coverage、cluster occupancy、embedding update量、A/A varianceを監視します。

## 制約と適用しにくい条件

この研究には、解釈と再現の両面で制約があります。

arXiv v1のプレプリントであり、査読済みvenueは記載されていない。データ、コンテンツモデル、RQ-VAE、ranking model、プロンプトやtraining codeは非公開で、第三者が同条件を再現できません。

オフライン実験は約100個のsparse featureを除いた簡略モデルであり、本番モデルと同一ではありません。区分ごとのサンプル数、複数runの分散、信頼区間、統計的検定が示されていない。

オンライン`0.15%` gainの指標名と実験条件が公開されていない。RQ-VAEの再学習頻度、コードブック migration、古いSemantic IDとの互換性が詳述されていない。

コンテンツが乏しいアイテム、意味より価格や鮮度が重要なドメイン、意図的に似せたspamでは、semantic clusterが良い共有単位にならない可能性があります。コンテンツエンコーダー由来のバイアス、公平性、プライバシー、adversarial manipulationへの影響は評価されていない。


また、人気アイテムが更新した共有埋め込みをtailへ渡すことはコールドスタートを助ける一方、人気側のバイアスをtailへ広げる可能性もあります。cluster単位だけでなく、item popularity、言語、地域、広告主規模などのsliceで悪化がないか確認する必要があります。

## まとめ

Semantic ID prefix n-gramの本質は、hash collisionを単に減らすことではありません。限られたembedding 予算の中で、パラメータを共有する相手を無作為な元のIDからコンテンツ上の近傍へ変えることです。

RQ-VAEが内容 embeddingをcoarse-to-fineな離散コードへ変換します。prefix n-gramが上位から下位までの階層をembedding parameterとして残す。

ヘッドで学んだシグナルをtailやnew itemへ共有し、コールドスタートを緩和します。元のIDが入れ替わってもsemantic prefixが残ることで、長期学習時の表現shiftを抑える。

ユーザー履歴のattention modelでも改善し、本番環境ではtop-line metric `+0.15%`、平均AAR `-43%`を報告した。


一方、Semantic IDは魔法のIDではありません。共有の質はコンテンツエンコーダーとquantizerに依存し、本番環境ではバージョン管理、fallback、drift監視が必要です。この論文から得られる実践的な示唆は、巨大tableの容量だけを増やす前に、どのアイテム同士なら学習を共有してよいかをID設計の問題として見直すことです。

## 参照

- Carolina Zheng et al., [Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID](https://arxiv.org/abs/2504.02137), arXiv:2504.02137v1, 2025-04-02（[PDF](https://arxiv.org/pdf/2504.02137)）。

本記事の数値、実験条件、production上の主張はこの論文に基づきます。「導入を検討するときの実装手順」は、公開情報をもとにした記事側の提案です。
