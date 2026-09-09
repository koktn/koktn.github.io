---
title: Semantic ID prefix n-gram――推薦モデルのhash衝突を「意味の共有」に変える
description: Metaの広告ranking論文をもとに、RQ-VAEによる階層Semantic ID、prefix n-gram、long-tail・cold start・予測安定性への効果とproduction導入上の注意点を解説します。
publishedAt: 2026-09-09
category: AI
tags:
  - Recommendation System
  - Semantic ID
  - Representation Learning
  - Vector Quantization
  - Meta
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Carolina Zhengらが2025年4月2日にarXivへ投稿した「[Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID](https://arxiv.org/abs/2504.02137)」です。Columbia UniversityとMetaの研究者によるpreprintで、Meta Adsのproduction systemへSemantic IDを導入した経験まで報告しています。

この論文の価値を一文でまとめると、**巨大で入れ替わりの激しいitem IDをランダムにembeddingへ押し込む代わりに、contentの意味から作った階層IDを共有させ、long-tail、cold start、時間変化に強い推薦表現をproduction規模で実現した**点にあります。

## 課題：random hashingは衝突相手の意味を選べない

大規模な推薦モデルは、広告や商品などのraw item IDをembedding tableのrowへ変換します。しかし、item数を `I`、用意できるrow数を `H` としたとき、productionではしばしば `I > H` です。すべてのitemへ専用rowを割り当てられないため、raw IDをhashして複数itemに同じembeddingを共有させます。

この方法はmemoryを抑えられる一方、同じrowへ入るitemの組み合わせはランダムです。pizza広告と旅行広告のように無関係なitemが衝突すると、それぞれの学習例が同じembeddingへ矛盾したgradient updateを加えます。

論文は、Meta Adsのitem分布にある3つの難しさを整理しています。

| 課題 | 論文中の観測 | random hashingへの影響 |
| --- | --- | --- |
| item cardinality | item数が実用的なembedding tableより大きい | 無関係なitem同士の衝突が避けられない |
| impression skew | 上位0.1%のitemが25%、次の5.5%が50%、残る94.4%が25%のimpressionを占める | tail itemは学習例が少なく、偶然の衝突相手から悪影響を受けやすい |
| ID drifting | 初期item集合の半分が6日後にはsystemから退出する | 同じrowが時間とともに別のitemを表し、学習済みの意味が変わる |

特に厄介なのは、古い広告が終了して似た内容の新しい広告が作られても、raw IDは別物になることです。専用embeddingなら新しいrowは未学習、random hashingなら無関係な既存rowから始まります。どちらも「似た内容についてすでに学んだこと」を自然には引き継げません。

## 全体像：contentを階層的な離散tokenへ変える

Semantic IDはraw IDそのものではなく、itemのtext・image・videoから得たcontent embeddingを離散化したIDです。論文のflowを単純化すると次のようになります。

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

意味が近いitemほど同じcode、または同じprefixを共有しやすくなります。衝突をなくすのではなく、**衝突するなら意味の近いitem同士にする**発想です。

## RQ-VAEがcoarse-to-fineなSemantic IDを作る

まず、content understanding modelがitemの内容をdense embedding `x` へ変換します。次にResidual Quantized VAE（RQ-VAE）が `x` をlatent表現 `z` へencodeし、残差を複数段のcodebookで順に量子化します。

各layer `l` は、その時点までに近似し切れなかった残差 `r_l` に最も近いcodebook vectorを選び、離散code `c_l` を出します。Semantic IDは、このcodeを並べた `(c1, c2, ..., cL)` です。

たとえば `L = 3` なら、論文は階層を次のように説明しています。

```text
(c1)       : foodに関する広告
(c1, c2)   : pizzaに関する広告
(c1, c2,c3): pizzaかつ英語で書かれた広告
```

これは説明用の例で、各codeに人間が直接「food」「pizza」とlabelを付けるわけではありません。前段ほど粗い情報を、後段ほど残差に含まれる細かな情報を表す、というRQ-VAEの構造を直感的に示したものです。

論文のオフライン実験では、過去3か月のtarget itemのcontent embeddingから `L = 3`、各codebook size `K = 2048` のRQ-VAEを学習し、loss中の係数 `β` を0.5に設定しています。productionでは `L = 6`、`K = 2048` を使います。これらはMetaのdataとsystemで採用された設定であり、別環境での最適値ではありません。

## 中核：prefix n-gramが階層をembedding tableへ残す

Semantic IDを作っただけでは、ranking modelが参照するembedding rowへどう写すかが決まりません。ここで論文が提案するのが**Semantic ID prefix n-gram**です。

Semantic IDが `(c1, c2, c3)` のとき、prefix 3-gramは次の3 tokenを作り、それぞれに対応するembeddingをsum-poolingします。

```text
token 1: (c1)
token 2: (c1, c2)
token 3: (c1, c2, c3)

item embedding = E[(c1)] + E[(c1,c2)] + E[(c1,c2,c3)]
```

これにより、完全に同じSemantic IDを持つitemだけでなく、粗いprefixだけが一致するitemもparameterを共有できます。新しい英語pizza広告は、同じ細粒度clusterに学習例がなくても、foodやpizzaという上位clusterのembeddingを利用できます。

一方、単一のtrigramとして `(c1,c2,c3)` 全体を1 rowへ変換すると、上位prefixの共有構造が表に現れません。隣接する全bigramも階層の全prefixを保持するわけではありません。論文のtoken parameterization比較では、階層を明示的に残すprefix n-gramが最も良いtrain NEを示しました。

| RQ-VAE設定 | parameterization | Train NE gain |
| --- | --- | ---: |
| `K=2048, L=3` | trigram | -0.028% |
| `K=2048, L=4` | all bigrams | -0.091% |
| `K=2048, L=3` | prefix 3-gram | -0.141% |
| `K=2048, L=5` | prefix 5-gram | -0.208% |
| `K=2048, L=6` | prefix 6-gram | -0.215% |

評価指標はNormalized Entropy（NE）で、モデルのcross-entropyをlabelの平均頻度だけを予測するbaselineのcross-entropyで正規化したものです。低いほど良く、表の負値はNEの改善を表します。深くするほど改善していますが、prefix 5-gramから6-gramの差は `-0.007` percentage pointです。深さとcardinalityを増やせば常に費用対効果が高い、とまでは言えません。

また、Semantic ID側のcardinalityがembedding table sizeを超える場合、論文の実装でもmodulo hashを適用します。random collisionが完全になくなる方式ではなく、階層prefixによる意味の共有と限られたtable容量を組み合わせた設計です。

## オフライン評価：効果はtailとnew itemで大きい

オフライン評価には、Metaのproduction広告ranking modelを簡略化したDLRM系modelを使います。dense featureとuserのitem interaction historyは残し、約100個ある他のsparse featureを外してtarget itemだけをsparse moduleへ入れています。4日分のproduction interaction dataを時系列順に1 epoch学習し、翌日の最初の6時間で評価しています。

比較対象は次の3つです。

- **Individual Embeddings（IE）**：各raw IDに専用rowを割り当てる。production規模では非現実的で、未見itemは未学習のrandom embeddingになる
- **Random Hashing（RH）**：raw IDをrandomにrowへ割り当てる
- **Semantic ID（SemID）**：multimodal content embeddingをRQ-VAEとprefix 3-gramでrowへ割り当てる

RHとSemIDは平均collision factorを3に揃えています。したがって、SemIDが単により大きなtableを使った比較ではありません。

| 評価segment | RHのNE | IEのNE | SemIDのNE | SemID gain vs. RH |
| --- | ---: | ---: | ---: | ---: |
| Head | 0.80105 | 0.80101 | 0.80108 | 0.00% |
| Torso | 0.83589 | 0.83583 | 0.83580 | -0.01% |
| Tail | 0.83904 | 0.83886 | 0.83872 | -0.04% |
| 学習期間に出現したitem | 0.82626 | 0.82612 | 0.82600 | -0.03% |
| 評価時に初めて出現したitem | 0.83524 | 0.83453 | 0.83180 | -0.41% |
| 全item | 0.82663 | 0.82645 | 0.82621 | -0.05% |

Headでは中立、Torsoでは小幅、Tailではより大きな改善となり、最大の差はnew itemに出ています。SemIDはnew itemに対しても、意味の近い既存itemが更新してきたprefix embeddingを使えます。これがRH比 `-0.41%`、IE比 `-0.33%` のNE gainにつながった、というのが著者らの解釈です。

ただし、これはprivateな広告data上の結果です。segmentごとのsample数、run間の分散、confidence interval、統計的検定は示されていません。小さな差を他datasetへそのまま一般化はできません。

## 時間変化への強さ：長期間学習しても意味が崩れにくい

論文は、学習終盤の6時間と、その42時間前からの6時間でNEの差を取り、古い時点へのfitがどれだけ失われたかを調べています。全itemでの差はRHが `0.0083`、IEが `0.0074`、SemIDが `0.0073` で、SemIDはIEと同程度、RHより小さくなりました。

さらに、4日学習から20日学習へ延ばしたときのEval NE gainはRHが `-0.18%`、SemIDが `-0.23%` でした。両方ともdata追加の恩恵を受けていますが、SemIDの方が長い履歴からやや大きな改善を得ています。

これは「Semantic IDが時間に対して不変」と証明した結果ではありません。content上の大分類がraw IDより長く残るという仮説と、限られた期間のMeta data上の観測が整合した、と読むべきです。RQ-VAE自体をいつ再学習するか、codebook更新時にID互換性をどう保つかは論文で詳しく扱われていません。

## User historyではattentionとの組み合わせが効く

Semantic IDはtarget itemだけでなく、userが過去に触れたitem列にも使われます。論文は長さ `O(100)` のhistoryを、文脈化しないBypass、Transformer、Pooled Multihead Attention（PMA）の3方式で集約しました。

各方式でRHをSemIDへ置き換えたEval NE gainは次の通りです。

| history aggregation | Eval NE gain |
| --- | ---: |
| Bypass | -0.085% |
| Transformer | -0.110% |
| PMA | -0.100% |

文脈化するTransformerとPMAの改善がBypassより大きくなっています。1,000件の評価exampleでattentionを分析すると、SemID modelはpaddingへのattentionとentropyが低く、sequence先頭の最新itemへのattentionが高い傾向を示しました。

ここから言えるのは、意味のある共有表現がattention moduleにとって使いやすいsignalになった可能性です。一方、attention weightはそれ自体が因果的な説明ではなく、1,000件というsubsetでの診断指標です。「なぜ予測が改善したか」を完全に証明する結果ではありません。

## Meta Adsでのproduction pipeline

MetaはSemantic ID featureを論文執筆時点ですでに1年以上production運用していたと報告しています。production構成は、offline学習とonline servingを分離しています。

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

productionではprefix 5-gramを使い、embedding table sizeは `O(50M)` です。text・image・videoなど異なるcontent embedding sourceから6つのsparse featureと1つのsequential featureを作っています。

flagship広告ranking modelのオフライン評価では、6 sparse featureの追加でEval NE `-0.071%`、1 sequential featureの追加で `-0.123%` でした。論文によれば、Meta Adsでは `0.02%`を超えるoffline NE gainをsignificantと見なしています。ただし、この「significant」が統計的有意性を意味するのか、社内運用上の実質的な基準なのかは明記されていません。

複数の広告ranking modelへ展開したonlineのtop-line metricでは、全体で**0.15%のperformance gain**を報告しています。これはNEではなく、metric名、control、traffic量、実験期間、confidence intervalも非公開です。大規模で高度に最適化されたMeta Adsにおいて著者らが重要と判断したproduction結果であり、一般的なCTR改善率として読むことはできません。

## 予測の安定性：同じ広告なのにscoreが変わる問題

random hashingでは、内容が同一の広告を別raw IDで複製すると異なるembedding rowへ入ります。そのため、同じ内容でも予測値やdeliveryが変わるA/A varianceが生じます。

論文はshadow ads実験で、A/A pairの予測差を両者の平均的な予測値で正規化したAARを測定しました。6つのSemantic ID sparse featureを加えたproduction modelは、加えない同じmodelと比べて**平均AARを43%削減**しています。これは0.15%のtop-line performance gainとは別の、予測安定性に関する相対改善です。

またonline A/B testでは、推薦集合のitemを50%の確率で同じSemantic ID prefixを持つ別itemへ入れ替え、CTRの変化を測っています。prefixが深くなるほどclick loss rateが単調に小さくなったため、細かいsemantic similarityほどprediction similarityと相関する、と著者らは結論づけています。Figureから精密な絶対値は読み取れないため、ここでは傾向だけを記載します。

この結果も「意味が似ていればuser反応が同じ」ことを保証しません。価格、brand、品質、在庫、creativeの微差など、content embeddingが十分に捉えない要因で反応は変わり得ます。論文自身もuser behaviorはsemanticsに対して単純に連続ではないと注意しています。

## 導入を検討するときの実装手順

content understanding model、広告data、RQ-VAE code、ranking model、feature storeは公開されていないため、論文の完全再現はできません。以下は論文のproduction手順そのものではなく、公開情報から導いた小規模な検証案です。

### 1. Raw ID問題をsegment別に測る

まず、item cardinality、embedding table size、collision factor、item寿命を記録します。全体metricだけでなく、impression数によるhead／torso／tail、学習時のseen／unseen、item ageで評価を分けます。SemIDの利点が全segmentで均一とは限りません。

### 2. Content embeddingの品質を先に確かめる

業務上同じ意味と見なすitemが近く、区別すべきitemが離れているかをretrieval testで確認します。content encoderのbiasや欠落はRQ-VAEで直らず、そのまま意味のある「つもり」の誤衝突になります。

### 3. RQ-VAEとparameterizationを別々にablationする

同じcontent embeddingとtable budgetで、少なくとも次を比較します。

```text
raw ID + random hashing
raw ID + individual embedding（小規模dataでの上限参考値）
Semantic ID + flat full-code token
Semantic ID + all bigrams
Semantic ID + prefix n-gram
```

`K`、`L`、prefix depth、table size、collision factorを同時に変えると原因が分からなくなります。まずtable budgetと学習dataを揃え、parameterizationだけを比較します。

### 4. Random splitだけでなく時間splitを使う

ID driftingへの効果を見るには、未来の期間をtestにし、new itemを分離します。NEやlog lossに加え、AUC、calibration、segment別metric、学習期間を延ばしたときのgainを確認します。複数seedで分散とconfidence intervalも出します。

### 5. Offline生成から始める

requestごとにcontentをencodeするとlatencyとcostが増えます。論文のようにitem作成・更新時にSemantic IDを事前計算し、version付きでfeature storeへ保存する構成が扱いやすいでしょう。

```text
semantic_id_version
content_encoder_version
rqvae_checkpoint_version
generated_at
prefix_tokens
fallback_raw_id_hash
```

lookup失敗、unsupported content、encoder timeoutにはraw ID hashなどのfallbackを残します。新旧RQ-VAEのdual writeとshadow readを行えば、codebook切り替え前にcoverageとprediction差を確認できます。

### 6. Qualityとstabilityを別のguardrailにする

平均NEだけでなく、同一contentの別ID、軽微なcreative変更、rare itemに対するscore差を測ります。onlineではCTRやconversionに加え、p95／p99 latency、feature欠損率、Semantic ID coverage、cluster occupancy、embedding update量、A/A varianceを監視します。

## Limitationと適用しにくい条件

この研究には、解釈と再現の両面で制約があります。

- arXiv v1のpreprintであり、査読済みvenueは記載されていない
- data、content model、RQ-VAE、ranking model、promptやtraining codeは非公開で、第三者が同条件を再現できない
- オフライン実験は約100個のsparse featureを除いた簡略modelであり、production modelと同一ではない
- segmentごとのsample数、複数runの分散、confidence interval、統計的検定が示されていない
- online `0.15%` gainのmetric名と実験条件が公開されていない
- RQ-VAEの再学習頻度、codebook migration、古いSemantic IDとの互換性が詳述されていない
- contentが乏しいitem、意味より価格や鮮度が重要なdomain、意図的に似せたspamでは、semantic clusterが良い共有単位にならない可能性がある
- content encoder由来のbias、公平性、privacy、adversarial manipulationへの影響は評価されていない

また、人気itemが更新した共有embeddingをtailへ渡すことはcold startを助ける一方、人気側のbiasをtailへ広げる可能性もあります。cluster単位だけでなく、item popularity、言語、地域、広告主規模などのsliceで悪化がないか確認する必要があります。

## まとめ

Semantic ID prefix n-gramの本質は、hash collisionを単に減らすことではありません。**限られたembedding budgetの中で、parameterを共有する相手をrandomなraw IDからcontent上の近傍へ変える**ことです。

- RQ-VAEがcontent embeddingをcoarse-to-fineな離散codeへ変換する
- prefix n-gramが上位から下位までの階層をembedding parameterとして残す
- headで学んだsignalをtailやnew itemへ共有し、cold startを緩和する
- raw IDが入れ替わってもsemantic prefixが残ることで、長期学習時の表現shiftを抑える
- user historyのattention modelでも改善し、productionではtop-line metric `+0.15%`、平均AAR `-43%`を報告した

一方、Semantic IDは魔法のIDではありません。共有の質はcontent encoderとquantizerに依存し、productionではversion管理、fallback、drift監視が必要です。この論文から得られる実践的な示唆は、巨大tableの容量だけを増やす前に、**どのitem同士なら学習を共有してよいか**をID設計の問題として見直すことです。

## 参照

- Carolina Zheng et al., [Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID](https://arxiv.org/abs/2504.02137), arXiv:2504.02137v1, 2025-04-02（[PDF](https://arxiv.org/pdf/2504.02137)）。

本記事の数値、実験条件、production上の主張はこの論文に基づきます。「導入を検討するときの実装手順」は、公開情報をもとにした記事側の提案です。
