---
title: TIGER――Semantic IDを生成して商品を検索する推薦モデル
description: NeurIPS 2023のTIGERを、RQ-VAEによるSemantic ID、自己回帰検索、Amazon Reviewsでの評価、cold startと推論コストの限界から解説します。
publishedAt: 2026-09-13
category: AI
tags:
  - Recommendation System
  - Generative Recommendation
  - Semantic ID
  - Vector Quantization
  - NeurIPS
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値と手法は原論文を確認して記載していますが、利用時は原文も確認してください。

推薦システムの候補検索では、ユーザーと商品をベクトルに変換し、近似最近傍探索（ANN）で商品を探す構成が一般的です。[Shashank Rajputらの「Recommender Systems with Generative Retrieval」](https://arxiv.org/abs/2305.05065)は、商品を短い**Semantic ID**で表し、ユーザーの行動履歴から次に選ばれる商品のIDを生成する**TIGER**（Transformer Index for GEnerative Recommenders）を提案しました。対象はarXiv v3（2023年11月3日改訂）のNeurIPS 2023論文です。

この論文の価値は、**商品ごとのembeddingとANN indexを使う候補検索を、内容に基づく離散IDとTransformerの自己回帰予測に置き換え、そのIDの共有を新規商品への推薦にも利用した**点にあります。ただし、indexが完全に消えるわけではありません。生成したIDから商品を引くlookup tableは必要です。

## 何を生成するのか

従来型の候補検索では、ユーザー履歴から得たvectorと各商品のvectorを比較します。TIGERは、まず商品内容からSemantic IDを作り、履歴中の商品もこのID列へ変換します。Transformerのencoder-decoderが次商品のIDを左から右へ予測し、生成結果を商品へ引き戻します。

```text
商品内容 → Sentence-T5 → 768次元vector → RQ-VAE → Semantic ID
                                                   ↓
ユーザーID + 履歴商品のSemantic ID列 → Transformer → 次商品のSemantic ID
                                                   ↓
                                          lookup table → 推薦商品
```

たとえば似た靴を `(5, 25, 55)` と `(5, 25, 78)` のように表せれば、前半のtokenを共有できます。無関係な連番の商品IDと違い、内容の近い商品から学んだ情報を共有できる、というのが設計上の狙いです。Semantic IDは自然言語の文字列ではなく、**各段のcodebookから選んだ整数の組**です。

## Semantic IDの作り方：残差を順に量子化する

論文の実験では、商品タイトル、価格、ブランド、カテゴリを文にして事前学習済みSentence-T5へ入力し、768次元の内容vectorを得ます。RQ-VAEのencoderがこれを32次元の潜在vectorへ変換し、3段のresidual quantizationでtokenを選びます。各段は256個のvectorを持つ別々のcodebookです。

1段目は潜在vectorに最も近いcodewordを選びます。2段目は、元のvectorから1段目のcodewordを引いた**残差**を量子化します。3段目も同様です。粗い特徴を前段、細かい差を後段で表す構造なので、同じprefixを持つ商品は意味的に近くなりやすいと考えられます。Beauty datasetの可視化でも、1番目のtokenは大まかなカテゴリ、2番目はその内側の細分化と対応しました。ただし、これは実験上の定性的観察であり、あらゆる商品カテゴリで厳密な分類木になる保証ではありません。

3 tokenだけでは複数の商品が同じIDになることがあります。TIGERは衝突した商品へ4番目の識別tokenを追加し、衝突しない商品にも `0` を付けます。したがって、実験で使うIDは**「3個の意味的なtoken＋1個の衝突回避token」**からなる4 tokenです。最後のtokenまで意味階層として解釈するのは誤りです。IDを一意にするため、商品IDからSemantic ID、Semantic IDから商品IDへの対応表を保持します。[原論文の手法と実装設定](https://arxiv.org/pdf/2305.05065#page=4)

## 履歴から次のIDを生成する

ユーザーが触れた商品のIDを時系列に連結し、先頭にユーザーIDのtokenを置きます。生のユーザーIDはhashing trickで2,000種類のtokenへ写します。encoderがこの履歴を読み、decoderが次商品の4 tokenを自己回帰で出します。

```text
入力  : user_42, (5,25,78,0), (8,3,12,0), ...
出力  : (5,25,55,0)
予測  : P(c1 | 履歴) × P(c2 | 履歴,c1) × ... × P(c4 | 履歴,c1,c2,c3)
```

この例の数値は説明用です。論文のmodelはencoder・decoder各4層、約1,300万parameterで、top-K候補を得る際にはbeam searchを使います。token用のvocabularyは各段256種類の計1,024 tokenで、これにユーザーIDなどのtokenが加わります。学習対象は商品の**次回interaction**であり、説明文を生成するLLMではありません。[原論文のモデル設定](https://arxiv.org/pdf/2305.05065#page=6)

生成IDがcatalogに存在しない場合もあります。実験のtop-10では無効IDが約0.1〜1.6%と報告され、論文はbeamを大きくして無効IDを除外する方法を示しています。prefixが近い別商品へ置き換える案も挙げていますが、これは今後の課題であり、報告された評価結果に組み込まれた手法ではありません。[原論文のInvalid IDs節](https://arxiv.org/pdf/2305.05065#page=10)

## 評価：3カテゴリのoffline次商品予測

評価にはAmazon Product ReviewsのBeauty、Sports and Outdoors、Toys and Gamesを使います。1996年5月〜2014年7月のreviewをinteractionとして時刻順に並べ、5件未満のreviewしかないユーザーを除外します。各ユーザーの最後の商品をtest、直前をvalidation、残りをtrainingに使い、学習時の履歴は最大20商品です。指標はRecall@5／10とNDCG@5／10です。[原論文Appendix C](https://arxiv.org/pdf/2305.05065#page=15)

| Dataset | Users | Items | 平均履歴長 |
| --- | ---: | ---: | ---: |
| Beauty | 22,363 | 12,101 | 8.87 |
| Sports and Outdoors | 35,598 | 18,357 | 8.32 |
| Toys and Games | 19,412 | 11,924 | 8.63 |

Table 1のNDCG@10を、各datasetで最も強い比較手法と並べると次の通りです。

| Dataset | 最良の比較手法 | 比較手法のNDCG@10 | TIGERのNDCG@10 | 相対改善 |
| --- | --- | ---: | ---: | ---: |
| Sports and Outdoors | S3-Rec | 0.0204 | 0.0225 | +10.29% |
| Beauty | S3-Rec | 0.0327 | 0.0384 | +17.43% |
| Toys and Games | S3-Rec | 0.0376 | 0.0432 | +14.97% |

3 datasetすべてでTIGERが上回りました。特にBeautyのNDCG@5はSASRecの `0.0249` からTIGERの `0.0321` へ**29.04%の相対改善**です。ただしTable 1の多くのbaseline値はS3-Recの公開結果から取り、P5のみ前処理を変更して比較しています。Appendix Dによれば、元のP5前処理には連番IDをsplit前に割り当てることによる情報漏洩の懸念がありました。数字はこのoffline設定のもので、オンラインCTRや売上の改善を意味しません。[原論文Table 1とAppendix D](https://arxiv.org/pdf/2305.05065#page=7)

Semantic IDそのものの寄与を確かめるTable 2では、NDCG@10がBeautyでRandom ID `0.0250`、LSHで作ったSemantic ID `0.0309`、RQ-VAEのSemantic ID `0.0384`でした。decoderを使うだけで同じ性能になるわけではなく、**内容に基づくIDの作り方**が重要だと分かります。ただし、これはID生成法の比較で、RQ-VAEだけを本番システムに追加したときの効果ではありません。[原論文Table 2](https://arxiv.org/pdf/2305.05065#page=8)

## Cold startと多様性はどう評価されたか

新規商品への評価はBeautyで行われました。test商品から5%をtraining splitから除き、履歴を持たない「unseen item」を模擬します。RQ-VAEと推薦modelはtraining splitで学習し、その後にunseen itemへもSemantic IDを付けます。生成したIDと一致する既存商品に加え、**先頭3 tokenが一致するunseen item**を候補へ追加します。top-K中のunseen item割合を制限する `ε` を設け、`ε = 0.1` では内容vectorのKNN手法をRecall@Kで上回りました。

これは「学習していない商品を4 tokenまで正確に生成した」という意味ではありません。**共有prefixから候補を広げる仕組み**を加えた結果です。新商品の内容特徴を取得でき、同じencoderと量子化器でIDを付けられることが前提になります。[原論文のCold-Start Recommendation節](https://arxiv.org/pdf/2305.05065#page=9)

多様性の実験では、Beautyのdecoding temperatureを `1.0` から `2.0` へ上げると、推薦top-10のカテゴリ分布のEntropy@10が `0.76` から `1.38` へ上昇しました。これはカテゴリの散らばりが増えたことを示しますが、同じ設定での関連性とのtrade-offやユーザー満足度までは測っていません。[原論文Table 3](https://arxiv.org/pdf/2305.05065#page=9)

## 実装するときに確認したい境界

論文の設定を小規模に検証するなら、まず商品の内容特徴と時系列interactionを用意し、train／validation／testをユーザー単位の時間順に固定します。次に内容encoderとRQ-VAEでIDを作り、衝突率とcodebook利用率を測ります。固定したIDで次商品予測を学習し、beam search後に無効IDを除外してRecall・NDCGを測ります。新規商品については、通常評価と分けてprefix一致の候補拡張を評価する必要があります。この順序は原論文の構成から整理した検証案であり、論文が公開した実行手順そのものではありません。

導入上の注意は3点あります。

- **推論コスト**：ANNは不要でも、beam searchを伴う自己回帰生成は重くなります。著者らもANN方式より推論計算が高くなり得ると明記し、latency最適化は本論文の対象外としています。
- **catalogの更新**：modelのembedding tableは商品数に比例しませんが、商品とSemantic IDの双方向lookup tableは商品数とともに増えます。新商品を入れる際はID付与、衝突回避、対応表の更新を整合させる必要があります。
- **ID長**：長いIDは表現容量を増やせても、履歴のtoken数と生成stepを増やします。Appendix Eは6 token構成でも指標はおおむね頑健だったと述べる一方、計算費用の増加を認めています。

この論文は、約1.2万〜1.8万商品規模の3カテゴリにおけるoffline比較です。より大きなcatalogの運用負荷、オンライン効果、tail latencyはここからは判断できません。後続研究の[RPGについての記事](/posts/2026/09/09/rpg-parallel-semantic-id-recommendation/)では、TIGERの自己回帰生成を長いSemantic IDへ拡張するときの課題と、別の並列生成方式を扱っています。[原論文Appendix E](https://arxiv.org/pdf/2305.05065#page=16)

原典：[論文概要](https://arxiv.org/abs/2305.05065)／[PDF全文（arXiv v3）](https://arxiv.org/pdf/2305.05065)
