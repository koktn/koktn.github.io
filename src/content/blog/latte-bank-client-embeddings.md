---
title: LLMを学習時の教師にするLATTE：銀行取引系列の意味を軽量モデルへ移す
description: 長い銀行取引履歴をLLMへ直接入力せず、統計要約から生成した文章と取引系列を対照学習でそろえるLATTEを、精度、推論速度、再現性から解説します。
publishedAt: 2026-10-04
category: AI
tags:
  - LLM
  - Representation Learning
  - Contrastive Learning
  - Financial Data
  - 論文解説
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文と著者公開コードを確認して記載しています。利用時は原文も確認してください。

この研究の価値は、LLMの知識を使う場所をオンライン推論から学習時へ移したことにあります。取引履歴を毎回LLMへ読ませる代わりに、顧客の統計要約を文章へ変換し、その意味を軽量な系列エンコーダへ対照学習で移します。

取り上げるのは、Egor Fadeevらによる「[LATTE: Learning Aligned Transactions and Textual Embeddings for Bank Clients](https://aclanthology.org/2025.emnlp-industry.179/)」です。EMNLP 2025 Industry Trackに採択された査読済み論文で、2025年11月に公開されました。本稿では[論文PDF](https://aclanthology.org/2025.emnlp-industry.179.pdf)と[著者公開コード](https://github.com/mathceo/latte)をもとに、仕組み、評価結果、実運用上の選択肢、再現可能な範囲を整理します。

## 長い取引履歴をLLMへ直接渡しにくい理由

銀行の顧客履歴には、取引時刻、金額、merchant category、取引種別などが時系列で並びます。論文によると、履歴の長さは公開データでも一人あたり数千イベント、銀行の内部データでは数百万イベントに達する場合があります。

各イベントをテキストへ変換してLLMへ渡す方法には、三つの問題があります。履歴が長くなるほど入力トークンが増え、context windowを圧迫します。推論時間とGPU memoryも増加します。さらに、業務で必要な出力は自由文とは限りません。churn、credit scoring、targetingなどの分類・回帰が中心です。

一方、GRUやTransformerで取引系列だけを自己教師あり学習すれば、軽量な顧客embeddingを作れます。ただし、merchant categoryの意味や行動パターンの解釈は、構造化された値だけから学びにくい場合があります。

LATTEは、この二つに役割を分けます。LLMは統計要約を短い文章へ変換します。系列モデルは生の取引順序を学び、文章embeddingへ近づくように追加学習します。

## 統計、文章、取引系列を同じ空間へそろえる

LATTEの学習は、統計要約から文章を作る経路と、生の取引系列をembeddingへ変える経路に分かれます。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/latte-training-inference-mobile.svg">
    <img src="/img/posts/latte-training-inference.svg" alt="顧客の取引履歴から統計要約と生系列の二経路を作り、文章embeddingと系列embeddingを対照学習で整列した後、LATTEとLATTE-Sへ分かれる処理" loading="lazy">
  </picture>
  <figcaption>図1：<a href="https://aclanthology.org/2025.emnlp-industry.179.pdf">原論文Figure 2・§3</a>を基に、学習時と推論時の違いが分かるよう本記事で再構成した独自図。矢印はデータまたはembeddingの受け渡しを示します。原論文は<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>です。</figcaption>
</figure>

図1の上段が学習時です。まず顧客`i`の取引系列`T_i`から、利用頻度、merchant categoryの多様性、取引期間、収支構造などの統計`s_i`を計算します。論文付録の例では、取引件数、活動日数、1日あたりの取引数、主要MCC、収入・支出合計などをプロンプトへ入れています。

instruction-tuned LLMは、統計値を行動の説明文`d_i`へ変換します。続いて、固定したtext encoderが文章から`z_i^text`を作ります。主実験ではQwen3-Embedding-8Bを使いました。

もう一方の経路では、生の取引系列をGRU-based encoderへ入力します。このencoderはCoLES objectiveで事前学習されています。同じ顧客から切り出した重複部分系列を近づけ、別顧客の系列を離す学習方法です。出力は系列embedding`z_i^seq`です。

最後に、同一顧客の`z_i^text`と`z_i^seq`が近くなるように対照学習します。text encoderを固定し、取引系列側だけを更新する方法です。文章は推論結果として使わず、取引系列のembeddingへ意味を与える教師信号にします。

### 二つの対照学習head

論文は二種類のheadを比較しています。

LATTE [1]は、CLIPと同じ発想のsymmetric softmax lossを使います。batch内で、系列から正しい文章を選ぶ方向と、文章から正しい系列を選ぶ方向を平均します。

```text
L_softmax = 1/2 × (L_seq→text + L_text→seq)
```

LATTE [2]は、共有情報と取引系列固有の情報を別の部分空間へ分けます。二つの表現の内積に罰則を加え、文章へ合わせる情報と、取引系列だけが持つ情報の両方を残そうとする設計です。

```text
L_reg = L_softmax + λ_ortho × || Z_shared^T Z_spec ||²_F
```

二つのheadに明確な勝者はいません。churnではLATTE [2]、age groupとgenderではLATTE [1]が最良でした。用途ごとに検証が必要です。

## LATTEとLATTE-Sは推論コストが違う

論文の`LATTE`と`LATTE-S`は、推論時に使う情報が異なります。

| 方式 | 推論時の入力 | 出力に使う情報 | 特徴 |
| --- | --- | --- | --- |
| LATTE-S | 生の取引系列 | 整列済み系列embedding | LLMとtext encoderを推論経路から外せる |
| LATTE | 生の取引系列、統計要約から得た文章 | 系列embeddingとtext側表現の連結 | 精度は高いが、文章生成とtext embeddingが必要 |

LATTE-Sは、学習時に文章から得た意味を系列encoderへ移し、推論では系列encoderだけを使います。論文が報告する「数百万parameter」「160〜200 samples/sec超」という軽量性は、主にLATTE-Sの結果です。

完全版LATTEは、系列embeddingと、文章embeddingへ数値統計を加えた表現を連結します。三つの下流taskで最も高いscoreを出しましたが、推論時にも生成モデルを含む経路が必要です。「LATTE全体がLLM不要で高速」と読むのは正確ではありません。

## 3データセットでの評価方法

評価には、匿名化されたクレジットカード取引系列の公開データを三つ使っています。

| データセット | 顧客数 | 下流task | 指標 |
| --- | ---: | --- | --- |
| Churn | 約10,000 | 将来の非活動を予測 | ROC-AUC |
| Gender | 約15,000 | gender labelを予測 | ROC-AUC |
| Age Group | 約50,000 | 年齢層を分類 | Accuracy |

各データセットで、label付き顧客の10%をtest partitionとして分離します。残り90%とlabelなし顧客がembedding modelの学習対象です。embeddingの評価では、label付きtraining dataを5分割し、4 foldのembeddingでLightGBMを学習して残り1 foldを評価しました。表の`±`は、この5 foldの平均と標準偏差を表します。

この記述には注意が必要です。論文は10%のtest partitionを確保したと説明する一方、主結果を5-fold cross-validationの平均として報告しています。最終的な表が独立test partitionの評価なのか、training partition内のheld-out foldなのかは、本文だけでは明確ではありません。

比較対象には、集約統計、CPC、CoLES、NPPR、temporal point process、BERT由来のobjective、LLaMA 3.2 3Bを使うTALLRecとHKFRが含まれます。方式ごとに使う情報量とmodel sizeが異なるため、単一条件のarchitecture比較ではありません。

## 精度は3 taskすべてでbest baselineを上回った

主結果では、LATTEが三つのtaskすべてで最良でした。図2は、各taskで最も高かった非LATTE方式と、最良のLATTEを比べています。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/latte-main-results-mobile.svg">
    <img src="/img/posts/latte-main-results.svg" alt="Churn、Age Group、Genderについて最良の非LATTE baselineと最良のLATTEの評価値を比較した棒グラフ" loading="lazy">
  </picture>
  <figcaption>図2：<a href="https://aclanthology.org/2025.emnlp-industry.179.pdf">原論文Table 1</a>の平均値を本記事でchart化。ChurnとGenderはROC-AUC、Age GroupはAccuracyで、異なる指標を同じ軸に置いています。誤差棒は省略し、数値を併記しました。</figcaption>
</figure>

| Task | best non-LATTE baseline | best LATTE | 絶対差 |
| --- | ---: | ---: | ---: |
| Churn ROC-AUC | NPPR `0.845 ± 0.003` | LATTE [2] `0.872 ± 0.004` | `+0.027` |
| Age Group Accuracy | TALLRec `0.659 ± 0.004` | LATTE [1] `0.665 ± 0.005` | `+0.006` |
| Gender ROC-AUC | CoLES `0.882 ± 0.004` | LATTE [1] `0.900 ± 0.005` | `+0.018` |

差はtaskによって大きく異なります。churnの絶対差は2.7 percentage pointですが、age groupは0.6 pointです。論文はfold間の標準偏差を示すものの、手法間の差に対する統計検定や信頼区間は報告していません。

### Alignmentと連結は別々に寄与する

Ablationでは、文章だけ、未整列のCoLESとtext embeddingの連結、LATTE-S、完全版LATTEを比べています。

文章だけでは、Churn `0.772`、Age Group `0.432`、Gender `0.644`でした。統計要約から生成した文章だけで取引系列全体を置き換えるには情報が足りません。

未整列の`CoLES + z_text`は、それぞれ`0.863`、`0.650`、`0.890`まで上がりました。文章表現を連結するだけでも強いbaselineです。そこから対照学習を加えた完全版LATTE [1]は`0.869`、`0.665`、`0.900`となりました。文章を加える効果と、二つのembedding空間を整列する効果は分けて読む必要があります。

LATTE-Sは、ChurnではCoLESの`0.841`に対して`0.847`、Age Groupでは`0.644`に対して`0.657`、Genderでは`0.882`に対して`0.891`でした。完全版より低いものの、LLMを推論経路から外しても三つのtaskでCoLESを上回っています。

## 速度とmodel sizeのtrade-off

Gender taskでLATTE-S [1]はROC-AUC `0.891`、推論速度`162 samples/sec/GPU`でした。論文によると、TALLRecやESQAなどのLLM方式より14倍以上高速です。Age GroupでもLATTE-Sは`200 samples/sec`を超えたと報告されています。

一方、完全版LATTEやTALLRecなど、生成モデルを含む方式は数samples/secにとどまり、30億を超えるparameterを使います。LATTE-Sは数百万parameterで、CoLESなどの軽量baselineに近い速度です。

ただし、速度の読み方には制約があります。実験環境として8基のNVIDIA Tesla A100 80GBが記載されていますが、図の`samples/sec/GPU`を測ったbatch size、系列長、並列化、I/O、数値精度は本文にまとまっていません。別環境で同じthroughputが出るとは限りません。

## 生成文は統計をどこまで保ったか

LATTEでは、LLMが統計要約を誤って言い換えると教師信号もずれます。著者らは各データセットから200件を無作為抽出し、別のLlama 3.1 8Bで生成文から統計を取り出した後、rule-based matchingで元の値と照合しました。

取引期間は全生成文に現れ、正確さは`98.31〜99.14%`でした。取引があった日の比率は`93.22〜95.41%`の生成文で使われ、正確さは`92.38〜99.07%`です。上位MCCは言及率が`24.61〜39.66%`と低い一方、言及された場合の一致率は`100%`でした。

ここでいう正確さは、言及された統計値が元の値と一致した割合です。説明文全体が事実に忠実であることや、下流予測の理由として正しいことを保証する指標ではありません。評価にも別のLLMを使っているため、人手によるfactuality評価とも区別が必要です。

## Generatorとtext encoderの選択

付録ではgeneratorとtext encoderを変えています。generatorにはGemma 3 4B、Qwen 3 32B、Gemma 3 27Bを使いました。taskごとの最良modelは異なり、3 taskすべてで同じgeneratorが勝ったわけではありません。

text encoderはmE5-large-instruct、Qwen3-Embedding-0.6B、Qwen3-Embedding-8Bを比較しています。task間の差は0.5〜1.0 percentage pointの狭い範囲でした。最小のQwen3-Embedding-0.6BがGenderで最良だったため、text encoderを大きくすれば常に改善する結果ではありません。

生成modelのhidden stateを直接mean poolingする方式と、専用のsentence encoderを使う方式もほぼ同等でした。生成とembeddingを別modelへ分ける設計は必須条件ではなく、計算資源や実装の都合に応じて検証できる部分です。

## 公開コードで再現できる範囲

著者のGitHub repositoryには、説明文生成、embedding抽出、LATTE-S推論、対照学習、LightGBM評価のcodeと、三つのデータセット向け設定があります。設定にはseed `42`、説明文生成のtemperature `0.6`、Qwen3-Embedding-8B、batch size、学習率などが記載されています。

一方、2026年10月4日に確認したrepositoryだけで、論文の表をそのまま再現するのは難しい状態です。READMEの手順はRosbank scenarioの説明が中心で、データの取得と前処理を完結していません。`requirements.txt`やlockfile、Dockerfileはなく、依存versionも固定されていません。release tag、model checkpoint、生成済み文章、全実験を一括実行する手順も見当たりませんでした。repository内にlicense fileもないため、codeを再利用する場合は権利条件を著者へ確認する必要があります。

論文と同じ計算条件では、説明文生成から対照学習まで8基のA100 80GBを使っています。公開データは利用できても、第三者が同じmodel、依存関係、生成結果をそろえて数値を完全再現できるpackageではありません。

## 実運用へ持ち込むなら段階を分ける

以下は論文で実証された結果ではなく、公開された設計を使って小規模検証する場合の提案です。

最初にCoLESなどの系列encoderと、集約統計を入れたLightGBMを強いbaselineとして固定します。LATTE-Sの評価では、accuracyだけでなく、学習時のLLM生成費用、推論時のp50・p95 latency、GPU memory、embedding更新時間を同じ表へ記録します。

次に、統計要約とpromptをversion管理します。生成文は教師データの一部です。model名、weight、sampling設定、prompt、乱数seed、生成日時を保存してください。LLMを更新した場合は、同じ顧客について説明文の差分と下流指標を測ります。

個人情報を含む取引履歴を外部APIへ送らない構成も必要です。統計要約だけでも、収支、活動期間、主要merchant categoryを組み合わせると機微なprofileになります。access control、retention、logging、暗号化、利用目的を、系列データと生成文の両方へ適用します。

最後に、用途別の評価を追加します。論文のtaskにはgenderとage groupが含まれます。一方、fairness、subgroup別error、属性推定の妥当性、本人への影響は評価されていません。融資やrisk assessmentへの導入をembedding精度だけで判断するのは不十分です。適用地域の法務確認、人による審査、説明・異議申立て、従来modelへのrollbackを別途設計する必要があります。

## 論文の限界

著者らが挙げる第一の限界は、LLMへ渡す統計量を事前に決める点です。選んだ統計が重要な行動を表していなければ、文章にも系列embeddingにも意味を移せません。

第二に、生成文の品質はpromptと固定LLMのgeneralizationに依存します。text encoderも固定されるため、data distributionが変わるとalignmentの品質が下がる可能性があります。

第三に、LATTE-Sは推論時に軽量でも、学習用の説明文を大量生成する費用がかかります。labelやLLM fine-tuningは不要です。ただし、LLMを使わない軽量な自己教師あり学習よりtraining overheadは増えます。

実証範囲も金融取引の三つの公開データセットに限られます。医療、教育、e-commerceへの展開はfuture workであり、論文内で検証された結果ではありません。

本稿の観点では、評価splitの説明、速度測定の詳細、公開codeの再現手順も不足しています。また、genderやage groupを取引から推定することのfairnessとprivacyは、性能表とは別に検討が必要です。

## まとめ

LATTEは、長い取引履歴をLLMへ直接入力せず、統計要約から作った文章を教師信号に使います。文章embeddingと取引系列embeddingを対照学習でそろえ、LLMの意味情報を軽量な系列encoderへ移す方法です。

完全版LATTEは、Churn `0.872`、Age Group `0.665`、Gender `0.900`で、各taskのbest baselineを上回りました。LATTE-SもCoLESより高いscoreを保ちながら、Genderで`162 samples/sec/GPU`を報告しています。

この結果から得られる設計上の示唆は、LLMを常にserving経路へ置く必要はないということです。高価なmodelを学習時の意味教師として使い、オンラインでは小さなencoderだけを動かせます。

一方、統計量の選び方が情報の上限を決めます。説明文の生成費用、distribution shift、再現手順の不足、金融データのprivacyとfairnessも残る課題です。LATTEはLLMの費用をなくす手法ではありません。費用をオンライン推論からofflineの教師データ生成へ移し、精度と速度を選べるようにする枠組みです。

## 参照資料

- Fadeev et al., [LATTE: Learning Aligned Transactions and Textual Embeddings for Bank Clients](https://aclanthology.org/2025.emnlp-industry.179/), EMNLP 2025 Industry Track, pp. 2635–2647.
- [論文PDF](https://aclanthology.org/2025.emnlp-industry.179.pdf) — 手法は§3、主結果は§5、限界はp. 2641、実験詳細はAppendix A〜C。
- [著者公開コード](https://github.com/mathceo/latte) — 説明文生成、embedding抽出、対照学習、LATTE-S推論の実装。
