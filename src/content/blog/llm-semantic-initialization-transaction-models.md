---
title: LLMで取引モデルのembeddingを初期化する：merchantの意味を推論コストを増やさず取り込む
description: Visa Researchの論文をもとに、MCC・merchant・所在地の説明文からLLMのhidden stateを抽出し、取引系列モデルを初期化する方法と、10億件の取引での評価、改善の偏り、再現上の制約を解説します。
publishedAt: 2026-10-06
category: AI
tags:
  - LLM
  - Embedding
  - Financial Data
  - Sequential Modeling
  - 論文解説
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文の本文と図表を確認して記載しています。利用時は原文も確認してください。

この研究では、LLMから得た意味表現を取引モデルのembeddingの初期値に使い、オンライン推論でLLMを動かさずにmerchantや業種の知識を取り込みます。

対象はVisa ResearchのXiran Fanらによる「[Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings](https://aclanthology.org/2025.emnlp-industry.61/)」です。EMNLP 2025 Industry Trackの論文で、2025年11月の会議録に掲載されています。[論文PDF](https://aclanthology.org/2025.emnlp-industry.61.pdf)の§3で手法、§4で実験、p. 909で限界を説明しています。

LLMへ毎回取引履歴を読ませる設計ではありません。merchant category code（MCC）、merchant、所在地の説明をオフラインでベクトル化し、その値で既存の取引系列モデルを初期化して学習します。推論では、学習した取引モデルだけを使います。

10億件の取引を用いた評価では、次のmerchantを予測する精度が一貫して改善しました。一方、金額や都市の予測は設定によって悪化しています。非公開の業務指標では最大3.93%の相対改善を報告していますが、指標の絶対値や詳細な定義は開示されていません。

## IDだけではmerchantの意味を学びにくい

取引モデルでは、merchant名やMCCを整数IDへ変換し、IDごとのembeddingを学習する方法が使われます。embeddingは、カテゴリの特徴を表す学習可能なベクトルです。時刻や金額などの特徴と組み合わせ、過去の取引系列から次の取引や異常を予測します。

IDはカテゴリを区別できますが、名称や説明の意味を直接表しません。論文ではCostcoを例に挙げています。整数IDだけでは、卸売・小売、会員制といった知識をモデルへ渡せません。取引データから利用パターンを学ぶことはできても、名前が持つ知識は別途学ぶ必要があります。

MCCでも同じ問題があります。コードの数値だけから、業種や近いカテゴリを読み取るのは難しいからです。公式の業種説明や関連merchantを加えれば、取引頻度とは異なる情報を学習の出発点にできます。

そこで著者らは、ランダムに初期化していたカテゴリembeddingを、LLMの意味表現で置き換えます。変更するのは初期値です。IDによる参照と、その後の取引データを使った学習は残ります。意味情報と取引パターンを同じモデルへ取り込む設計です。

## オフラインで意味を作り、取引モデルを学習する

提案手法は、カテゴリの情報を集め、説明を補い、プロンプトを作り、LLMからembeddingを抽出する処理と、そのembeddingを使う取引モデルの学習に分かれます。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/transaction-semantic-initialization-flow-mobile.svg">
    <img src="/img/posts/transaction-semantic-initialization-flow.svg" alt="カテゴリの説明からオフラインでLLMのembeddingを作り、取引モデルを初期化して学習した後、推論では取引系列だけを入力する流れ" loading="lazy">
  </picture>
  <figcaption>図1。Fanらの<a href="https://aclanthology.org/2025.emnlp-industry.61.pdf">原論文Figure 1・§2・§3.4</a>を基に、初期化、学習、推論の関係を本記事で再構成した独自図。矢印はデータまたは重みの受け渡しを示します。原論文は<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>です。</figcaption>
</figure>

図1では、LLMを使う経路が取引モデルの初期化までに限られている点を見てください。取引のたびに説明文を作り直す構成ではありません。カテゴリごとに事前計算したベクトルが、その後の学習の初期値になります。

初期化の対象は、著者らの先行研究によるTransformerベースの取引理解モデルです。原論文の参考文献では「TREASURE」と呼ばれています。時刻、金額、merchant、MCC、所在地、異常に関するラベルなど、属性ごとのembeddingを連結して取引を表現し、取引系列を処理します。

今回の変更点は、MCC、merchant、所在地のうち選んだフィールドの初期値です。系列モデルの残りの構造や学習手順は元の方式を維持し、次の取引の予測と取引指標の評価を複数の目的として学習します。LLMの重みを取引モデルへ組み込む方式ではありません。

この設計なら、意味を補うためのLLM推論をオフラインへ移せます。ただし、初期embeddingの生成費用、保存、更新は必要です。また、論文にはオンラインのlatencyやthroughputを比較した実測表がありません。「推論経路にLLMが残らない」という設計上の利点と、実測による高速化率は分けて読む必要があります。

## コードや名前へ、業種と地域の文脈を足す

著者らは、MCCの数値だけをLLMへ渡す初期実験では十分な結果を得られなかったと説明しています。そのため、公式MCC資料と社内のmerchant・地理情報データベースを使い、説明を補いました。§3.2ではこれをmulti-source data fusionと呼んでいます。

| フィールド | 補う情報 | embeddingで表したい関係 |
| --- | --- | --- |
| MCC | 公式の業種説明、事業種別、関連merchantの例 | コードが指す業種と、近いカテゴリの関係 |
| Merchant | 名称、所在地、MCC、事業説明 | 同名や曖昧な名称を、地域と業種の文脈で捉える |
| Location | 地理情報、経済指標、人口構成、地域の金融的特徴 | 場所の名称に加え、取引に関係する地域特性を表す |

MCCの例はコード`5044`です。Listing 2では、写真・複写・マイクロフィルム機器と関連用品という業種説明を加え、扱う商品や近い業種も文章へ含めています。数値コードだけの入力を、業種間の関係が読める入力へ変える処理です。

Merchantの例では、名称に数字や電話番号を含む`365 MARKET 888 432-3299`へ、米国Michigan州Troyという所在地と、MCC`5814`の業種説明を足しています。名称だけでは判断しにくいmerchantを、所在地と業種で説明する方法です。

所在地の例では、New Yorkについて経済動向、人口構成、主要産業、金融規制を考慮するよう指示しています。ただし、Listing 1に具体的な経済統計値はありません。詳細なデータの取得時点や、統計を文章へ組み込む完全な手順も公開されていません。プロンプトが考慮を求める情報と、実際に検証済みデータとして入力した情報は、本文だけではすべて区別できません。

序論ではrule-based filteringやnull tokenへの置換も挙げていますが、具体的なルールや閾値は説明していません。merchant名から電話番号を必ず削除するなどの処理を、この論文の仕様として補うことはできません。

## 「一語で表す」という指示と、hidden stateの抽出

文脈を補った文章からembeddingを作る際、著者らはExplicit One-word Limitation Prompt Designを使います。説明の意味を一語へまとめるように指示し、表現をそろえる狙いです。

最終的に使うembeddingは、生成した一語をIDに変換したものではありません。論文は、LLMの最終層のhidden stateから、最後の非padding tokenの表現を取り出すと説明しています。hidden stateは、モデル内部で各tokenに対応するベクトルです。paddingは、入力長をそろえるために足すtokenを指します。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/transaction-last-token-embedding-mobile.svg">
    <img src="/img/posts/transaction-last-token-embedding.svg" alt="説明文と一語制約を含むプロンプトをLLMへ渡し、最終層の最後の非padding tokenのhidden stateをベクトルとして抽出する概念図" loading="lazy">
  </picture>
  <figcaption>図2。<a href="https://aclanthology.org/2025.emnlp-industry.61.pdf">原論文§1・§3.3.4</a>の抽出規則を説明する独自図。tokenとpaddingの配置は説明用の仮想例です。token位置の決め方や生成前後の詳細を、原論文の実装として補う図ではありません。</figcaption>
</figure>

図2で見るべき箇所は、paddingの直前にある有効なtokenです。padding側の表現を選ぶと、論文が指定する抽出規則に合いません。実装では、実際のtoken列とattention maskを使って有効な位置を特定する必要があります。

抽出規則は、概念的には次のように書けます。これは説明用の表記で、論文が公開した実行コードではありません。

```text
P_f(v) = フィールドfの値vについて、文脈を補ったプロンプト
H = LLMの最終層のhidden state
t = 最後の非padding tokenの位置
e_f(v) = H[t]
E_f[ID(v)] の初期値 ← e_f(v)
```

`E_f`は取引モデル側のフィールド別embedding tableです。LLMから得た`e_f(v)`を、対応するIDの初期値にします。その後、取引データを使ってモデルを学習します。初期化によって意味を与えることと、オンラインでLLMを呼び出すことは別の処理です。

再現には未公開の詳細が残ります。Listing 1〜3のフィールド別プロンプトには、一語制約の文言が明示されていません。説明文と制約を結合する完全なテンプレートや、tokenを選ぶ段階が生成前か生成後かも、掲載例だけでは確定できません。LLMごとに異なるhidden dimensionを取引モデル側へ合わせる処理、正規化、重みを固定するかどうかの詳細も記載されていません。

また、一語制約の有無、追加文脈の有無、ノイズ除去の有無を個別に比較するablationは掲載されていません。全体として改善した結果を、それぞれの工夫の単独効果として読むことはできません。

## 10億件の取引を時間順に分割して評価

実験には、2022年1月から2023年12月までの10億件の取引を使っています。最初の20か月を学習、21か月目をvalidation、最後の3か月をtestに分けます。暦に置き換えると、学習は2022年1月〜2023年8月、validationは2023年9月、testは2023年10〜12月です。

比較対象のVanillaは、カテゴリembeddingを従来どおり初期化する取引モデルです。embedding生成にはLlama2-7b、Llama2-13b、Llama3-8b、Mistral-7bを使います。Table 1では、MCCのみ、merchantのみ、MCCとmerchant、stateとcity、全フィールドという5群を比べ、計20設定を評価しています。

| 評価対象 | 予測するもの | 掲載指標 |
| --- | --- | --- |
| Next Amount | 次の取引金額 | MAE、sMAPE。小さいほどよい |
| Next MCC | 次の取引のMCC | Accuracy、F1。大きいほどよい |
| Next City | 次の取引の都市 | Accuracy、F1。大きいほどよい |
| Next Merchant | 次に利用するmerchant | Accuracy、F1。大きいほどよい |
| Transaction Metrics Assessment | 異常な取引の識別に関する評価 | 非公開の社内指標に対するRelative Improvement |

MAEは予測金額と実際の金額の差の絶対値を平均する指標です。sMAPEは予測値と実測値の大きさで誤差を規格化します。Accuracyは正しいカテゴリを予測した割合、F1は適合率と再現率の調和平均です。カテゴリの偏りや平均化方式によって、AccuracyとF1の評価は変わります。

Table 1の値は3回の実行の平均です。標準偏差、信頼区間、統計的有意性の検定は掲載されていません。金額の単位、sMAPEの計算上の詳細、F1の平均化方式、カテゴリ数も本文には示されていません。小さな差を評価するうえで、これらの条件は不足しています。

## 次のmerchantの予測は一貫して改善した

Table 1で最も一貫した改善があるのはNext Merchantです。VanillaのAccuracy`0.0760`に対し、LLMによる初期化を使った20設定はすべて上回っています。F1も、Vanillaの`0.0037`から全設定で改善しました。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/transaction-next-merchant-results-mobile.svg">
    <img src="/img/posts/transaction-next-merchant-results.svg" alt="Llama3-8bで初期化するフィールドを変えた5設定とVanillaについて、Next MerchantのAccuracyを0から12%の軸で比較した棒グラフ" loading="lazy">
  </picture>
  <figcaption>図3。<a href="https://aclanthology.org/2025.emnlp-industry.61.pdf">原論文Table 1</a>から、同じLLMでフィールドを変えた結果を比較。LLMを使う5設定はすべてLlama3-8bで、3回の平均値を百分率で表示しています。原表に分散の記載はなく、誤差棒は付けていません。</figcaption>
</figure>

図3では、MCCとmerchantを組み合わせた場合のAccuracyが`0.0994`で最も高くなっています。Vanillaとの差は`0.0234`、百分率で2.34ポイントです。相対改善は約30.8%になります。Accuracyそのものは9.94%なので、相対改善の大きさだけで実用上十分な精度と判断することはできません。

Llama3-8bのMCCとmerchantの組み合わせは、Next MCCのAccuracyでも`0.4168`と全設定中の最高値です。ただし、Next MCCのF1の最高値は全フィールド初期化の`0.1208`でした。AccuracyとF1で最良設定が変わります。

### 金額と都市は、指標を分けて読む

ほかのタスクまで一様に改善したわけではありません。次の表は、Llama3-8bで全フィールドを初期化した一つの設定を、同じVanillaと比べたものです。タスクごとの最高値を寄せ集めたモデルではありません。

| 指標 | Vanilla | Llama3-8b・全フィールド | 絶対差と方向 |
| --- | ---: | ---: | --- |
| Amount MAE | 37.0430 | 36.8128 | −0.2302、改善 |
| Amount sMAPE | 0.3952 | 0.3927 | −0.0025、改善 |
| MCC Accuracy | 0.4107 | 0.4155 | +0.0048、改善 |
| MCC F1 | 0.1118 | 0.1208 | +0.0090、改善 |
| City Accuracy | 0.8454 | 0.8455 | +0.0001、改善 |
| City F1 | 0.6721 | 0.6716 | −0.0005、悪化 |
| Merchant Accuracy | 0.0760 | 0.0979 | +0.0219、改善 |
| Merchant F1 | 0.0037 | 0.0110 | +0.0073、改善 |

全フィールド初期化でもCity F1は少し下がっています。MCCだけをLlama3-8bで初期化すると、Amount MAEは`37.5144`、sMAPEは`0.4010`となり、どちらもVanillaより悪化します。一方、その設定のCity F1は`0.6803`で、Table 1の最高値です。意味表現を足したフィールドだけが改善する、という単純な対応でもありません。

原論文§4.5は、location中心の初期化で都市予測が改善すると総括しています。しかし、stateとcityを初期化した4設定のCity F1は、すべてVanillaの`0.6721`を下回ります。City Accuracyがbaselineを上回るのは、Llama2-13bの`0.8459`だけです。著者の文章による総括と、指標別の実数値にはずれがあるため、この記事ではTable 1の値を基準にしています。

## 非公開の業務指標では最大3.93%の相対改善

Table 2はTransaction Metrics Assessmentの結果です。これは社内の業務評価で、詳細なスコアを公開する代わりに、既存システムに対する相対改善率RIを示しています。

```text
RI = (S_evaluated − S_baseline) / S_baseline × 100%
```

`S_evaluated`は評価モデルのスコア、`S_baseline`は既存システムのスコアです。この値はAccuracyのポイント差ではありません。基準のスコアから何%変わったかを表します。

| embeddingを生成するLLM | MCC + Merchant | Location | All Fields |
| --- | ---: | ---: | ---: |
| Llama2-7b | −0.40% | +2.85% | +3.72% |
| Llama2-13b | +2.66% | +1.77% | +2.92% |
| Llama3-8b | +0.37% | +2.78% | +3.32% |
| Mistral-7b | +0.83% | +2.89% | +3.93% |

12設定のうち11設定でRIが正になり、全フィールド初期化が各LLMで最良でした。最大値はMistral-7bの`+3.93%`です。Next MerchantのAccuracyで最良だったLlama3-8bとは異なります。予測タスクと社内の評価では、選ぶべき設定も変わります。

RIが`+3.93%`なら、評価モデルのスコアはbaselineの`1.0393`倍です。ただし、元のスコア、指標の内訳、閾値、誤検知と見逃しの関係は非公開です。不正検知率が3.93ポイント上がった、損失額が3.93%減った、あるいはA/Bテストで事業効果が確認された、と読み替えることはできません。

## 公開情報で再現できる範囲

論文本文とAnthologyページには、この手法の再現用コード、学習済み重み、10億件の取引データへの配布リンクがありません。学習の詳細は先行モデルへ参照を委ねており、この論文だけでTable 1・2を完全再現できる状態ではありません。

確認できるのは、対象フィールド、文脈を補う考え方、プロンプト例、hidden stateの抽出規則、時間分割、比較設定、結果です。正確なmodel checkpoint、tokenizerとテンプレート、embedding dimensionの変換、optimizer、学習率、batch size、系列長、損失の重みなどは、この論文では十分に指定されていません。

また、初期化後にembeddingをどのように更新するか、学習時に見なかった新merchantをどう登録するかも詳細がありません。§3.4.1では未知カテゴリへの汎化を利点に挙げていますが、未知カテゴリを分離したcold-start評価は掲載されていません。意味が近いカテゴリへ一般化しやすいという設計意図と、実測されたcold-start性能は分けて扱う必要があります。

## 自分の取引モデルで検証するなら

以下は論文の実験手順を完全再現する方法ではなく、公開された設計を自分のモデルで検証するための提案です。

### まず、初期化だけを変えた比較を作る

最初にランダム初期化の系列モデルをbaselineとして固定します。カテゴリ辞書、学習・評価期間、系列長、損失、学習回数をそろえ、MCCだけの意味初期化から試します。MCCは説明を用意しやすく、merchant全体より小さな辞書で試せる対象です。ただし、論文でMCCだけが全タスクに最良だったという意味ではありません。

各カテゴリについて、ID、説明、説明の出典と取得日、prompt version、LLMのcheckpoint、tokenizer、抽出規則を記録します。生成したベクトルも、これらの設定と対応させて保存します。学習と推論でIDの並びが変わると、別カテゴリのembeddingを参照してしまうため、カテゴリ辞書も同じversionで管理してください。

embedding dimensionが一致しない場合は、射影層などの変換が必要です。この論文では変換方法が公開されていないため、試す方式を独自の実装として明示し、同じdimensionのbaselineと比較します。追加parameterや重みを固定する条件もそろえます。

### 追加文脈と一語制約を別々に検証する

意味初期化の効果が確認できたら、名称だけ、説明を補った入力、一語制約を加えた入力を分けて比較してください。merchantと所在地も段階的に追加します。論文が個別に測っていない要素を、自分のデータではablationとして評価する方法です。

評価には、主タスクのAccuracyやF1に加え、カテゴリ頻度別の性能、未知カテゴリ、経時変化を含めます。複数seedのばらつきも確認します。説明データへ将来の情報が混ざっていれば、時系列でtestを後ろに置いても比較は適切になりません。業種や所在地の説明は、評価時点で利用できた情報にそろえます。

### オフライン費用と更新を運用へ組み込む

LLM生成にかかった時間と費用、embedding保存量、辞書更新時間、取引モデルのp50・p95 latencyを別々に記録します。オンライン推論でLLMを使わない方式でも、説明の更新と再学習には費用が必要です。

merchantの業種変更や地域特性の変化へ対応するには、説明の更新日と差分を追い、必要なカテゴリを再計算する仕組みを検証します。新しい初期値を学習済みモデルへそのまま上書きする操作は、論文の評価対象ではありません。再学習したモデルを旧版と比較し、辞書、embedding、モデルをまとめて切り戻せるようにします。

更新に失敗した場合は検証済みの旧版を使い、未登録カテゴリには学習・検証したfallbackを用意します。外部のLLMサービスを使う場合は、merchant情報の送信範囲を決め、個別の取引履歴を送る必要があるかを見直してください。この手法のembedding生成はカテゴリの説明が中心なので、顧客ごとの履歴を入力へ追加することは別の設計変更になります。

## 論文が残している課題

著者らは、プロンプトの高度化、評価モデルの拡張、対象フィールドの拡張、静的なembeddingの更新を課題として挙げています。今回の一語制約は単純な方式であり、金融分野に合わせたLLMのfine-tuningや、より高度なprompt optimizationは将来の検討です。

専用のsentence embeddingモデルであるNV-EmbedやQwen3-Embeddingは、この実験では比較していません。取引チャネル、支払方法、時間的なパターンへの拡張も未検証です。新しいモデルや別フィールドで同じ改善が得られるとは断定できません。

静的な意味初期値では、merchantの変化、季節性、市場の変動を直接反映できません。取引系列モデルが時系列を学習することと、カテゴリの説明が最新であることは別の問題です。

この記事の観点では、個別の工夫のablationがないこと、再現設定の不足、業務指標の非公開性も制約です。学習時間が短くなるという利点も記されていますが、学習時間や収束速度の比較値は示されていません。意味を含むembeddingが、予測理由を人間に説明できることも、この実験では検証されていません。

この研究は、カテゴリIDによる効率的な処理を保ちながら、学習の出発点に意味情報を加える方法を示しています。取引データで学べる利用パターンと、名称・業種・地域から得られる知識を組み合わせる設計です。導入時には、merchant予測の改善、ほかのタスクでの悪化、オフラインの費用と更新を、用途ごとに検証する必要があります。

## 参照資料

- Xiran Fan et al., [Enhancing Foundation Models in Transaction Understanding with LLM-based Sentence Embeddings](https://aclanthology.org/2025.emnlp-industry.61/), EMNLP 2025 Industry Track, pp. 903–911, DOI: 10.18653/v1/2025.emnlp-industry.61.
- [原論文PDF](https://aclanthology.org/2025.emnlp-industry.61.pdf)。手法は§3、実験条件は§4.1〜4.4、結果はTable 1・2、限界はp. 909。先行モデルTREASUREは参考文献のYeh et al. (2025)に記載されています。
