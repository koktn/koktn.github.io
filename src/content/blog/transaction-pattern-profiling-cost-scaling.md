---
title: 「誰を推測するか」から「購入の意味」へ――銀行規模のLLMプロファイリング
description: ユーザー単位のLLM推論を取引パターン単位へ置き換え、検索可能な属性データベースを構築する研究を、手法、評価、実運用、再現性から解説します。
publishedAt: 2026-09-25
category: AI
tags:
  - LLM
  - User Profiling
  - Transaction Data
  - Data Mining
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値や主張は原論文と公開実装を確認して記載していますが、利用時は原文も確認してください。

この研究の価値は、購買履歴を読むLLMの推論単位を**ユーザーから、複数ユーザーに共有される取引パターンへ変えたこと**にあります。モデルを軽量化するのではなく、同じ意味を何度も推論しない設計によって、数千万人規模の処理を可能にしています。

取り上げるのは、Ryota Mitsuhashi、Tetsuro Morimura、Hirotake Itoによる「[From “Who Is This User?” to “What Does This Purchase Mean?”](https://arxiv.org/abs/2609.19928)」です。2026年9月17日にarXiv v1が公開され、ICDM 2026 Applied Research trackに採択されています。本稿では[論文PDF](https://arxiv.org/pdf/2609.19928)と[公式実装](https://github.com/CyberAgentAILab/profiling-agent-open-ecommerce)をもとに、仕組みと結果、まだ確かめられていない点を整理します。

## 1人ずつLLMに読ませる設計はscaleしない

購買履歴から「幼い子どもがいる家庭」「健康を意識した生活を送る人」のような自然文のprofileを作るなら、各ユーザーの履歴をLLMへ渡す方法が最も素直です。しかし、この方法には三つの問題があります。

第一に、推論回数がユーザー数に比例します。論文は、数千万人のユーザーへ1人あたり1回推論するだけでも、公開APIの代表的な料金と1人あたり数千tokenという仮定では、全件処理が数万〜数十万ドル規模になり得ると試算しています。履歴が増えるたびに更新対象も生じます。

第二に、同じような購入をした人でも、LLMが生成する表現は揺れます。「子どもの教育を重視する親」と「教育熱心な保護者」は意味が近くても、文字列としては別物です。横断検索には、別途canonicalizationが必要になります。

第三に、毎回ユーザーの全履歴を読むと、過去に解釈済みの購入を繰り返し処理します。共通する行動が多いほど、重複計算も大きくなります。

論文は、この問題を問いの立て方から変えます。

```text
従来: このユーザーはどのような人か？
提案: この購入パターンは何を意味するか？
```

ユーザー数を `N`、共有されるユニークな取引パターン数を `P` とすると、LLM推論の対象は概念的に `N` 件から `P` 件へ変わります。人気商品へ行動が偏り、一定以上の頻度を持つパターン数が飽和する環境では、`P`はユーザー数ほど速く増えません。

## Resolve・Profile・Tagの3段階

提案pipelineは、商品や取引先の意味を整えるResolve、属性を生成するProfile、自由文を検索可能な語彙へまとめるTagで構成されます。

```text
取引履歴
  ↓
Resolve: item名を解釈し、推論に必要な意味へ抽象化
  ↓
Profile: 頻出patternごとに属性とpriorを生成
  ↓
Tag: 自由文属性をcluster化してtagを付与
  ↓
patternの結果をユーザーへ集約した属性database
```

### Resolve：推論前にitemの意味を整える

商品名や銀行の取引先名には、略称、表記揺れ、ブランド、容量、包装といった情報が混在します。Resolveはitemごとに外部情報が必要かを判断し、必要な場合だけWeb検索を使います。検索結果から根拠となるsnippetを選び、属性推論に有効な意味を残した表現へ書き換えます。

この段階はprecisionを重視します。特定の人物属性を示す材料がないitemでは棄権でき、曖昧な情報を無理に後段へ流しません。銀行で一意に取引先を特定できない場合も、そのpatternは推論対象から除かれます。

### Profile：patternごとにhybridな属性を作る

Profileでは、複数ユーザーに出現する単一itemやitemの組み合わせを頻出patternとして抽出します。組み合わせを使う理由は、単品よりも共起に意味があるからです。たとえば、粉ミルク、おむつ、ベビーベッドの同時購入は、それぞれを単独で見るより最近の出産を強く示唆します。

各patternについて、LLMは1回の呼び出しで次の情報を生成します。

| 出力 | 役割 |
| --- | --- |
| closed-set属性 | 事前に定義した選択肢へ分類する。各項目にunknownを持つ |
| open-set属性 | schemaへ収まりにくい人物像を短い自由文で表す |
| prior `π(d)` | pattern購入者に自由文属性`d`が当てはまる強さを`0`〜`1`で表す |

公開データ用のclosed-set schemaは、demographic 7項目、psychographic-behavioral 4項目、life-event 8項目です。open-set側は各カテゴリで0〜3件の候補を生成します。固定選択肢は評価と集計を安定させ、自由文は固定taxonomyでは列挙しきれないlong-tailの属性を補います。

`π(d)`は、利用時にprecisionとrecallを調整するための値です。高いthresholdを使えば弱い推測を除きやすくなり、低くすれば候補を広く拾えます。ただし、論文は`π(d)`を校正済み確率とは位置付けていません。`0.8`を「80%の確率で正しい」と読むことはできません。

### Tag：自由文を検索できる単位へまとめる

自由文属性は柔軟ですが、同じ意味でも表現が変わります。そこでTag段階では、属性文をsentence encoderで埋め込み、UMAPで次元削減し、HDBSCANでcluster化します。最後にLLMが各clusterを短いtag名へ要約します。

公開データでは、noise clusterを除いて76 tagが作られました。内訳はdemographic 23、psychographic-behavioral 23、life-event 30です。97,801商品のうち62,447商品、63.9%に少なくとも1 tagが付き、1商品あたりの平均は1.05 tagでした。

こうして得られるのは、元の履歴を完全保存した表現ではありません。具体的な商品名を属性へ変換し、さらにclusterへまとめるため、情報は圧縮されます。その代わり、ユーザーを自然言語のtagで検索し、同じ概念を持つ集団として集計できるようになります。

## 公開データでは何を検証したのか

評価には[Open E-Commerce 1.0](https://doi.org/10.1038/s41597-024-03329-6)が使われています。Amazonの購入履歴と自己申告surveyを組み合わせたdatasetで、約5,027人、約185万件の購入、97,801商品を含みます。商品を3人以上が購入していることなどのfilter後、closed-set属性の評価には4,990人、tagを用いる評価には3,781人が使われました。

論文の中心的な問いは、属性databaseへ圧縮した後も、生の履歴が持つsignalを保てるかです。比較する入力は次の三つです。

| 入力 | 内容 |
| --- | --- |
| oracle | ユーザーの生の購入履歴 |
| tags-only | open-set属性をcluster化したtag |
| hybrid | tagにclosed-set属性を加えた提案形式 |

ここでoracleは競合手法ではなく、圧縮前の情報を持つ参照入力です。

### closed-set属性の結果

年齢、性別、収入、教育、転居では、生履歴を読むLLMと、属性databaseを決定的なruleで集約した予測を比較しています。

- 性別のmacro-F1は`0.814`から`0.794`
- 収入のordinal MAEは`1.252`から`1.272` bin
- 年齢MAEは`8.22`年から`8.80`年
- 転居F1は`0.128`から`0.287`

転居のF1改善は、recallが`0.079`から`0.318`へ上がった一方、precisionが`0.337`から`0.262`へ下がった結果です。候補を広く拾う方向に変わったため、用途によって良し悪しが異なります。

教育のordinal MAEも`0.900`から`0.815`へ小さくなりましたが、Spearmanの順位相関は`0.154`から`0.038`へ低下しました。modeによる集約が中央付近へ予測を寄せた影響とされており、MAEだけから情報保持が改善したとは言えません。

### tagを使う10属性の結果

喫煙、飲酒、糖尿病、車いす利用、出産、妊娠、転居、失職、離婚など10属性では、同じLLMをjudgeとして使い、各入力から自己申告属性を予測させています。結果のmacro-AUCは次の通りです。

| 入力 | macro-AUC |
| --- | ---: |
| oracle | 0.611 |
| tags-only | 0.593 |
| hybrid | 0.611 |

oracleとhybridの属性別AUCを比べたWilcoxon signed-rank testは`p=0.922`で、有意差は検出されませんでした。ただし、これは同等性試験ではありません。「差が見つからなかった」ことと「同じ性能だと証明した」ことは区別する必要があります。

属性別では、出産が`0.791`から`0.743`、妊娠が`0.816`から`0.780`へ下がりました。特定商品が強いsignalになる属性では、商品名をtagへ圧縮する損失が表れています。失職と離婚は、全入力で95% confidence intervalが偶然水準の0.5をまたぎました。

hybridとtags-onlyの比較では、hybridが10属性中9属性で上回り、macro-AUC差は`+0.018`、Wilcoxon testは`p=0.004`でした。closed-set属性とopen-set tagを組み合わせる設計には、tagだけでは持たない情報があることを示す結果です。

### priorは属性によって有効性が異なる

`π(d)`が自己申告の陽性・陰性を分けるかは、妊娠、車いす利用、糖尿病の3属性で検証されました。車いす利用は`p=2.3×10^-4`、Cliff's `δ=0.334`、糖尿病は`p=8.6×10^-3`、`δ=0.221`でした。妊娠は`p=0.17`、`δ=0.10`で有意ではありません。

したがって、priorには順位signalが含まれる場合がありますが、すべての属性へ一律に使えるわけではありません。利用先の属性ごとにvalidationと、必要なら確率校正が必要です。

## 銀行では約5万patternへ圧縮した

論文は、日本の大手銀行で稼働するsystemも報告しています。対象は数千万人規模です。Amazon商品とは異なり、銀行では「取引先口座名と入出金方向の組」をpatternとします。同じ企業でも、支払った人と報酬を受け取った人では役割が変わるためです。

半角カタカナの取引先名は、全文検索engineのTantivyとgBizINFOを使って法人名へ解決します。productionのclosed-set schemaも公開データ用とは異なり、職業、full-time、part-time、学生、退職者、子どもの有無、教育志向の7項目です。

頻出patternが元の取引の約95%を覆うようにthresholdを設定すると、数千万人のユーザーが約5万patternへ圧縮されました。論文の報告値は次の通りです。

| 指標 | 報告値 |
| --- | ---: |
| LLM推論対象数 | per-user方式の約`1/600` |
| 推定費用 | 同じtoken数・単価を仮定したper-user方式の約`1/300` |
| 取引先を一意に解決できないpatternを除いた取引coverage | `82.6%` |
| 生成された意味tag | 約149 |

銀行dataは機密のため、個人ごとの属性正解率や生の取引は公開されていません。約600倍の圧縮はproductionでのscalabilityを示す値であり、予測精度の検証はOpen E-Commerce上の実験に基づきます。この二種類のevidenceを混同しないことが重要です。

また、同じ取引先でも方向によって異なるprofileが生成されます。家事代行、ベビーシッター、介護支援を提供する企業なら、出金側ではサービス利用者、入金側では従業員や業務委託者が候補になります。pattern定義にdomain固有のcontextを含める重要性が分かる例です。

## 公式実装を試すときの注意

[profiling-agent-open-ecommerce](https://github.com/CyberAgentAILab/profiling-agent-open-ecommerce)には、Open E-Commerce向けのpipeline、設定、sample結果、offline testが公開されています。銀行向けのdataとproduction実装は含まれません。

対応環境はUbuntu 24.04 LTSのx86_64で、macOSとWindowsは未検証です。依存関係は`uv`で管理され、実modelを動かすにはNVIDIA GPUが必要です。

```sh
uv sync
uv run scripts/open_ecommerce/local/download_dataset.py
uv run python -m unittest tests.e2e.test_pipeline_e2e -v
```

offline testは、LLM応答、検索結果、埋め込みをfixtureとstubへ置き換え、7段階の処理がfile I/Oで接続されることを確認します。model品質や論文の数値を再現するtestではありません。

repositoryの既定modelは生成用のQwen3.5-4Bと埋め込み用のQwen3-Embedding-0.6Bです。論文の公開データ実験はQwen3.5-27Bとplamo-embedding-1bを使い、単一のNVIDIA A100 80GBで属性database構築に約33 GPU時間を要しました。既定設定を実行しただけでは、論文と同条件の再現にはなりません。

Web検索も、既定ではDuckDuckGoを利用しますが、論文結果の再現にはSerper API tokenの設定が案内されています。公開済みの複合patternは2〜4商品からなる541件で、datasetを取得済みならFP-Growthによる再抽出はCPUで約15秒とされています。

小規模に試す場合は、まずoffline testでpipelineの接続を確認し、10件のsample、`NUM_SAMPLES`を指定したsubset、全dataの順に広げるのが安全です。model、検索provider、prompt、cluster設定が変われば結果も変わるため、resolved item、生成属性、cluster、tagを段階ごとにsample監査する必要があります。

## この手法が向かない場合

取引パターン方式は、複数ユーザーに共有される行動が多いほど有効です。反対に、次の条件では利点が小さくなります。

- 希少なitemに最も重要なsignalがある
- 購入順序や時刻の変化が重要である
- 同じ商品でもユーザー固有の文脈で意味が変わる
- 頻出pattern数がユーザー数とともに増え続ける
- 誤った属性推定の影響が大きく、集団的な推測を許容できない

論文のpipelineはtimestampを入力として使わないため、「いつ買ったか」「どの順序で買ったか」を直接扱えません。頻出patternへ絞ることでrare itemも落ちます。論文は、希少だが重要な履歴を持つユーザーだけper-user推論へfallbackする構成を将来の選択肢として挙げています。

評価にも制約があります。tag評価のcohortは、対象survey属性のうち少なくとも一つが陽性のユーザーへ限定されています。同じLLMがdatabase構築とjudgeの両方へ使われるため、promptを揃えてpositionやverbosityのbiasを抑えても、自己選好biasは残ります。非LLM classifierとの対称比較、Resolve・Profile・Tagのcomponent ablation、model間比較、集約ruleのablationは今後の課題です。

## profile生成はprivacy設計そのもの

購買や送金から人物属性を推測する処理は、attribute inference attackと同じ構造を持ちます。読みやすいtagへ変換しても、元のsignalがsensitiveでなくなるわけではありません。

論文は性的指向をschemaから意図的に除外し、銀行ではtransactionを外部へ出さず、on-premisesのQwen3-30B-A3B-Instruct-2507を使っています。それでも、健康、家族、雇用、資産に関する推測は、広告配信を通じて住宅や雇用などの機会を不公平に制限する可能性があります。

導入時には、技術的なaccuracyとは別に、推論しない属性、利用可能な目的、閲覧権限、保持期間、監査log、本人による訂正や異議申立て、誤推論時の救済を定める必要があります。on-premises処理はdata流出riskを下げますが、推論と利用の正当性までは保証しません。

## まとめ

この研究が示したのは、LLMのcost問題をmodel選択だけで解く必要はないということです。ユーザーごとに繰り返していた意味解釈を、共有可能な取引パターンへ移せば、推論結果をcacheし、一貫した属性語彙として再利用できます。

一方、属性databaseは生の履歴をlosslessに置き換えるものではありません。商品名、rare item、時系列、個人固有の文脈を落とす代わりに、cost、検索性、一貫性を得る設計です。公開データではmacro-AUCが生履歴と同じ値になりましたが、属性別には損失があり、同等性が証明されたわけでもありません。

実運用で問うべきなのは、「profileを作れるか」だけではなく、**どの意味を共有してよいか、どの情報を捨ててよいか、その推測を何に使ってよいか**です。推論単位の変更はscaleを解決しますが、profileの妥当性と利用責任は別途設計しなければなりません。

## 参照

- Ryota Mitsuhashi, Tetsuro Morimura, Hirotake Ito, [From “Who Is This User?” to “What Does This Purchase Mean?”](https://arxiv.org/abs/2609.19928), [PDF](https://arxiv.org/pdf/2609.19928), arXiv:2609.19928v1, 2026-09-17.
- CyberAgentAILab, [profiling-agent-open-ecommerce](https://github.com/CyberAgentAILab/profiling-agent-open-ecommerce), Open E-Commerce向け公式実装.
- Berke et al., [Open E-Commerce 1.0, five years of crowdsourced U.S. Amazon purchase histories with user demographics](https://doi.org/10.1038/s41597-024-03329-6), Scientific Data 11, 491, 2024.
