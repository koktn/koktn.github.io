---
title: LLMを広告rankerにしない：Pinterestの補助予測器がRoASを改善した仕組み
description: Pinterestの広告推薦論文を題材に、LLMによる広告主予測、SFTとGRPO、Semantic ID、既存のretrieval・rankingへの統合、オンラインRoAS改善と制約を解説します。
publishedAt: 2026-10-04
category: AI
tags:
  - Large Language Models
  - Recommender Systems
  - Ads Ranking
  - Semantic ID
  - GRPO
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、PinterestのHui Yangらによる論文「[Fine-Tuned LLM as a Complementary Predictor Improving Ads System](https://arxiv.org/abs/2605.27856)」です。2026年5月27日にarXiv v1として公開されたプレプリントで、[HTML全文](https://arxiv.org/html/2605.27856)と[PDF](https://arxiv.org/pdf/2605.27856)も公開されています。

この論文の価値は、LLMに広告を直接順位付けさせず、「次にconversionしそうな広告主」を予測する補助モデルとして使った点にあります。予測結果を既存のretrievalとrankingの両方へ渡し、米国Shopping AdsでRoASを相対4.94%改善しました。

ただし、公開情報だけで完全には再現できません。学習データ、サンプル数、基盤となるopen-source LLMの名称、学習設定、GPU台数、推論コスト、遅延、コードは公開されていません。本記事では、論文が報告した結果と、そこから導く導入案を分けて説明します。

## なぜLLMを広告rankerにしないのか

大規模な広告推薦は、少数の広告候補を集めるretrievalと、候補を精密に評価するrankingを組み合わせます。従来モデルの強みは、user ID、advertiser ID、campaign IDなどの疎なID特徴と、特徴量同士の交互作用を効率よく扱えることです。

一方、LLMはテキストの意味や一般知識を利用できますが、疎なIDをそのまま扱うのは得意ではありません。全リクエストでLLMを動かすと、tail latency、GPU memory、推論費用も問題になります。既存の広告配信基盤をLLMへ置き換える場合、品質以外にも大きな負担が生じます。

著者らは役割を限定しました。LLMはuser profileと行動履歴を読み、conversion意図が高そうな広告主20件と興味を最大5件出します。広告の最終scoreは計算しません。広告主予測を既存モデルが使える小さな信号へ変換する設計です。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/llm-complementary-ads-predictor-system-mobile.svg">
    <img src="/img/posts/llm-complementary-ads-predictor-system.svg" alt="ユーザー選定、特徴量編集、LLMによる広告主予測、retrievalとrankingへの統合を示す処理図" loading="lazy">
  </picture>
  <figcaption>原論文Figure 1と§3.1〜3.7を基に本記事用に再構成。LLMは既存の配信系を置き換えず、広告主予測を2つの経路へ供給します。</figcaption>
</figure>

## 入力と正解ラベルをどう作るか

日次推論の対象は、米国のactive userのうち、過去90日以内にon-Pinterestまたはoff-Pinterestのconversionがあるユーザーです。全ユーザーを処理せず、商業的な意図を観測できる層へ絞り、推論量を抑えています。

日付`x`を特徴量の基準日とし、`x+1`から`x+7`までに最初にconversionした広告主を正解とします。複数の広告主を正解にするmulti-label taskではなく、「次の広告主」1件を当てる問題です。trainとevaluationはuser ID単位で9対1に分け、同じユーザーが両方へ入るleakageを避けています。

入力には年齢、性別、user stateのprofileに加え、次の行動情報を含めます。

- on-site検索
- off-siteのattributed conversionとmatched conversion
- conversionに関連するoff-site検索とURL
- 行動から集約したcategory、interest、brand
- 過去にconversionしたactive advertiser
- 米国Shopping Adsで日次売上上位のadvertiser pool

論文の実験で使った期間は、検索queryが過去3か月、URLが過去2週間です。すべての特徴量に同じ期間を適用してはいません。prompt長と予測性能の兼ね合いで、特徴量ごとに期間を決めています。

## SFTからGRPOへ段階的に学習する

最初のSFTでは、正解の広告主1件だけを自由テキストで予測させます。出力空間を狭くし、最も精密なnext-advertiser predictionを先に学ばせるためです。

続くGRPOでは、同じユーザー情報から広告主20件の順位付きリストと興味5件をXMLで出力します。正解ラベルは引き続き広告主1件です。正解が何位に入ったかを報酬とし、1〜4位なら2.0を加えます。広告主数や興味数が指定と違う場合は、長さに応じたpenaltyがかかる仕組みです。

```text
total reward
  = correct advertiserの順位報酬
  - advertiser件数のpenalty
  - interest件数のpenalty

順位iの基本報酬 = 0.1 × (20 - i)
順位1〜4には +2.0
```

推論時もGRPOと同じXML形式を使い、学習時と推論時の形式差を減らします。後段のsystemがparseしやすくなる利点もあります。論文のV1評価では、20件を直接SFTする方法より、1件予測のSFTを経てGRPOへ進む方法が高いRecallを得ました。

明示的なreasoning promptは、このタスクでは一貫した利益を示しませんでした。広告主予測は長い推論chainより、複数の履歴から意図を集約する能力に依存すると著者らは考察しています。

## Semantic IDはテキストにない行動の近さを補う

テキストだけでは、似たコンテンツを見たユーザー同士の共起関係や画像の意味を十分に表せません。著者らはPinCLIPのmultimodal embeddingをRQ-VAEで量子化し、5階層、各階層20,248 codeのSemantic IDを作りました。

Semantic IDをLLMへ覚えさせる学習は3段階です。最初に通常tokenとTransformerを固定し、SID tokenのembeddingだけを更新します。次に全parameterを解凍し、SIDを含む推薦データと一般domainのデータを混ぜてcontinued pre-trainingを行います。最後の段階は、テキストと最近のSID列から次の広告主を予測するSFTです。

この手順は、SID tokenを追加しながら一般知識とinstruction-following能力を失いにくくするための設計です。ただし、SID版はofflineでしか評価されていません。オンラインRoAS改善をSIDの効果として読むことはできません。

## 予測結果をretrievalとrankingへ戻す

retrievalでは、LLMが予測した広告主をtargeting filterとして使います。その広告主の広告を、engagement向けに学習したtwo-tower modelで検索し、既存のcandidate generatorへ候補を追加します。新しいretrieval channelの役割は、主経路の置き換えではなく、既存経路が十分に拾えていない広告主を補うことです。

rankingでは、予測した広告主とinterestを特徴量に変換し、ctcvrとvtcvrのconversion modelへ加えます。同じLLM出力を、候補集合を広げる処理とconversion確率を推定する処理の両方に利用しています。

LLM推論はrequest pathから外し、vLLMとRayを使った分散batchで処理します。prefix caching、paged attention、continuous batchingによってGPU利用率を高める構成です。virtual epochごとにcheckpointを保存するため、失敗時は未完了分から再開できます。行動が増えたユーザーだけを再推論するincremental updateも、日次の処理量を減らす仕組みです。

## Offline評価で何が改善したか

本番trafficに近いV1では、zero-shotから1件予測のSFT、GRPO、SIDへ進むにつれRecallが上がりました。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/llm-complementary-ads-predictor-results-mobile.svg">
    <img src="/img/posts/llm-complementary-ads-predictor-results.svg" alt="V1データにおける広告主予測のRecall at 1、5、20を学習方法別に比較した棒グラフ" loading="lazy">
  </picture>
  <figcaption>原論文Table 3のV1 offline advertiser-prediction結果を可視化。SID版と非SID版の差はoffline評価であり、オンライン実験の差ではありません。</figcaption>
</figure>

| Method | Recall@1 | Recall@5 | Recall@20 |
| --- | ---: | ---: | ---: |
| Zero-shot prompting | 0.117 | 0.301 | 0.422 |
| SFT、20件予測prompt | 0.156 | 0.314 | 0.456 |
| SFT、1件予測prompt | 0.214 | 0.413 | 0.501 |
| 1件SFT + GRPO | 0.223 | 0.461 | 0.683 |
| SID対応SFT + GRPO | 0.248 | 0.515 | 0.755 |

SID対応版は非SID版のSFT + GRPOと比べ、Recall@1、@5、@20を相対11.2%、11.7%、10.5%改善しています。論文が述べる「10%以上」は相対改善です。

ranking側では、LLM由来の広告主特徴を加えるとctcvrのAUCが相対0.06%、PR-AUCが0.71%改善しました。vtcvrではAUCが0.09%、PR-AUCが1.64%改善しています。conversionのようなpositiveが疎な問題では、PR-AUCの変化も確認する必要があります。

feature ablationでは、過去にconversionしたactive advertiserを除くとRecall@5が0.1000低下しました。off-site URLの除去は0.0290、off-site検索は0.0140の低下です。user profileの除去は0.0020にとどまり、top categoryを除くと逆に0.0039上がりました。この設定では、単純な属性より直近の商業行動が予測に寄与しています。

## オンライン実験でRoASが相対4.94%改善

オンライン実験は、米国市場でdata利用にopt-inしたユーザーを対象に、Home Feed、Related Pins、Searchの3つの面で行われました。LLM-based candidate generatorを追加した結果は次の通りです。

| 対象 | RoASの相対変化 | p値 |
| --- | ---: | ---: |
| US Shopping slice | +4.94% | 0.021 |
| Opt-in US Shopping treatment slice | +6.69% | 0.029 |

4.94%と6.69%は売上の絶対増加ではなく、Return on Ad Spendの相対変化です。論文は実験期間、traffic量、信頼区間、絶対RoASを公開していません。p値は報告されていますが、効果の安定性や他市場への一般化までは判断できません。

candidate generatorの学習目標も結果を左右しました。impressionをpositiveにすると後段まで残る候補は増えましたが、engagement指標が悪化しました。conversionを直接positiveにするとlabelが疎で学習が不安定でした。clickへ滞在時間による重みを付けた目標が、funnel survivalとengagementのバランスを最もよく取れました。

候補の割り当て量も増やせばよいわけではありません。LLM経路はconversion意図の強い広告主へ集中するため、枠を広げすぎるとcandidate blendingを占有します。その後の重複排除を経ると、広告主の多様性が下がりました。

## 実装するときは小さな補助経路から始める

ここからは論文の内部実装を再現する手順ではなく、公開情報から導いた導入案です。

まず、既存のretrievalとrankingは変えず、LLM出力をofflineで保存します。`user_snapshot_at`、入力特徴量の対象期間、model version、広告主候補、順位、生成日時を持たせます。広告主名を自由生成させるより、配信可能な広告主poolへ制約し、IDへの変換失敗と無効な広告主を記録できる設計が安全です。

次にhistorical replayでRecall@Kを測ります。user ID単位でtrainとevaluationを分け、特徴量の基準日より後の情報がpromptへ混ざっていないことを確認します。zero-shot、1件SFT、複数件SFT、必要ならRLの順で比較し、複雑な学習方法を最初から前提にしません。

retrievalへの統合は小さなquotaから始めます。新経路だけのRecallでは判断せず、既存経路との重複率、funnel survival、広告主の多様性、p95／p99 latency、1,000ユーザーあたりのbatch推論費用も測ります。rankingでは特徴量の欠損を許容し、LLM batchが遅延または失敗した場合に従来のscoreへ戻せるようにします。

off-site URLやconversionは予測に寄与しましたが、privacy上の取り扱いが難しい情報です。利用目的、同意、retention、削除要求、地域ごとの規制を先に確認し、必要な期間と粒度へ制限します。性能が上がることと、その特徴量を利用できることは別の判断です。

オンライン実験では、RoASだけでなくCTR、CVR、CPA、広告主の集中度、ユーザー体験、計算費用を同時に見ます。treatment対象、candidate quota、two-towerの学習目標を固定しなければ、LLM予測そのものの効果と経路設計の効果を分けられません。

## 再現性と適用範囲

本論文は、本番広告システムへLLMを追加する現実的な設計を示しています。一方、外部で検証するには次の情報が不足しています。

- arXiv v1のプレプリントで、査読済みとは記載されていない
- privateなPinterestデータを使い、データ量とサンプル数を公開していない
- base LLMの名称、parameter数、学習hyperparameter、GPU構成を公開していない
- promptは掲載されているが、学習・配信コードは公開していない
- batch推論の所要時間、GPU費用、更新頻度ごとの処理量を公開していない
- online experimentの期間、traffic量、絶対指標、confidence intervalを公開していない
- SID対応版はoffline評価のみで、online効果を検証していない

また、対象は過去90日以内にconversionがある米国のactive userです。新規ユーザーやconversion履歴の薄いユーザー、Shopping以外の広告、市場や規制の違う地域へ、そのまま効果を外挿することはできません。

## まとめ

Pinterestの設計は、LLMを既存の広告rankerと競わせず、意味理解を活かせる広告主予測へ役割を限定しました。日次batchで作った予測を、retrievalでは候補を補うfilterに、rankingではconversion modelの特徴量に使います。この分業により、既存systemの効率とLLMの一般知識を組み合わせています。

V1のoffline評価では、1件予測のSFTからGRPOへ進む学習とSemantic IDがRecallを改善しました。オンラインでは、慎重にquotaを調整した補助retrieval経路がUS Shopping sliceのRoASを相対4.94%改善しています。LLMを主役へ置き換えるより、既存systemが不足している信号を限定的に補わせる方が、本番導入へつなげやすいことを示した事例です。

## 参考文献

- Hui Yang et al., [Fine-Tuned LLM as a Complementary Predictor Improving Ads System](https://arxiv.org/abs/2605.27856), arXiv:2605.27856v1, 2026.
- 同論文の[HTML全文](https://arxiv.org/html/2605.27856)と[PDF](https://arxiv.org/pdf/2605.27856)
