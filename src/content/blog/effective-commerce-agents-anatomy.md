---
title: コマースエージェントの設計解剖――Anthropicの実務ガイドを読む
description: Anthropicのコマースエージェント設計ガイドをもとに、単一agentとskill、UI tool、遅延・cache、memory、安全策、evalのつながりと実装上の注意を解説します。
publishedAt: 2026-09-14
category: AI
tags:
  - AI Agent
  - Commerce
  - Claude
  - Agent Evaluation
  - System Design
draft: false
---

> **AI利用の明示**
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。導入判断に使う際は原典と自社の要件も確認してください。

AnthropicのAli ShazalとMatthew Koenが2026年9月2日に公開した「[A guide to the anatomy of effective commerce agents](https://claude.com/blog/the-anatomy-of-effective-commerce-agents)」は、オンラインでの購入・販売を助けるagentを、構成、速度と費用、本番運用の3面から整理した実務ガイドです。**価値は、agentの品質をmodel選択だけで説明せず、既存業務systemへの接続、権限の境界、UI、評価までを一つの設計として示した点**にあります。

対象は、商品検索やカート作成を助けるshopping agentと、売上分析、在庫、価格、campaignを扱うmerchant agentです。記事はAnthropicが顧客企業との導入経験から得た知見を述べていますが、比較の詳細な評価条件は公開されていません。以下では設計指針として読み、実測値を一般的な効果として扱わないようにします。

## 中心は「一つの会話を保つagent」

原典の基本形は、**一つのmodelによるagent loopに、業務別のskillと既存systemを呼ぶtoolを接続する**ものです。買い物の会話は「候補を比較する→在庫を確認する→返品条件を聞く→カートを直す」のように複数領域をまたぎます。領域ごとにsubagentへ引き継ぐと、好み、カート、直前に見せた商品などの文脈を毎回渡す必要があり、情報欠落、token消費、遅延が増えます。

Anthropicは複数の企業導入で、単一agent＋skillが巨大な単一promptや領域別subagentより品質で優れ、多くの場合費用や遅延も低かったと報告しています。ただし比較task、sample数、品質指標の内訳は示されていません。これは「subagentは常に不要」という実証ではありません。原典も、深い調査のように独立した作業領域が必要なtaskや、独自の規制・権限を持つ専用agentへの会話引き継ぎを例外に挙げています。

```text
利用者との会話と状態
       ↓
   一つのagent loop
    ├─ よく使う共通規則・検索手順：system prompt
    ├─ 低頻度の業務手順：skill
    └─ 検索・在庫・カート・価格など：既存systemを呼ぶtool
       ↓
   検証済みのUIと承認画面
```

### Promptとskillの境界は利用頻度で決める

skillを読むには追加のmodel turnが必要です。そのため原典は、trafficの約3分の1以上で必要な指示はsystem prompt、それ以外はskillを出発点として勧めています。これは固定の最適値ではなく、実際の利用分布とevalで調整するための目安です。商品検索は多くの買い物で使うためprompt側に置き、返品対応や長期的な購入計画はskillへ分ける、といった設計です。

安全・法務上の必須規則、brand制約、アレルギーのような重要な利用者情報は、頻度だけでskillへ追い出しません。ページ遷移などから次のskillを予測できるなら、初回model呼び出し前にharness側で読み込む方法も示されています。

## Toolは既存の業務logicを呼び、表示は型付きで渡す

agent用toolに検索順位、価格計算、配送可否を再実装すると、長年調整してきた業務logicと結果がずれます。原典の`search_products`は、既存検索systemがrankingした結果を返します。modelが判断するのは、利用者の目的に合う候補、表示件数、説明の仕方です。tool responseはmodelが判断に使うfieldへ絞り、不要な画像URLなどを大量にcontextへ入れないようにします。

商品carousel、旅程、比較表も自由文をclientで解析するのではなく、`present_products`などの**表示用tool**として定義します。modelが型付き引数を出し、serverが検証・補完してUI eventを発行します。tool呼び出しは会話履歴に残るので、「左から3番目の商品」と後で言われたとき、表示順を再参照できます。そのためtool引数の順序と実際の画面配置を一致させる必要があります。

この方式にはtrade-offもあります。serverが各引数を検証する間、細かい単位での表示更新は遅れます。原典は検証を省いて早くstreamする設定にも触れますが、その場合はschema保証が弱まります。表示の速さと不正なUI dataの拒否を、実際の失敗率で比較するべきです。

## 速度と費用は「完了したtask」で測る

一回の応答だけを速くしても、agentが追加のtool callを重ねれば全体は遅くなります。原典は完了までの時間を、各model turnの生成時間とtool処理時間の総和として考え、turn数、tool速度、token生成速度の3点を調整します。商品ページから起動したならその商品の情報を先に渡し、独立した検索は同じturnで並列化する。複数のbackend呼び出しをtool内で継ぎ足し続ける場合は、業務system側に一つの適切なendpointを用意する。これらはmodelの判断力を削らずに待ち時間を減らす設計です。

体感速度には、UI componentの段階的表示や「近くのホテルを探しています」のような短い進捗表示が効きます。原典では、通常のcommerce応答を約500〜700出力tokenとし、streamがなければspinnerが5秒以上続き得ると説明しています。ここで改善するのは主に**最初に画面が動くまでの時間**であり、taskの総処理時間とは区別します。

費用面では、繰り返されるprompt前半をcacheできるよう、contextを更新頻度順に置きます。

| 順序 | 内容 | 例 |
| --- | --- | --- |
| Global | session間で共通 | 共通prompt、tool定義 |
| Session | 同じ利用者の会話内で安定 | 利用者情報、会話履歴 |
| Volatile | turnごとに変わる | 現在時刻、表示中のページ |

時刻を共通promptの先頭へ置くと、prefix型cacheが毎回切れます。Anthropicは優れた導入例でcache hit率が90〜99%、約10万tokenではcache読取が約1.5〜2倍速いと述べています。ただし公開された測定条件やtraffic分布はなく、自社でもこの率に達する保証はありません。費用比較はmodel呼び出し単価ではなく、**完了taskあたりの費用、品質、p50/p99遅延**で行うべきだという指摘が重要です。

## 長期memoryは保存・抽出・読込を分ける

利用者の靴のサイズや店舗の担当者が毎週見る指標は、sessionをまたいで使いたい情報です。原典は、memoryをmodel内部の曖昧な「記憶」ではなく、出所sessionを持つ型付きrecordとして既存databaseに保存する構成を示します。販売側では共有accountではなく担当者ごとに記録し、読むときもその人の権限で絞ります。

書き込みは会話の途中でagentに保存toolを呼ばせず、turn後に別の抽出処理で非同期に行います。抽出対象を利用者とassistantの発言に限り、商品説明やreviewに書かれた内容を利用者の事実として覚えないようにします。Anthropicは自社のcommerce memory evalで、この方式が事実の再現率を13%高めたと報告していますが、比較対象、件数、絶対値は公開していません。

読込は、常に必要な少数の事実、requestに応じた事前取得、残りを引くlookup toolの3層です。保存してよい情報の種類、訂正・削除、保持期間も設計に含めます。特にアレルギーのような情報は有用性と個人情報保護の両方に関わるため、保存可否をprompt任せにせず書き込み経路で検証します。

## 最も重要な境界は「提案」と「適用」の間

購入、返金、値下げ、campaign開始は金銭や事業状態を変えます。原典の設計では、modelが実行を決定するtoolを直接持ちません。購買側ではcheckoutがcartと注文ボタンを表示し、決済はhost側が処理します。販売側の変更は一度stagingされ、server発行のIDを使い、人または既存の承認policyを経た`apply_change`だけが反映します。適用時には最新の上限を再確認します。

さらに次の条件をharnessのcodeで強制します。

- 書き込みや商品表示に使えるIDは、そのsessionでserverがagentへ返したものだけにする
- 購入数や値引き率の上限は各requestではなく、**変更後の状態**に対して確認し、同一sessionの並列書き込みを直列化する
- 商品説明、review、販売者messageなどを信頼できないdataとして整形・区切り、会話やtool callを偽装する文を指示として実行しない
- 手数料や規制上の開示文は承認済みの文面をserverから供給し、modelに言い換えさせない

この仕組みの要点は、modelの意図を推測して危険を止めることではなく、**権限のない状態変更がAPI上できないようにする**ことです。ただし、商品IDの出所検証や文字列の整形だけで、すべてのprompt injectionや業務上の誤判断が消えるわけではありません。後段のevalと運用監視が必要です。

## Evalは会話の筋書きより最終状態を見る

原典は、長い模擬会話だけで品質を測るのではなく、会話履歴、tool結果、cartなどを含む**snapshotから再開**する評価を勧めます。採点対象は、最後の表示と書き込み後の状態です。内部で通った手順を細かく固定すると、正しい別経路まで不合格になりやすいためです。模擬利用者との対話は未発見のcaseを探す用途に使い、見つけた失敗を再現可能なsnapshotへ移します。

特に用意したいのは、価格・在庫の根拠確認、過去の表示への参照、長く矛盾した履歴、商品説明に仕込まれた指示、上限超過、timeout、空の検索結果です。「応じるべき依頼」と「断るべき依頼」、「すぐ実行する依頼」と「確認が必要な依頼」を対にします。値下げと在庫見通しを同時に聞くような**領域横断の依頼**も、片方だけ答えて成功と判定しないようにします。

Anthropicは業務担当者や法務・顧客対応担当者とともに、利用者flowごとに50〜100件を出発点として作ることを勧めています。これは統計的に十分な件数の証明ではありません。利用頻度の高いcaseと全安全caseをCIに含め、変更したskillやtoolの周辺caseを追加し、全件は夜間やrelease前に走らせる、という運用までが提案されています。

## 実装を始めるなら

Anthropicは[shopping agentとmerchant agentの参照実装](https://github.com/anthropics/commerce-agents)を公開しています。retail、travel、telecom、entertainmentの例があり、READMEではPython 3.11以上とNode 22をdemoの前提としています。ただし架空企業のsampleで、認証や実際の決済・業務ruleは導入側の責任です。demoをそのまま本番systemへつなぐための完成品ではありません。

自社向けに進めるなら、まず購買側の「検索→比較→カート案」など、**一つのflowをread中心で実装**します。既存APIをtool化し、商品IDの出所、型付き表示、価格・在庫の根拠を検証します。次に、失敗・矛盾・複数領域のsnapshot evalを作り、品質と完了taskあたりの遅延・費用を測ります。その後にmemoryと販売側のstaged changeを加え、適用経路で権限、上限、承認を検証します。これは原典をもとにした段階的な導入案で、Anthropicが公開した実験手順ではありません。

原典の知見には適用範囲があります。単一agent優位、cache hit率、memory改善の数値はAnthropicの導入・内部評価に基づき、詳細な比較条件や独立した再現結果は示されていません。また、modelやAPIの設定、費用、遅延は更新されます。自社のcatalog規模、会話の複雑さ、地域ごとの個人情報要件、承認手順を入れたevalで判断する必要があります。

## まとめ

このガイドの核は、**一つの会話を保つagentにskillと既存systemのtoolを与え、表示・memory・変更適用をharness側の明示的な契約にする**ことです。速さはturn全体と体感表示の両方で、費用は完了task単位で、安全性は変更後の状態と承認経路で測ります。modelを交換しやすい設計にするには、promptを整えるだけでなく、業務上の正しさを検証できるevalを先に持つことが欠かせません。

## 参照

- Ali Shazal, Matthew Koen, [A guide to the anatomy of effective commerce agents](https://claude.com/blog/the-anatomy-of-effective-commerce-agents), Anthropic, 2026-09-02.
- Anthropic, [commerce-agents reference implementation](https://github.com/anthropics/commerce-agents), GitHub. 本記事ではREADMEに記載された構成とdemoの前提を参照。
