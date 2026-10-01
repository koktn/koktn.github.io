---
title: 専門家の知識で作る、文脈に即したLLM評価ベンチマーク
description: ドメイン専門家の知識をschemaと呼ばれる評価仕様に整理し、網羅性、多様性、内容と文体の現実性からLLM評価データを診断する手法を解説します。
publishedAt: 2026-09-30
updatedAt: 2026-10-01
category: AI
tags:
  - LLM
  - Evaluation
  - Synthetic Data
  - Benchmark
  - 論文解説
draft: false
---

> AI利用の明示<br>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文のv1と著者公開コードを確認して記載しています。利用時は原文も確認してください。

この研究では、専門家の知識から合成ベンチマークを作る手順を示しています。専門家が大量の評価例を書く代わりに、「誰が、何を、どの状況で評価したいか」をschemaと呼ばれる評価仕様にまとめる方法です。LLMはそのschemaをもとに評価例を生成し、研究チームが品質を診断します。

対象はKimberly Le Truongらによる「[A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance](https://arxiv.org/abs/2609.16592v1)」（[本文HTML](https://arxiv.org/html/2609.16592v1)、[PDF](https://arxiv.org/pdf/2609.16592v1)）です。2026年9月15日公開のarXiv v1で、EMNLP 2026 Findingsへの採択が記載されています。

論文では、米国の学校におけるソーシャルワークを事例にしています。単純なfew-shot生成とschema-guided生成を比べたところ、後者は内容の現実性と網羅性を改善しました。多様性には差がなく、文体の現実性については自動指標と専門家の評価が逆の結果になりました。

構造化したプロンプトだけで「妥当なベンチマーク」が自動的に完成するわけではありません。専門家の知識を再利用できる仕様に変え、自動指標と専門家の確認を組み合わせる枠組みです。

## なぜ汎用ベンチマークだけでは足りないのか

公開ベンチマークはモデル間の共通比較に便利ですが、特定組織の実運用をそのまま表すとは限りません。たとえば同じ「良い回答」でも、利用者、目的、専門領域、許容できない失敗、入力の書き方によって基準は変わります。

実際の利用ログを集めれば、現実に近いデータが得られます。しかし、導入前にはログがありません。導入後も個人情報や同意の問題が残ります。専門家が例を一件ずつ書けば質を高めやすいものの、時間と費用がかかります。

LLMを使う利点は、合成データの件数を増やしやすいことです。ただし、現場では起こりにくい状況をもっともらしく生成することがあります。例として与えるseedの表面的なパターンを繰り返し、評価したい要素の一部をほとんど含まない場合もあります。語調や詳しさも、実際の利用者とかけ離れることがあります。

この研究では、専門家の関与とLLMによる合成生成を組み合わせます。専門家が利用文脈を定義し、LLMがその仕様から評価例を増やします。

## 利用文脈をschemaにしてから生成する

提案手法では、利用文脈の定義、合成データの生成、品質の診断、専門家による確認を繰り返します。

<picture>
  <source media="(max-width: 600px)" srcset="/img/posts/expert-guided-benchmark-flow-mobile.svg">
  <img src="/img/posts/expert-guided-benchmark-flow.svg" alt="専門家の知識をpopulation、concept、instanceのschemaに整理し、LLMで評価データを生成して4指標と専門家の確認で診断する流れ" loading="lazy">
</picture>

*図1。[原論文Figure 1と§4](https://arxiv.org/html/2609.16592v1#S4)をもとに、本記事で処理関係を再構成した独自図。原図の転載ではありません。*

schemaは、評価の文脈を大きく3群へ分けます。

| schemaの群 | 明確にすること | ソーシャルワーク事例での例 |
| --- | --- | --- |
| Population | 実際に誰が使い、どの領域・期間・状況に制約されるか | 教室を観察するeducational facilitatorやmental health consultant |
| Concept | 測りたい能力やリスク、その定義、構成要素と取りうる値 | 良いreflective question、関係する人物、観察、解釈、説明の種類 |
| Instance | 一つの入力に必ず入る情報と、例ごとに変化させる情報 | 一人称のsingle-turn prompt、匿名化、具体的な観察、任意の背景情報 |

これに、実際の利用者を代表する例として与えるseedを加えます。事例研究では、先行研究で19人のソーシャルワーカーが作成・改良した16件のテストケースをseedに使いました。

schemaは、単なるトピック一覧ではありません。「子どもの行動」というトピックだけでなく、関係者や観察、仮説、利用者が相談する場面まで定義します。入力の必須情報と任意情報も分ける仕組みです。LLMはこのschemaをシステムプロンプトとユーザープロンプトで受け取り、生成文と各constituentのラベルをCSVで返します。constituentとは、schemaを構成する個々の要素を指します。

## 4つの指標は、別々の失敗を見つける

論文は、データセット単体から診断しやすい妥当性を二つに分けました。一つは内容的妥当性、もう一つは生態学的妥当性です。それぞれを、さらに二つの指標で測ります。

| validity | 指標 | 問うこと | 論文での実装 |
| --- | --- | --- | --- |
| 内容的妥当性 | Coverage | 評価対象の要素を取りこぼしていないか | schemaが定義するconstituentの組み合わせを、少なくとも`k`件含む割合 |
| 内容的妥当性 | Diversity | 同じ要素でも、意味の異なる表現や状況があるか | semantic embeddingに対するDCScore |
| 生態学的妥当性 | 内容の現実性 | 内容と状況が実際の利用場面に近いか | seed群と生成群の埋め込み分布間のSinkhorn distance |
| 生態学的妥当性 | 文体の現実性 | 言い回し、語調、長さが実際の利用場面に近いか | 文体 embedding空間で生成例と例として与えるseedのcosine distanceを変換 |

Coverageは、各要素を一度でも含むかだけを調べる指標ではありません。必要な要素の組み合わせを見ます。論文の主実験では`k = 1`とし、期待される各組み合わせが少なくとも1件あるかを測りました。100件のデータで比較手法の値がほぼ0にならないように選ばれた、保守的な設定です。本番評価に十分なサンプル数があることを意味しません。

内容の現実性では、seedと生成例のsemantic embedding分布を最適輸送で比べます。seedをbootstrapした「近い」基準と、無作為なベクトルを使う「遠い」基準によって、結果を0〜1へ正規化します。文体の現実性も値の範囲は0〜1です。ただし、専用の文体 embeddingを使い、意味の近さとは分けて測ります。

4指標は、一つの総合点へ足し合わせるものではありません。現実には使われない奇抜な例が増えた結果、diversityだけが高くなることもあります。反対に、realismが高くても、seedに似た狭いデータに偏っているかもしれません。論文では、関係者が何を優先するかに応じて各指標を解釈するよう求めています。

## ソーシャルワーカーが使うreflective-question支援の事例

対象組織では、資格を持つソーシャルワーカーが教室を観察します。その後、教師とのmeetingで使うreflective questionをLLMに相談する業務です。研究チームは、この用途を評価する入力プロンプトのデータセットを作りました。

研究チームは先行研究をもとにschemaの草案を作り、6人のソーシャルワーカーと非同期に改良しました。続くthink-aloud studyの参加者も、同じ組織の6人です。このうち3人はschemaの検証にも参加しています。二つの調査に参加した専門家は、重複を除くと9人でした。これは組織全体のおよそ4分の1にあたります。

比較対象は、2文のドメイン説明と`S`件のseedだけをLLMへ渡すfew-shot baselineです。生成にはClaude Sonnet 4.6を使い、temperatureを0.7に設定しました。比較手法とschema-guided方式で100件ずつ生成し、どちらにも3件の例として与えるseedを与えています。

参加者は各方式から5件ずつ、合計10件を確認しました。文体と内容のrealismを1〜5のLikert 規模で評価しています。参加者には比較手法とschemaという名称を伏せ、提示順も入れ替えました。

自動評価を含む実験には、Claude Haiku 4.5、Claude Sonnet 4.6、GPT-5.4 mini、GPT-5.5の4モデルを使っています。Coverageに必要なconstituent labelは、Claude Haiku 4.5をLLMによる評価役として付与しました。

## コンテンツとcoverageは改善し、文体評価は食い違った

専門家による評価は1〜5のLikert 規模、自動指標は0〜1です。数値の規模が異なるため、そのまま比べることはできません。

<picture>
  <source media="(max-width: 600px)" srcset="/img/posts/expert-guided-benchmark-results-mobile.svg">
  <img src="/img/posts/expert-guided-benchmark-results.svg" alt="few-shot baselineとschema-guided生成を、専門家によるstyle・content評価と4つの自動指標で比較した結果" loading="lazy">
</picture>

*図2。[原論文Table 1](https://arxiv.org/html/2609.16592v1#S6)の値を本記事でchart化。各方式100件、Claude Sonnet 4.6、seed 3件。専門家評価は6人、自動指標は0〜1。各ラベルの±表記も原表に従っています。*

schema-guided方式では、専門家による文体の評価が3.0から4.2、内容の評価が3.3から4.3へ上がりました。6人中5人が、全体としてschema-guided方式を選びました。自動指標でも、内容の現実性は0.73から0.77、coverageは0.16から0.23へ上がっています。ただし、どちらの方式も、期待される要素の組み合わせを4分の1未満しか網羅できませんでした。

Diversityは0.14対0.13で、明確な改善はありません。文体の現実性の自動指標も、比較手法の0.60に対してschema-guided方式は0.56でした。一方、専門家による文体の評価はschema-guided方式の方が高くなっています。

専門家はschema-guided方式の生成例を高く評価しました。組織が重視するstrength-basedな考え方や、教師に解決策を押し付けずreflectionを促す姿勢が表れていたためです。一方、自動指標が参照した16件のseedは、複数人が慎重に作ったテストケースでした。一人のworkerが日常的に書くプロンプトとは、文体が違う可能性があります。

二つの評価が食い違った点は、提案手法の弱点であると同時に重要な発見です。埋め込み上で近い文体と、組織の価値に合うと専門家が判断する文体は一致しません。自動指標だけでは、専門家の判断を代替できないことが分かります。

## Ablation studyで分かった、専門家に先に聞くべきこと

### Seedの数よりschemaの情報が有効

著者らはseed数を1〜16件で変え、4モデルについて各条件100例を5回生成しました。比較手法はseedが増えると内容の現実性とcoverageが改善しました。一方、schema-guided方式の品質は、seedなしの条件も含めてほぼ一定でした。専門家が例を増やすより先に、利用文脈をschemaで明示した方が効率のよい場合があると分かります。

ただし、seedが不要という結論ではありません。realism指標そのものが、実世界の代理としてseedを使います。また、1013通りのschemaのフィールド構成を各3回比較した実験では、seedがあるとcoverageは平均3.44%下がりました。著者らは、LLMが少数のseedに引っ張られ、constituentの範囲を広く探索しなくなった可能性を挙げています。これは観測結果から立てた仮説であり、因果関係を直接検証した結果ではありません。

### 目的に応じて専門家の時間を配分する

同じablationでは、systematized instanceとconstituentがcoverageに大きく寄与しました。事例研究では、特にactorとobservationの効果が大きく、これらのフィールドがあるとcoverageは最大12%増えています。一方、短いフィールドであるintended deployment populationは、内容の現実性への寄与が最も大きく、平均で11.87%上がりました。

専門家が使える時間は限られています。coverageを優先するなら、評価対象の要素と取りうる値を詳しく書きます。入力の必須部分と可変部分も具体化が必要です。内容の現実性を優先する場合は、実際に誰がどの状況で書くのかを先に詰めます。重視する品質に応じて、詳しく書くフィールドを選ぶ方法です。

### 一度に100件生成しても品質を保ちやすい

付録では、1回のリクエストで生成する件数`n`も比較しています。比較手法では、`n = 10`から`25`の間で生成例の平均長がおよそ半分になりました。`n = 100`では、一文または一問だけに短くなっています。schema-guided方式では、`n = 10...100`の範囲で長さが安定しました。schemaを与えると、大きなバッチでinstructionの影響が弱くなる問題も抑えられたと考えられます。

## 実務で試すための最小手順

著者はschema editor、生成、labeling、指標計算のコードを[GitHubでMIT Licenseとして公開](https://github.com/KimberlyTruong/expert-informed-eval-gen)しています。小規模に試す場合は、次の手順で進めます。

### 1. 評価に使う意思決定を一つに絞る

「chatbotの品質」のような広い目的ではなく、誰がどの場面で能力やリスクを判定し、その結果から何を決めるのかを書きます。データセットを作る前に、評価結果を使う人と意思決定を固定します。

### 2. 専門家とschemaを作る

リポジトリの`tools/schema_form.html`をブラウザで開くか、`configs/schemas/empty_schema.json`を埋めます。少なくとも、次の項目をバージョン管理します。

```text
capability
systematizedConcept
deploymentPopulation
contextualConstraints
systematizedInstance
constituents[].required
constituents[].possibleValues
seedExamples[]
```

各constituentの組み合わせが、現実に成立するかも確認します。単純なCartesian productには、現場で起こり得ない組み合わせや不適切な組み合わせが含まれるためです。公開コードでは、非現実的な組み合わせをcoverageの分母から外すexclusion listも使えます。

### 3. 小さなバッチで生成する

公開実装はPython 3.10以上とOpenAIまたはAnthropicのAPI keyを必要とします。READMEの例を短くすると、次の形です。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/generation.py \
  --schema-file configs/schemas/social-work/schema_data-5-7.json \
  --output-dir data/generated/pilot \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --n 20 \
  --runs 1 \
  --fixed-seed-count 3 \
  --random-seed 42
```

これは著者実装の使い方を示す例です。本記事ではAPIを実行していません。モデル名、料金、API仕様は変わる可能性があります。実行時点の各providerの公式ドキュメントも確認してください。

### 4. ラベルを付け、4指標で診断する

生成時にラベルを出させるか、別のLLMによる評価役でconstituent labelを付けます。その後、公開スクリプトで指標を計算します。

```sh
python scripts/metrics.py \
  --schema configs/schemas/social-work/schema_data-5-7.json \
  --dataset data/generated/pilot/example_labeled.csv \
  --coverage-k 1 \
  --embedding-model all-mpnet-base-v2
```

一つの総合scoreを作るよりも、不足している組み合わせ、重複、seedから遠い例、文体の外れ値を確認対象として抽出します。指標が低い場合は、生成モデルだけを原因と決めつけてはいけません。schemaの漏れ、seedの偏り、labeling errorも調査対象です。

### 5. blind reviewを行い、schemaを修正する

ドメイン専門家には生成方式を伏せ、同じ件数を無作為な順で見せます。記録するのは、現場で実際に起こる状況か、起きたとしてLLMに相談する場面かという判断です。組織の価値や専門職としての判断に合うかも確認します。緊急対応や人間の判断が必要で、AI評価の対象外にすべき例も分けます。不足しているactorや状況、少数グループがないかという観点も必要です。

個別の生成例だけを直して終わらせてはいけません。指摘をpopulation、constituent、instance rule、exclusionのどこへ反映するか判断します。schemaのrevision、プロンプト、モデルのバージョン、seed集合、乱数シード、生成日時を保存すれば、変更前後を比較できます。

## 再現性と適用範囲の限界

この論文の結果を、別のドメインへそのまま一般化することはできません。実証実験は、一つのソーシャルワーク組織と一つのタスクに限られています。参加した専門家は重複を除いて9人で、human studyへの参加者は6人でした。参加者一人が評価したのは、各方式5例ずつです。主なデータセットも各100件で、利用できたseedは16件でした。

付録には、healthcareとpolitical misinformationのschema例もあります。ただし、同じ実験で有効性を検証したわけではありません。4指標はデータセット単体の診断に使うもので、完全なevaluation validityを保証しません。model response、rubric、annotator agreement、deployment outcomeは評価の対象外です。

内容の現実性と文体の現実性は、seedが実際の利用場面を代表するという仮定に依存します。embedding modelを変えても、比較手法とschema-guided方式の順位は保たれました。一方で、内容の現実性の絶対値と差の大きさは変わっています。

公開リポジトリには、schema、プロンプト、生成スクリプト、指標計算スクリプトが含まれています。一方、Python dependencyのバージョンは固定されていません。論文で生成した全データと、7000万トークン分のAPI実行結果も公開されていません。APIモデルも時間とともに変わるため、論文の数値を完全に再現できるpackageではありません。

事例研究では、子どもの行動、教育、メンタルヘルスという機微な文脈を扱っています。合成データであっても、seedやschemaに実在人物を特定できる情報が残れば、privacy riskは消えません。先に公開範囲、アクセス制御、retention、匿名化、専門家の同意を設計する必要があります。合成例を実際の事例記録として扱ってはいけません。

## まとめ

このframeworkは、LLMへ「多様で現実的な評価例を100件作って」と頼むためのprompt 手法ではありません。専門家の知識をpopulation、concept、instanceへ分けてschemaにまとめます。そのschemaを、生成プロンプトとcoverageの期待空間の両方に使う手順です。

データの品質は、coverage、diversity、内容の現実性、文体の現実性の4指標で個別に診断します。seedを増やす前に、目的に関係するschemaのフィールドを詳しく書きます。自動指標と専門家の評価が食い違った箇所も、問題を見つける手掛かりになります。

schema-guided方式は、この事例研究で内容の現実性とcoverageを改善しました。しかし、diversityは改善していません。期待される組み合わせの多くは網羅できず、文体の自動指標も専門家の判断と一致しませんでした。専門家を生成作業から外すのではなく、専門家の時間を利用文脈の定義と検証へ振り向けることが大切です。

## 参照資料

- Truong et al., [A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance, arXiv:2609.16592v1](https://arxiv.org/abs/2609.16592v1), 2026-09-15.
- [論文HTML](https://arxiv.org/html/2609.16592v1)／[PDF](https://arxiv.org/pdf/2609.16592v1)。frameworkは§4、実験条件は§5、expert studyは§6、ablationは§7とAppendix Gを参照。
- KimberlyTruong, [expert-informed-eval-gen](https://github.com/KimberlyTruong/expert-informed-eval-gen)。schema editor、生成、labeling、4指標の実装。MIT License。
