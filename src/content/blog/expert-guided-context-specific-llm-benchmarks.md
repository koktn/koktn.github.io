---
title: 専門家の知識でLLM評価benchmarkを作る――context-specificな合成data生成
description: domain expertの知識をschemaへ落とし込み、coverage、diversity、content realism、stylistic realismでLLM評価dataを診断するframeworkを解説します。
publishedAt: 2026-09-30
category: AI
tags:
  - LLM
  - Evaluation
  - Synthetic Data
  - Benchmark
  - 論文解説
draft: false
---

> **AI利用の明示**<br>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文のv1と著者公開codeを確認して記載しています。利用時は原文も確認してください。

**この研究の価値は、domain expertに大量の評価例を書いてもらう代わりに、誰が・何を・どの状況で評価したいかをschemaへ構造化し、その情報で合成benchmarkを生成・診断する一連の手順を示したことです。**

対象はKimberly Le Truongらによる「[A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance](https://arxiv.org/abs/2609.16592v1)」（[本文HTML](https://arxiv.org/html/2609.16592v1)、[PDF](https://arxiv.org/pdf/2609.16592v1)）です。2026年9月15日公開のarXiv v1で、EMNLP 2026 Findingsへの採択が記載されています。

論文は米国のschool social workをcase studyにし、単純なfew-shot生成と比べて、schema-guided生成がcontent realismとcoverageを改善したと報告します。一方、diversityには差がなく、stylistic realismは自動metricと専門家評価が逆方向でした。つまり、structured promptを使えば自動的に「妥当なbenchmark」が完成するという研究ではありません。**専門家の知識を再利用可能な仕様へ変え、自動metricと専門家reviewを組み合わせるframework**として読む必要があります。

## なぜ汎用benchmarkだけでは足りないのか

公開benchmarkはmodel間の共通比較に便利ですが、特定組織の実運用をそのまま表すとは限りません。たとえば同じ「良い回答」でも、利用者、目的、専門領域、許容できない失敗、入力の書き方によって基準は変わります。

実際の利用logを集めれば現実に近づきますが、deployment前には存在せず、個人情報や同意の問題もあります。domain expertが一件ずつ例を書く方法は質を高めやすい一方、時間と費用がかかります。LLMによる合成dataはscaleしますが、次のような失敗を起こします。

- 現場では起こりにくい状況をもっともらしく生成する
- seed exampleの表面的なpatternを繰り返す
- 評価したいfactorの一部をほとんど含まない
- 実際の利用者とは違う語調や詳しさで書く

この研究は、専門家の関与と合成生成を二者択一にせず、専門家にはcontextを定義してもらい、LLMにはその仕様から例を増やしてもらう構成を採ります。

## 全体像：contextをschemaにしてから生成する

frameworkは、contextの定義、合成data生成、data品質の診断、専門家による確認を一つのloopとして扱います。

<picture>
  <source media="(max-width: 600px)" srcset="/img/posts/expert-guided-benchmark-flow-mobile.svg">
  <img src="/img/posts/expert-guided-benchmark-flow.svg" alt="専門家の知識をpopulation、concept、instanceのschemaへ整理し、LLMで評価dataを生成して4指標と専門家reviewで診断する流れ" loading="lazy">
</picture>

*図1：[原論文Figure 1と§4](https://arxiv.org/html/2609.16592v1#S4)をもとに、本記事で処理関係を再構成した独自図。原図の転載ではありません。*

schemaは、評価contextを大きく3群へ分けます。

| schemaの群 | 明確にすること | social-work事例での例 |
| --- | --- | --- |
| Population | 実際に誰が使い、どの領域・期間・状況に制約されるか | classroomを観察するeducational facilitatorやmental health consultant |
| Concept | 測りたいcapabilityやrisk、その定義、構成要素と取りうる値 | 良いreflective question、関係するactor、観察、解釈、説明の種類 |
| Instance | 一つの入力に必ず入る情報と、例ごとに変化させる情報 | 一人称のsingle-turn prompt、匿名化、具体的な観察、任意の背景情報 |

これに、実際の利用者が書いたrepresentativeなseed exampleを加えます。case studyでは、先行研究で19人のsocial workerが作成・改良した16件のtest caseをseedに使いました。

重要なのは、schemaが単なるtopic一覧ではないことです。たとえば「子どもの行動」というtopicだけでなく、誰が関わるか、どんな観察や仮説があるか、利用者が何を相談する場面か、必須情報と任意情報は何かまで定義します。LLMはこのschemaをsystem promptとuser promptで受け取り、生成文と各constituentのlabelをCSVとして返します。

## 4つの指標は、別々の失敗を見つける

論文はdataset artifactだけから直接診断しやすいvalidityを、content validityとecological validityに分け、さらに4指標へ落とし込みます。

| validity | 指標 | 問うこと | 論文での実装 |
| --- | --- | --- | --- |
| Content validity | Coverage | 評価対象のfactor群を取りこぼしていないか | schemaが定義するconstituentの組み合わせを、少なくとも`k`件含む割合 |
| Content validity | Diversity | 同じfactorでも意味的に異なる表現や状況があるか | semantic embeddingに対するDCScore |
| Ecological validity | Content realism | 内容と状況が実利用に近いか | seed群と生成群のembedding分布間のSinkhorn distance |
| Ecological validity | Stylistic realism | 言い回し、tone、長さが実利用に近いか | style embedding空間で生成例とseed例のcosine distanceを変換 |

Coverageは「各factorを一度でも含むか」だけでなく、必要なfactorの組み合わせを見ます。論文の主実験は`k = 1`、つまり期待される各組み合わせが少なくとも1件あるかを使います。100件のdataでbaselineの値がほぼ0にならないように選ばれた保守的な設定であり、productionの十分なsample数を意味しません。

Content realismは、seedと生成例のsemantic embedding分布を最適輸送で比較します。seedをbootstrapした「近い」基準と、random vectorを使う「遠い」基準で0〜1へ正規化します。Stylistic realismも0〜1ですが、専用のstyle embeddingを使い、semanticな近さとは分けます。

4指標は総合点の構成要素ではありません。高いdiversityが、現実には使われない奇抜な例によって生じることもあります。高いrealismが、seedに似すぎた狭いdataを意味することもあります。論文も、すべてを最大化するのではなく、stakeholderの優先事項に応じて解釈するよう求めています。

## Case study：social workerが使うreflective-question支援

case studyの組織では、licensed social workerがclassroomを観察し、teacherとの今後のmeetingに使うreflective questionをLLMへ相談します。研究は、この用途を評価する入力promptのdatasetを作りました。

まず先行研究をもとにschemaのdraftを作り、6人のsocial workerと非同期に改良しました。続くthink-aloud studyには同じ組織の6人が参加し、そのうち3人はschema validationにも参加しています。二つのstudyを合わせたdistinctな専門家は9人で、組織のおよそ4分の1です。

比較対象は、2文のdomain説明と`S`件のseedだけをLLMへ渡すfew-shot baselineです。主なhuman studyの条件は次の通りです。

- 生成modelはClaude Sonnet 4.6、temperatureは0.7
- baselineとschema-guided方式で各100件を生成
- 両方式とも3件のseed exampleを使用
- 参加者は各方式から5件ずつ、合計10件をreview
- styleとcontentのrealismを1〜5のLikert scaleで評価
- baseline／schemaという名称は伏せ、提示順も入れ替えた

自動評価を含む広い実験では、Claude Haiku 4.5、Claude Sonnet 4.6、GPT-5.4 mini、GPT-5.5の4 modelを使っています。coverage用のconstituent labelはClaude Haiku 4.5をLLM judgeとして付与しました。

## 結果：contentとcoverageは改善、style評価は食い違った

結果を見るときは、人間のLikert評価と0〜1の自動metricが別scaleであることに注意してください。

<picture>
  <source media="(max-width: 600px)" srcset="/img/posts/expert-guided-benchmark-results-mobile.svg">
  <img src="/img/posts/expert-guided-benchmark-results.svg" alt="few-shot baselineとschema-guided生成を、専門家によるstyle・content評価と4つの自動metricで比較した結果" loading="lazy">
</picture>

*図2：[原論文Table 1](https://arxiv.org/html/2609.16592v1#S6)の値を本記事でchart化。各方式100件、Claude Sonnet 4.6、seed 3件。expert評価は6人、自動metricは0〜1。各labelの±表記も原表に従っています。*

schema-guided方式は、専門家評価のstyleで3.0から4.2、contentで3.3から4.3へ上がり、6人中5人が全体としてこちらを好みました。自動metricでもcontent realismは0.73から0.77、coverageは0.16から0.23へ上がりました。ただし、どちらも期待されるfactor組み合わせの4分の1未満しかcoverしていません。

Diversityは0.14対0.13で、明確な改善はありません。さらに、stylistic realismの自動metricはbaselineの0.60に対してschemaが0.56と低い一方、専門家のstyle評価はschemaの方が高くなりました。

専門家はschema-generated exampleについて、組織が重視するstrength-basedな考え方や、teacherへ直接解決策を押し付けずreflectionを促す姿勢が現れていると評価しました。反対に、自動metricが参照した16件のseedは複数人で慎重に作られたtest caseであり、一人のworkerが日常的に書くpromptとはstyleが違う可能性があります。

この食い違いはframeworkの弱点であると同時に、重要な発見です。embedding metricが近いと判定するstyleと、組織の価値に沿っていると専門家が判断するstyleは同じではありません。自動metricをexpert judgmentの代替にできない理由が、実験内に現れています。

## Ablationから分かる「何を先に聞くべきか」

### Seedの数よりschemaの情報が効く

著者らはseed数を1〜16件で変え、4 modelについて各条件100例を5回生成しました。baselineはseedが増えるとcontent realismとcoverageが改善しますが、schema-guided方式はseedなしを含めて品質がほぼ一定でした。domain expertが書く例を増やす前に、contextをschemaとして明示する方が効率的な場合があることを示します。

ただし「seedは不要」という結論ではありません。realism metric自体がseedを実世界のproxyとして使います。また、1013通りのschema field構成を各3回比較した実験では、seedの存在がcoverageを平均3.44%下げました。著者らは、少数のseedにLLMが引っ張られ、constituent空間を広く探索しなくなるためではないかと推測しています。これは観測された関連に対する仮説で、因果機構を直接検証した結果ではありません。

### 目的に応じて専門家の時間を配分する

同じablationでは、systematized instanceとconstituent、特にcase studyのactorとobservationがcoverageへ大きく寄与し、fieldがある場合に最大12%増えました。一方、短いfieldであるintended deployment populationはcontent realismへの寄与が最も大きく、平均+11.87%でした。

したがって、限られた時間でcoverageを優先するなら、評価対象のfactorと各値、入力の必須・可変部分を詳しくします。content realismを優先するなら、「実際に誰が、どの状況で書くのか」を先に詰めます。すべてのfieldを同じ深さで埋めるのではなく、欲しいquality dimensionから逆算する考え方です。

### 一度に100件生成しても崩れにくい

付録では、一回のrequestで生成する件数`n`も比較しています。baselineは`n = 10`から`25`の間で平均example長がおよそ半分になり、`n = 100`では一文または一問だけへ劣化しました。schema-guided方式は、試した`n = 10...100`で長さが安定しました。schemaは内容の制御だけでなく、大きなbatchでinstructionが薄まる失敗も抑えたと解釈できます。

## 実務で試す最小workflow

著者はschema editor、生成、labeling、metric計算のcodeを[GitHubでMIT Licenseとして公開](https://github.com/KimberlyTruong/expert-informed-eval-gen)しています。次の順序で小さく試せます。

### 1. 評価するdecisionを一つに絞る

「chatbotの品質」のような広い目的ではなく、「誰が、どの場面で、どのcapabilityまたはriskを判定し、その結果で何を決めるか」を書きます。datasetを作る前に、評価結果の利用者と意思決定を固定します。

### 2. Schemaを専門家と作る

repositoryの`tools/schema_form.html`をbrowserで開くか、`configs/schemas/empty_schema.json`を埋めます。最低限、次をversion管理します。

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

各constituentの組み合わせが現実に成立するかも確認します。単純なCartesian productには、現場では不可能または不適切な組み合わせが含まれるためです。公開codeは、非現実的な組み合わせをcoverageの分母から外すexclusion listにも対応します。

### 3. 小さなbatchを生成する

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

これは著者実装の使い方を示す例で、本記事ではAPIを実行していません。model名、料金、API仕様は変わりうるため、実行時点のprovider documentationも確認してください。

### 4. Labelと4指標を診断に使う

生成時にlabelを出させるか、別のLLM judgeでconstituent labelを付けます。その後、公開scriptでmetricを計算します。

```sh
python scripts/metrics.py \
  --schema configs/schemas/social-work/schema_data-5-7.json \
  --dataset data/generated/pilot/example_labeled.csv \
  --coverage-k 1 \
  --embedding-model all-mpnet-base-v2
```

総合scoreを作るより、missing combination、重複、seedから遠い例、styleの外れ値をreview対象として抽出します。metricが低いときは生成modelだけでなく、schemaの漏れ、seedの偏り、labeling errorも疑います。

### 5. Blind reviewして修正loopを回す

domain expertには生成方式を伏せ、同じ件数をrandomな順で見せます。少なくとも次を記録します。

- 現場で実際に起こるか
- 起きてもLLMへ相談する場面か
- 組織の価値やprofessional judgmentに沿うか
- 緊急対応や人間の判断が必要で、AI評価の対象外にすべきか
- 不足しているactor、状況、少数groupがないか

指摘を個別exampleの修正で終わらせず、population、constituent、instance rule、exclusionのどこへ反映するかを判断します。schema revision、prompt、model version、seed集合、random seed、生成日時を保存すると、変更前後を比較できます。

## 再現性と適用範囲の限界

この論文の結果を別domainへそのまま一般化はできません。

- empirical studyは一つのsocial-work組織と一つのtaskに限られる
- distinctな専門家は9人、human studyは6人で、各人が各方式5例ずつ見た小規模な定性評価
- 主なdatasetは各100件、利用できたseedは16件
- healthcareとpolitical misinformationのschema例は付録にあるが、同じ実験で有効性を検証していない
- 4指標はdataset単体のdiagnosticであり、model response、rubric、annotator agreement、deployment outcomeまで含む完全なevaluation validityを保証しない
- content／style realismはseedが実利用を代表するという仮定へ依存する
- embedding modelを変えてもbaselineとschemaの順位は保たれたが、content realismの絶対値と差の大きさは変化した

公開repositoryにはschema、prompt、generation・metric scriptがありますが、Python dependencyのversion lockはなく、論文の全生成dataと70 million token分のAPI実行結果は含まれていません。API modelも時間とともに変わりうるため、論文の数値を完全に再現できるpackageではありません。

また、case studyは子どもの行動、教育、mental healthに関わる機微なcontextです。合成dataでも、seedやschemaに実在人物を特定できる情報が残ればprivacy riskは消えません。公開範囲、access control、retention、匿名化、専門家の同意を先に設計し、合成例を実際のcase記録として扱わないことが必要です。

## まとめ

このframeworkが提案するのは、LLMへ「多様で現実的な評価例を100件作って」と頼むprompt techniqueだけではありません。

- 専門家の知識をpopulation、concept、instanceへ構造化する
- schemaを生成promptとcoverageの期待空間の両方に使う
- coverage、diversity、content realism、stylistic realismを別々に診断する
- seedを増やす前に、目的に効くschema fieldを深くする
- 自動metricと専門家評価が食い違う箇所を、失敗signalとして扱う

schema-guided方式は、このcase studyでcontent realismとcoverageを改善しました。しかし、diversityは改善せず、期待される組み合わせの多くは未coverageで、style metricは専門家判断と一致しませんでした。最も重要な教訓は、専門家を生成作業から外すことではなく、**専門家の時間をcontext定義とvalidationへ集中させること**です。

## 参照資料

- Truong et al., [A Framework for Generating Valid Context-Specific Benchmarks through Expert Guidance, arXiv:2609.16592v1](https://arxiv.org/abs/2609.16592v1), 2026-09-15.
- [論文HTML](https://arxiv.org/html/2609.16592v1)／[PDF](https://arxiv.org/pdf/2609.16592v1) — frameworkは§4、実験条件は§5、expert studyは§6、ablationは§7とAppendix G。
- KimberlyTruong, [expert-informed-eval-gen](https://github.com/KimberlyTruong/expert-informed-eval-gen) — schema editor、生成、labeling、4指標の実装。MIT License。
