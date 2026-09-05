---
title: Warpのself-improving agent設計――2つのSkillでfeedbackを継続的な改善に変える
description: WarpがClaude上で構築したself-improving agentを、base skill、improver skill、human feedback、評価と実装の観点から解説します。
publishedAt: 2026-09-05
category: AI
tags:
  - AI Agent
  - Agent Skills
  - Claude
  - Warp
draft: false
---

> **AI利用の明示**  
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。主張と数値は参照元を確認していますが、公開前に原文も確認してください。

今回取り上げるのは、Claude公式blogで2026年8月26日に公開された「[How Warp builds self-improving agents on Claude](https://claude.com/blog/how-warp-builds-self-improving-agents-on-claude)」です。AnthropicのMichael Segner氏が、Warpによるself-improving agentの設計と運用例を紹介しています。

この記事の要点は、**taskを実行するbase skillと、human feedbackからbase skillの小さな修正を提案するimprover skillを分け、通常のPull Request workflowで改善を管理する**というものです。

## 課題：agentへのfeedbackが次のsessionに残らない

Warpでは、社内のcode review agentが役に立たないcommentや低品質なoutputを返し、engineerから不満が出ていました。失敗例を見ながら人がpromptを書き直したり、`AGENTS.md`のようなcontext fileを改善したりすると一時的には良くなります。しかし、taskごとのfeedbackはsession終了時に消え、修正作業も人手のままなのでscaleしません。

初回のpromptでtaskの80%を正しく処理できたとしても、残り20%の不要な指摘が繰り返し現れれば、userにとってはnoiseになります。必要なのは「強いpromptを一度書くこと」だけでなく、productionで得たfeedbackを次の実行へ安全に反映するloopです。

## 2-skill architecture

Warpの構成には、役割の異なる2つのSkillがあります。

| Component | 実行timing | 役割 |
| --- | --- | --- |
| inner / base skill | taskごと | domain knowledgeと実行手順をagentへ与える |
| human feedback | task後 | outputの良否と、その理由を普段の作業場所で記録する |
| outer / improver skill | schedule実行 | feedbackを集約し、base skillへの小さな変更を提案する |
| human review | Skill更新時 | 提案されたdiffを確認し、mergeするか判断する |

処理のflowは次のようになります。

```text
base skillを使ってtaskを実行
  → userやmaintainerが具体的なfeedbackを残す
  → improver agentがfeedbackを定期収集
  → 現在のbase skillと比較し、最小のpatchを作る
  → Pull Requestを作成
  → 人間がreview・approve・merge
  → 次のtaskから更新済みbase skillを利用
```

重要なのは、improver agentがbase skillを直接書き換えて即時反映するのではない点です。Skillはplain fileなので、diffを確認でき、version control、review、rollbackという既存のsoftware development workflowへ載せられます。

## Warpのissue triage agentでの実例

紹介されているissue triage agentは、新しいGitHub issueをtriggerに起動します。codebaseを調べ、complexityとfeasibilityを分析し、labelと修正方針を提案します。base skillには、各labelの意味や調査手順が書かれています。

あるissueでagentは主要な判断には成功しましたが、仕様作成へ進めることを示す`ready to spec` labelを付け忘れました。maintainerはissue上で、付けるべきlabelだけでなく、なぜ必要なのかもfeedbackとして残しました。

その後、Warpのorchestration platformであるOz上のscheduled agentが動きます。SkillにbundleされたPython scriptでfeedback付きissueを取得してJSONへまとめ、improver agentが読み込みます。agentはfeedbackから具体的なsignalを取り出し、`ready to spec`を適用する条件をbase skillへ加える小さな変更を提案してPull Requestを作成します。最終的に人間がreviewしてmergeし、次回から新しい知識が使われます。

公開されている「[Ambient Agents Demo - GitHub Actions](https://github.com/warpdotdev/warp-agents-demo-github-issue-triage)」では、GitHub ActionからWarp agentを起動する構成を確認できます。ただし、このrepositoryは2026年6月2日にarchiveされておりread-onlyです。Claude公式blogの完全なself-improvement loop一式が、そのまま実行可能な形で公開されているわけではありません。

## Skillを書くときのポイント

Warpの推奨事項は、次のように整理できます。

### Rulesではなくprinciplesを書く

変数名の全patternを列挙するようなrulesより、「重複したcodeを探す」のようなprincipleを与えます。agentが未知のcaseへgeneralizeしやすくなるためです。

### Whyを含める

指示だけでなく理由を書くと、agentは表面的なpattern matchingではなく、目的に沿って判断できます。feedbackにも「間違い」だけでなく「なぜ間違いか」を含めます。

### Feedbackの摩擦を下げる

専用formへ転記させず、Pull Request commentやissue commentなど、普段の作業場所でfeedbackを収集します。収集が面倒だと改善のsignalが途切れます。

### Skillを小さく保つ

すべてを`SKILL.md`へ詰め込まず、必要なreferenceやscriptを別fileにしてprogressive disclosureを使います。[Claude PlatformのAgent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)でも、必要なfileだけをon-demandで読み込む構成が説明されています。

### 量より質。ただし質の高い量は役立つ

単純なthumbs-up／downを大量に集めるより、domain expertによる具体的なfeedbackの方が改善に使えます。一方で、質の揃ったsignalならcorpusが増えるほど繰り返しpatternを見つけやすくなります。

### Improverへ投資する

domain knowledgeを持つbase skillはtaskごとに変わりますが、feedback収集、差分抽出、最小patch作成、評価、PR作成というimprover側の処理は再利用しやすい部分です。

## Skillとmemoryは別物

Claude公式blogは、Skillとmemoryの違いも明確にしています。

- **Skill**：taskの進め方を表すproceduralで比較的stableな知識。意図的なreviewを経て更新する
- **Memory**：inference時にagentが書き込み、sessionやuserに関する状態として継続的に変化する

すべてのfeedbackをmemoryへ自動保存すると、誤った指示や一時的な例外が蓄積する恐れがあります。組織の標準手順として繰り返し使う知識は、review可能なSkillとして管理する方が向いています。

## どこまで効果が検証されているか

元記事には、Warpで月間80万人のdeveloperがbuildしている、Fortune 500企業の56%が利用している、Warp内のClaude Code sessionが累計1,000万回・週40万回以上、Warp Agentのconversationが累計4,000万件、といった利用規模が掲載されています。

ただし、これらはWarpやClaudeの利用規模であり、2-skill loopによる品質改善を直接測った値ではありません。記事では、導入前後の正解率、不要comment率、Skill変更の採用率、control groupとの比較などは公開されていません。

したがって、このsourceから言えるのは「Warpが複数のagentでこのpatternをproduction運用している」「issue triageの改善例がある」というcase studyまでです。self-improvementの改善率や、別組織でも同じ効果が出ることが定量的に実証された、とまでは言えません。

## 実際に実装するには

以下は、元記事の設計を一般的なcode review agentへ適用する実装案です。Warpの内部platformやimprover promptは公開されていないため、完全再現ではありません。

### 1. Base skillと評価対象を固定する

まず、1つの狭いtaskから始めます。たとえば「Pull Requestにsecurity上の問題がないかreviewする」のように対象を限定します。base skill、model version、tool、repository revisionを実行logへ保存し、どの条件で生成されたoutputか追跡できるようにします。

### 2. Feedback schemaを決める

thumbs-up／downだけでなく、期待する動作と理由を保存します。

```ts
type AgentFeedback = {
  taskId: string;
  skillRevision: string;
  outputId: string;
  verdict: 'useful' | 'not-useful' | 'partially-useful';
  expectedBehavior: string;
  rationale: string;
  authorRole: 'maintainer' | 'domain-expert' | 'contributor';
  sourceUrl: string;
  createdAt: string;
};
```

feedback本文はuntrusted inputとして扱います。issueやPRに書かれた命令をそのままSkillへコピーせず、誰のfeedbackを採用対象にするか、どのrepositoryやlabelを収集するかをallowlistで制限します。

### 3. Improverをschedule実行する

毎回のtask終了時ではなく、日次または週次でまとめて処理します。単発の例外をgeneral ruleにしにくく、関連するfeedbackを比較できるためです。

```text
承認されたfeedbackを取得
  → 重複・矛盾・一時的な例外を分類
  → 現在のbase skillで既に扱えるか確認
  → 繰り返し発生し、generalizeできるsignalを選ぶ
  → 1つの目的に絞った最小patchを生成
  → evalを実行
  → evidenceと結果を添えてdraft PRを作成
```

改善内容を一度に詰め込まず、1つのPRを1つのbehavior changeへ絞ると、失敗時の原因特定とrevertが容易になります。

### 4. Eval harnessを先に作る

improverが「改善した」と自己評価するだけでは不十分です。過去の成功例、失敗例、境界例からgolden corpusを作り、現行Skillと変更後Skillを同じmodel・設定で比較します。

code reviewなら、次のmetricが候補になります。

- 実際に採用されたcommentの割合
- false positiveとなった指摘の割合
- 重大な問題のrecall
- maintainerによるusefulness評価
- token／API costとlatency
- time to mergeなどsystem全体のmetric

改善対象のmetricだけでなく、既に成功していたcaseを壊していないかregression testも行います。domainが機械的に検証しにくい場合は、golden outputを使える部分をdeterministic evalにし、主観評価は少数のdomain expertへ限定します。

### 5. Pull Requestを安全装置にする

improverが作るPRには、少なくとも次を含めます。

- 根拠となったfeedbackへのlink
- 問題が繰り返しpatternなのか、単発なのか
- Skillの変更前後のdiff
- eval結果とregressionの有無
- 想定する影響範囲
- rollback方法

`CODEOWNERS`などでdomain ownerのapproveを必須にし、agent自身にはmerge権限を与えない構成が安全です。feedbackが誤っている前提でsanity checkし、機密情報、個人情報、prompt injectionをSkillへ取り込まないreviewも必要です。

### 6. Crawl–walk–runで導入する

最初はimproverにreportだけ作らせ、次にdraft PRまで許可し、evalと人間の承認が安定してから対象agentを増やします。次の運用metricを継続して確認します。

- feedbackが付いたtaskの割合
- domain expert由来のfeedback数
- improverが提案したPRの採用率
- Skill更新後のregression率
- task品質、time to merge、costの長期trend

## Limitationと改善課題

このpatternには、まだ次の課題があります。

- 誤ったfeedback、組織内の少数意見、prompt injectionをどうfilterするか
- 最新のfeedbackへ過剰適合し、以前のcaseを壊すcatastrophic forgettingをどう防ぐか
- 複数の矛盾するprincipleをどう統合するか
- base skillが肥大化したとき、何をreferenceへ移し何を削除するか
- model更新による変化とSkill更新の効果をどう分離するか
- user満足度とcost、latencyなど複数metricのtrade-offをどう判断するか

特に、「agentが自分を改善する」という表現から完全自動の自己書き換えを想像しないことが重要です。Warpの設計は、human feedbackを収集し、agentが小さなSkill変更を提案し、人間がreviewしてmergeする**human-in-the-loopの継続改善**です。

## まとめ

Warpの事例で最も再利用しやすいのは、base skillとimprover skillを分けたこと、feedbackを普段の作業場所で集めたこと、Skill更新をPull Requestとして扱ったことです。

agentを改善するには、feedbackを増やすだけでなく、どのsignalを信頼し、どのevalで改善を判定し、誰が変更を承認するかを設計する必要があります。Skillをcodeと同じようにversion管理することで、agentの「学習」を監査可能なengineering processへ変えられます。

## 参照

- Michael Segner, [How Warp builds self-improving agents on Claude](https://claude.com/blog/how-warp-builds-self-improving-agents-on-claude), Claude Blog, 2026-08-26.
- Anthropic, [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview).
- Anthropic, [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).
- Warp, [Ambient Agents Demo - GitHub Actions](https://github.com/warpdotdev/warp-agents-demo-github-issue-triage)（2026-06-02にarchive済み）。
