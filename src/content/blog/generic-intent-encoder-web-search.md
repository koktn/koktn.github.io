---
title: クリックから検索意図を学ぶGEN Encoder――Bingの7億クエリANN活用
description: SIGIR 2019のGEN Encoder論文をもとに、co-click弱教師学習、multi-task fine-tuning、検索意図評価、long-tail queryへのANN活用を解説します。
publishedAt: 2026-09-10
category: AI
tags:
  - 検索
  - Information Retrieval
  - Embedding
  - Weak Supervision
  - 論文解説
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。人間による内容確認はまだ実施していません。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Microsoft AI & ResearchのHongfei Zhangらによる論文「[Generic Intent Representation in Web Search](https://arxiv.org/abs/1907.10710)」です。SIGIR 2019で発表された10ページのconference paperで、arXivには2019年7月24日にv1が公開されています（[PDF](https://arxiv.org/pdf/1907.10710)、[Microsoft Researchの公開ページ](https://www.microsoft.com/en-us/research/publication/generic-intent-representation-in-web-search/)）。

この論文の価値を一文でまとめると、**文章として似ているqueryを学ぶのではなく、同じURLをclickしたqueryを近づけることで検索意図に特化した表現を学び、7億queryの近傍検索によって未観測queryの不足を補った**点にあります。

ただし、2019年のBing検索logを使った研究です。学習dataと評価dataは非公開で、現在の検索環境や別serviceへ報告値をそのまま移せるわけではありません。

## 意味が近いことと、検索意図が同じことは違う

一般的な言語embeddingは、周辺文脈が似た単語や文を近くへ配置します。これは多くの自然言語処理taskで有効ですが、検索では「文章として似ている」と「同じ結果を求めている」が一致しないことがあります。

論文が挙げる例では、`Harvard student housing`は単語の意味だけなら`Cornell student housing`に近く見えます。しかし、利用者が必要とする検索結果を考えると、`Cambridge dorm MA`の方が同じ意図を表し得ます。逆に、`horse racing`と`racing horses`はほぼ同じ単語を含みながら、語順によって意味が変わります。

そこでGEN Encoder（GEneric iNtent Encoder）は、検索意図のclassやontologyを人手で先に定義しません。**同じURLへのclickを生んだquery、すなわちco-click queryは、同じ情報需要を持つ可能性が高い**という仮定を弱教師信号にします。

## GEN Encoderの構造

GEN Encoderは、1つのqueryを100次元のvectorへ変換します。構造は次の3要素です。

| 要素 | 役割 | 論文の設定 |
| --- | --- | --- |
| Word embedding | 単語単位の検索意図を表す | 語彙100万、200次元、highway network付き |
| Character-aware embedding | typoやrare wordを文字n-gramから補う | 文字語彙1,000、200次元、CNNと2層highway network |
| Mix Encoder | 語順に敏感な表現と、語順に依存しない表現を統合する | 1層Bi-GRUと単語vector平均を連結し、residual layerで100次元へ変換 |

Character-aware embeddingがあるため、たとえば低頻度のmisspellingである`retreval`も、`retrieval`と共有する文字列から表現できます。

Mix Encoderは、Bi-GRUで`horse racing`と`racing horse`のような語順差を扱う一方、単語vectorの平均も残します。後者は`Cambridge MA`と`MA Cambridge`のように語順が本質ではないqueryに向いています。どちらか一方へ固定せず、検索queryの2つの性質を結合した設計です。

```text
query
  ├─ word embedding ─ highway ─┐
  └─ character CNN ─ highway ──┴─ term embedding
                                      ├─ Bi-GRU
                                      └─ average pooling
                                             ↓
                                      residual layer
                                             ↓
                                  100次元のGEN encoding
```

## 2段階学習の中心はco-click弱教師学習

### Phase 1：2億groupのco-clickから学ぶ

第1段階では、6か月分のBing検索logから約2億のco-click groupを作ります。同じURLをclickしたqueryをpositive pairとし、cosine similarityが高くなるようend-to-endで学習します。

ただし、同じURLが常に同じ意図を表すとは限りません。多様な目的で開かれるURLを減らすため、論文では5種類を超えるqueryからclickされたURLを除外しています。それでもclickにはposition bias、既存rankingの影響、誤clickなどが残るため、著者ら自身も弱教師信号にはnoiseと既存system由来のbiasがあると述べています。

negative pairをrandomに選ぶだけでは、明らかに無関係なqueryが多くなります。そこで各mini-batch内から、現在のencoderが最も似ていると判断したnegative queryを選ぶnoise-contrastive estimation（NCE）を使います。名前はNCEですが、実際の役割は学習中の**hard negative mining**です。

### Phase 2：3taskを混ぜてfine-tuningする

第2段階では、次のdataをmini-batch単位でrandomに混ぜ、3つのlossを足してfine-tuningします。

| Dataset | Training | Validation | Testing | Positive / Negative |
| --- | ---: | ---: | ---: | ---: |
| Co-click query | 約40万group | 約1万group | なし | 50% / 50% |
| Query paraphrase | 約80万pair | 約1万pair | 約1万pair | 25% / 75% |
| Question paraphrase | 約35万pair | 約1万pair | 約1万pair | 30% / 70% |

co-clickの量を活かしつつ、高品質だが高価な人手labelで「同じclick先になりやすい」以外の一般性を補う構成です。optimizerはAdam、learning rateは`1e-4`、batch sizeは256です。論文のいうtypical GPUでは、第1段階は1 epochあたり約300時間で1 epoch後に収束し、第2段階は1 epoch約1時間で数epochを要しました。GPU型番は示されていないため、現在のhardwareとの直接比較はできません。

## 検証は検索意図の近さだけを測る

評価では、queryをembeddingへ変換し、query pair間の距離と人手の検索意図labelが合うかを調べます。評価labelは学習にもvalidationにも使われません。

| Dataset | 対象 | 規模 | Metric |
| --- | --- | ---: | --- |
| General | session内で共起しやすい候補を、同一意図・一部共通・別意図の3段階で評価 | 3,773 target query | NDCG（cut-offなし） |
| Easy | spellingや単語単位の単純な変化 | 2,864 target query | NDCG（cut-offなし） |
| Hard | 既存のquery表現が失敗したadversarial case | 385 target query、426 pair | AUC |

統計的有意差はFisherのrandomization testで検定し、`p < 0.05`を基準にしています。主要結果から、代表的なbaselineだけを抜き出すと次の通りです。

| Method | General NDCG | Easy NDCG | Hard AUC |
| --- | ---: | ---: | ---: |
| TF-IDF BOW | 0.4969 | 0.8047 | 0.4740 |
| RLM+（clicked title） | 0.4985 | 0.8570 | 0.5036 |
| BERT Encoder | 0.4643 | 0.8585 | 0.4977 |
| Universal Sentence Encoder | 0.4958 | 0.8635 | 0.5675 |
| **GEN Encoder** | **0.5244** | **0.8688** | **0.6667** |

GEN EncoderのTF-IDF比の相対改善は、Generalで5.53%、Easyで7.97%、Hardで40.64%です。とくに難例で差が広がっています。

一方、この表を「GEN Encoderという古いRNNがBERTより優れている」と読むのは不正確です。比較されたBERTは、公開済みBERT baseの最終層を平均しただけで、この検索logやparaphrase labelによるfine-tuningをしていません。USEも公開版をそのまま使用しています。したがって、この実験が強く示すのはarchitectureの世代差ではなく、**検索意図に合った教師信号を使う効果**です。

また、General・Easy・Hardはいずれもembeddingのintrinsic evaluationです。最終的な検索rankingのNDCGや利用者満足度を直接測った結果ではありません。

## Ablationが示す「dataがarchitectureより先」

論文で最も実務的なのはablationです。

- co-clickを使わず、2種類のparaphrase taskだけで学習すると、Generalは`0.5101`、Hardは`0.5220`に留まる
- co-clickを100%使い、第2段階を省くと、Generalは`0.5278`、Easyは`0.8734`まで上がる一方、Hardは`0.6059`に留まる
- full modelはGeneral `0.5244`、Easy `0.8688`、Hard `0.6667`で、multi-task化によりHard AUCがco-click単独から約10%相対改善する
- full modelと同じ2段階dataで単語embeddingの平均だけを学習したAvg-Embでも、General `0.5081`、Hard `0.5670`へ到達する

つまり、大規模co-clickは主要dataで高い性能を作り、人手paraphrase labelは難例へのgeneralizationを補っています。複雑なencoderを先に選ぶより、目的に合う行動signalをどう作るかが先です。

ただし、co-click dataを10%の約2,000万groupから100%の約2億groupへ増やしても精度はほぼplateauしています。論文は、encoder容量がdata規模を十分に消化できていない可能性を挙げています。「logが多いほど同じ比率で精度が上がる」という結果ではありません。

## 7億queryのANNでlong tailを補う

GEN Encoderのdownstream利用として、論文は6か月間からsampleした7億queryのencodingをHNSW indexへ格納しました。index構築後のある1日から100万queryをrandom sampleし、各queryについてcosine distanceのradiusを`0.15`、`0.10`、`0.05`と変えて上位10近傍を検索しています。navigational queryとadult queryは除外されています。

論文のtypical parallel computing environmentでは、lookupは1 queryあたり最大10msでした。radiusを狭めるほど同一意図率は上がりますが、近傍が見つかるcoverageは下がります。

| Query頻度 | Radius | Coverage | 平均近傍数 | 同一意図率 |
| --- | ---: | ---: | ---: | ---: |
| Head | 0.15 | 96.1% | 5.05 | 92% |
| Torso | 0.15 | 88.3% | 3.77 | 80% |
| Tail | 0.15 | 57.9% | 2.91 | 47% |
| Tail | 0.05 | 15.9% | 2.32 | 80% |

同一意図率は、100 queryの取得近傍を3人のexpertが判定した結果です。平均Cohen's kappaは0.717でした。tail queryでradius `0.15`を使うと、広く拾える代わりに同一意図は47%です。平均すると、coverage対象のtail queryごとに約`1.37`件（`2.91 × 0.47`）の同一意図queryが得られます。

近傍queryの過去signalを合算すると、直前6か月に一度も現れなかったqueryの割合は約39%から半分程度へ減りました。ただし誤った近傍を精度で補正すると、相対減少は約35%です。「未観測queryを完全に解消した」のではなく、coverageと誤結合のtrade-offを取りながら観測を借りています。

著者らは、このANN検索がBingの複数componentへ導入され、その一例では回答不能な質問を同一意図の回答可能な質問へ対応付けることで、online QA pipelineの「重要な一部分」のcoverageを回答品質を落とさず2倍にしたと報告しています。ただし、元のcoverage、対象traffic比率、品質metric、実験期間は示されていません。この記述だけからservice全体の効果量は判断できません。

## Session内の情報探索行動も距離へ現れた

GEN Encoderは、session内で隣接するquery pairを、Topic Change、Explore、Specify、Paraphraseという4段階に近い順序で分離しました。100 pairを3人がlabelした評価で、expert間のCohen's kappaは0.64です。

人手labelとのSpearman順位相関は、全100 pairでGEN Encoderが`0.800`、TF-IDFが`0.642`、BERT Encoderが`0.626`でした。人間同士の平均は`0.859`です。これは興味深いemergent behaviorですが、sampleは100 pairに限られます。論文もsession理解への応用を将来可能性として扱っており、本番効果を実証した結果ではありません。

## 現在のsystemで試すなら

論文は学習dataを公開していません。また、本文はGEN Encoderを公開したとして短縮URLを示していますが、本稿作成時には再現に使えるcanonicalなcode repository、学習済みmodel、完全な前処理を確認できませんでした。したがって、以下は原論文の完全再現ではなく、考え方を別の検索systemへ適用するための記事側の提案です。

```text
同意・保持期間を定めてquery / impression / clickを記録
  ↓
position・表示条件を含めてco-click候補を作成
  ↓
random negativeとin-batch hard negativeでencoderを学習
  ↓
人手の同一意図pairでthresholdとhard caseを評価
  ↓
ANN indexをshadow構築し、頻度帯ごとにcoverageと誤結合率を測定
  ↓
既存rankingへfeatureとして限定導入
  ↓
品質悪化時はANN signalを無効化してlexical baselineへrollback
```

最初から7億queryを扱う必要はありません。まず頻出・中頻度・tailを分けた評価setを作り、次を確認します。

1. TF-IDFやBM25、現在利用中のsentence embeddingに対して、同一意図判定が改善するか
2. head queryとtail queryで、同じdistance thresholdが妥当か
3. click position、device、地域、時刻、新旧ranking modelごとに誤差が偏らないか
4. ANNから借りたsignalを加えたとき、offline ranking metricだけでなくonline guardrailが悪化しないか
5. encoder・index・thresholdを独立にversion管理し、即時に切り戻せるか

現在ならencoder自体をTransformer系へ置き換える選択肢もあります。しかし検証すべき仮説は「新しいarchitectureなら勝つ」ではなく、「同じarchitectureでも一般文の類似度学習より、検索行動にgroundingした学習の方が目的に合うか」です。encoder、教師data、negative samplingを一度に変えず、ablationで寄与を分ける必要があります。

## 導入前に押さえる限界とrisk

- 同じURLへのclickは同一意図の完全なground truthではなく、position bias、presentation bias、既存rankingの偏りを含む
- privacyや同意、保存期間、削除要求への対応は論文で具体化されておらず、query logを扱う組織側で設計が必要
- intrinsic evaluationでの改善を、検索rankingや事業metricの改善へ直接読み替えられない
- BERTとUSEは検索dataでfine-tuningされておらず、architecture同士の公平な上限比較ではない
- ANNの同一意図率は100 query、session分析は100 pairの人手評価であり、細かなsliceの信頼性は分からない
- 7億queryのindex容量、構築時間、更新cost、hardware構成は報告されていない
- 100次元encodingの推論約15msとANN lookup最大10msは「typical」な環境での値で、現在のservice latencyを保証しない
- 公開情報だけではdata、model、index、online QA評価を第三者が完全再現できない

## まとめ

GEN Encoderから得られる最も重要な教訓は、**embeddingの品質はnetwork構造だけでなく、何を「近い」と教えたかで決まる**ことです。

大規模co-click dataは検索意図に合った表現の土台を作り、queryとquestionのparaphrase labelは難例へのgeneralizationを補いました。さらに、その表現をHNSWと組み合わせることで、未観測queryにも過去の行動signalを引き継げる可能性を示しています。

一方、clickは利用者の意図そのものではなく、既存systemを通して観測されたproxyです。ANNもradiusを広げればcoverageとともに誤結合が増えます。実装では「近傍が見つかった割合」だけを成功指標にせず、同一意図率、ranking品質、slice別の偏り、privacy、rollbackを含むsystemとして評価する必要があります。

## 参照

- Hongfei Zhang et al., [Generic Intent Representation in Web Search](https://arxiv.org/abs/1907.10710), SIGIR 2019, pp. 65–74, arXiv:1907.10710v1（[PDF](https://arxiv.org/pdf/1907.10710)、[DOI](https://doi.org/10.1145/3331184.3331198)）。
- Microsoft Research, [Generic Intent Representation in Web Search](https://www.microsoft.com/en-us/research/publication/generic-intent-representation-in-web-search/)。

本記事の手法、設定、数値、実験条件は原論文に基づき、「現在のsystemで試すなら」は公開情報から導いた記事側の提案です。
