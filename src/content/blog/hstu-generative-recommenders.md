---
title: HSTUとGenerative Recommenders――行動履歴から1.5兆parameterの推薦モデルへ
description: MetaのHSTU論文を、推薦のsequential transduction化、pointwise attention、Stochastic Length、M-FALCON、公開実験とproduction評価の限界から解説します。
publishedAt: 2026-09-29
category: AI
tags:
  - Recommendation System
  - Generative Recommendation
  - HSTU
  - Transformer
  - ICML
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値と手法は原論文と公開実装を確認して記載していますが、利用時は原文も確認してください。

大規模な推薦システムは、ユーザーID、商品ID、属性、集計値などを人手で設計したfeatureへ変換し、複数のnetworkで組み合わせるDLRM（Deep Learning Recommendation Model）を長く使ってきました。MetaのJiaqi Zhaiらによる「[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152)」は、この構成をユーザーの行動列を処理するGenerative Recommender（GR）へ置き換え、そのためのencoder **HSTU**（Hierarchical Sequential Transduction Unit）を提案したICML 2024論文です。対象はarXiv v3（2024年5月6日改訂）です。

この論文の価値は、**推薦のrankingとretrievalをsequential transductionとして統一し、学習と推論の重複計算を減らす仕組みまで含めて、1.5兆parameterのmodelをproductionへ展開した**点にあります。一方、「生成推薦」という名前からLLMが説明文や商品IDのtoken列を生成する手法を想像すると、論文の中心を取り違えます。

## DLRMから行動sequenceへ何を変えたのか

従来の産業向けDLRMは、数値featureとカテゴリfeatureを別々に処理し、embedding、feature interaction、multi-task headなどを組み合わせます。新しいfeatureを加えるたびに、集計方法、保存、学習時の結合、serving時の整合性を管理する必要があります。

GRは、この異種featureを時間順のtoken列へまとめます。

```text
従来のDLRM
  user/item ID + 集計値 + 属性 + cross feature
    → 個別のembedding・network
    → feature interaction
    → ranking / retrieval

Generative Recommender
  content Φ0, action a0, content Φ1, action a1, ...
    + 時間変化の遅いカテゴリfeature
    → 統合された時系列
    → HSTU
    → ranking / retrieval
```

主系列には、ユーザーが接触したcontentと、その後のlike、skip、視聴完了、shareなどのactionを置きます。言語や居住地、follow中のcreatorのように変化が遅いカテゴリfeatureは、同じ値が連続する区間を圧縮して主系列へmergeします。

CTRや回数のような数値featureはinteractionごとに変わるため、すべてをtoken化するとstorageと計算量が膨らみます。論文は、集計元となるカテゴリ情報が既にsequenceへ入っているなら、十分に表現力のあるtarget-aware modelが必要な集計を学べるとして、数値featureをmodel側の表現へ置き換えます。有限の履歴とmodel容量で完全に同じ情報を復元できる保証ではなく、production実験で有効性を確かめた設計仮説です。

## Rankingとretrievalを同じsequential transductionで表す

入力token列を `x0, x1, ...`、各位置に対応する出力を `y0, y1, ...` とすると、論文は2つのtaskを次のように表します。

| Task | 入力 | 予測するもの |
| --- | --- | --- |
| ranking | contentとactionを交互に並べた列 | 各contentに対する次のaction |
| retrieval | contentとactionを対にした履歴 | positive actionにつながる次のcontent |

rankingでは、候補contentを履歴の末尾へ置き、その候補に対するaction確率を予測します。これにより、候補と過去の履歴を早い段階で相互作用させるtarget-awareな処理になります。実運用では、content位置の出力を小さなnetworkへ通し、複数taskの予測へ変換します。

retrievalでは、履歴から次にpositive actionを得るcontentの分布を学びます。ただし、TIGERのように短いSemantic IDを自己回帰生成する方式とは異なり、この論文のfeature空間には大規模なatomic IDが残ります。著者らはretrieval時にMIPS、beam search、階層的retrievalなどを利用できるとしています。

ここでいう「generative」のもう1つの意味は**学習例の作り方**です。従来のimpression単位の学習では、同じユーザー履歴をtargetごとに何度もencoderへ通します。GRは1本のsequenceから複数位置のlossをまとめて計算し、encoderの仕事を複数targetで共有します。論文の解析では、ユーザーsequence長に反比例する割合でsampleすることで、全体の学習計算量を長さ `N` に対して1段階減らせます。[論文Section 2](https://arxiv.org/pdf/2402.17152#page=2)

## HSTUはTransformerと何が違うのか

HSTUは、同じlayerをresidual connectionで積み重ねるcausalなsequence encoderです。各layerは大きく3段に分かれます。

1. 入力から `U`、`V`、`Q`、`K` を一度にprojectする
2. `QKᵀ`、位置・時間のrelative bias、`V`から履歴をaggregateする
3. aggregate結果をnormalizeし、`U`との要素積でgateして出力へprojectする

通常のTransformerと比べた中核的な違いは、attention weightをsequence全体のsoftmaxで正規化せず、SiLUを使った**pointwise aggregated attention**にしたことです。

softmaxは各行のweight総和を1にするため、関連する行動が増えても「どの履歴を相対的に重く見るか」は表現できる一方、関連行動が何件あったかという強度を薄める可能性があります。推薦では、特定topicを1回見た人と100回見た人の違いが重要です。HSTUはpointwiseにweightを作り、集約後のlayer normalizationで学習を安定させます。位置と経過時間をrelative attention biasへ含めるため、順序だけでなく時間間隔も扱えます。

非定常なvocabularyを模したsynthetic streaming dataでは、HR@10が標準Transformerの `0.0442`、softmax版HSTUの `0.0617`、pointwise版HSTUの `0.0893`でした。pointwise版はsoftmax版より**44.7%の相対改善**ですが、これは合成data上のablationであり、production全体における単独効果ではありません。[論文Table 2](https://arxiv.org/pdf/2402.17152#page=5)

## 長い履歴を現実の計算量へ収める4つの工夫

HSTU layerだけでは、最大8,192 tokenの履歴、巨大なID vocabulary、数千から数万のranking候補を処理できません。論文のsystemは複数の最適化を組み合わせています。

### 1. 長さの異なるsequenceをpaddingせず処理する

ユーザーごとの履歴長は大きく偏ります。HSTUのGPU kernelはraggedなsequenceをgrouped GEMMとして処理し、padding部分の計算を避けます。論文では、この実装だけで2〜5倍のthroughput向上を報告しています。

### 2. Stochastic Lengthで学習時の履歴を間引く

ユーザー行動には、短期と長期の両方で似た行動が繰り返されるという仮定があります。Stochastic Length（SL）は、長い履歴を一定確率でsubsequenceへ縮め、attention計算のsparsityを増やします。30日分の履歴を使うproduction相当の設定では、最大長4,096、`α = 1.6`のとき80.5%のsparsityになり、実際のsequenceは多くの場合776 tokenまで縮みました。適切な `α` では主要taskのNormalized Entropy（NE）の悪化が `0.002` 未満だったと報告されています。

これはinference時に常に80%を捨てるという話ではなく、主に高コストなtrainingを安くするsamplingです。Appendixの比較では、同程度のsparsityを作るzero-shot／fine-tuningによる長さ外挿よりSLのNE悪化が小さかったものの、非公開dataに基づく結果です。[論文Section 3.2とAppendix F](https://arxiv.org/pdf/2402.17152#page=5)

### 3. Activationとembedding optimizerのmemoryを減らす

推薦modelは大きなbatchを必要とするため、parameterだけでなくactivation memoryがbottleneckになります。論文の見積もりでは、HSTUはlinear layerの削減とoperator fusionにより1 layerあたりのactivation stateをTransformerの `33d` から `14d` へ減らし、2倍を超える深さを同じmemoryで扱えます。

一方、1.5兆parameterという規模を理解するには巨大なID embeddingを無視できません。論文は、100億語彙、512次元、fp32のAdamでembeddingとoptimizer stateだけで60TBになる例を示します。row-wise AdamWとoptimizer stateのDRAM配置により、HBM使用量を1要素あたり12 byteから2 byteへ減らしています。「1.5兆parameter」は、1.5兆個のdense Transformer weightを持つLLMと同じ構成ではありません。

### 4. M-FALCONで候補間の履歴計算を共有する

rankingでは、同じユーザー履歴に対して最大数万件の候補をscoreします。候補ごとに履歴全体を再計算すると、target-aware attentionの利点がそのままcostになります。

M-FALCON（Microbatched-Fast Attention Leveraging Cacheable OperatioNs）は、候補をmicrobatchへまとめ、attention maskとrelative biasを調整して履歴側の計算を共有します。さらにencoderのKV cacheをmicrobatch間で再利用します。候補数を `m`、microbatch内の候補数を `bₘ`、履歴長を `n` とすると、候補ごとのcross-attentionに相当する計算を `O(bₘn²d)` から `O((n + bₘ)²d)` へ変えます。`bₘ` が `n` より十分小さければ、支配項はほぼ `O(n²d)` です。[論文Section 3.4](https://arxiv.org/pdf/2402.17152#page=6)

## Public datasetではどこまで改善したか

公開評価はMovieLens-1M、MovieLens-20M、Amazon Reviews Booksで行われています。baselineは、sampled softmaxを使う強いSASRec recipeです。HSTUはSASRecとlayer数・head数などを揃え、HSTU-largeはlayer数を4倍、head数を2倍にしています。次の値はarXiv v3のTable 4です。

| Dataset | Model | HR@10 | NDCG@10 |
| --- | --- | ---: | ---: |
| ML-1M | SASRec | 0.2853 | 0.1603 |
| ML-1M | HSTU | 0.3097（+8.6%） | 0.1720（+7.3%） |
| ML-1M | HSTU-large | 0.3294（+15.5%） | 0.1893（+18.1%） |
| ML-20M | SASRec | 0.2906 | 0.1621 |
| ML-20M | HSTU | 0.3252（+11.9%） | 0.1878（+15.9%） |
| ML-20M | HSTU-large | 0.3567（+22.8%） | 0.2106（+30.0%） |
| Books | SASRec | 0.0292 | 0.0156 |
| Books | HSTU | 0.0404（+38.4%） | 0.0219（+40.6%） |
| Books | HSTU-large | 0.0469（+60.6%） | 0.0257（+65.8%） |

括弧内はSASRecに対する相対改善です。HSTUと同じ構成の比較でも全datasetで改善し、modelを大きくするとさらに伸びています。ただし、この評価はfull-shuffle・multi-epoch学習です。1回だけ時系列順に流すproductionのstreaming学習とは条件が異なり、この65.8%をそのままオンライン効果として読んではいけません。[論文Table 4](https://arxiv.org/pdf/2402.17152#page=7)

## Production評価の12.4%をどう読むか

産業規模のencoder比較では、1000億件のDLRM相当exampleを1 passで学習し、1 jobあたり64〜256基のNVIDIA H100を使っています。rankingはmain engagement task（E-Task）とmain consumption task（C-Task）のNE、retrievalはlog perplexityで比較しています。

end-to-end比較では、retrievalのGRを新しい候補sourceとして加えると、匿名化されたonline指標がE-Taskで `+6.2%`、C-Taskで `+5.0%`でした。主要なDLRM sourceをGRで置き換えた場合は、それぞれ `+5.1%`、`+1.9%`です。rankingではGRが `+12.4%`、`+4.4%`を記録しました。

論文がabstractで掲げる「online A/B testで12.4%改善」は、このうちrankingのE-Taskにおける最大値です。すべてのsurfaceや指標が12.4%改善したわけではありません。E-TaskとC-Taskの具体的定義、traffic量、実験期間、信頼区間は公開されていないため、別のserviceへ移したときの効果量は推定できません。

効率面では、8,192 token、`d = 512`、8 head、H100、bfloat16のencoder比較で、FlashAttention 2を使うTransformerに対してtraining最大15.2倍、inference最大5.6倍でした。end-to-endのproduction rankingでは、FLOPsが285倍のGRが、1,024候補で1.50倍、16,384候補で2.99倍のQPSを達成しています。この結果はHSTU単体ではなく、ragged kernel、M-FALCON、cachingを含むsystem全体の値です。[論文Section 4.2–4.3](https://arxiv.org/pdf/2402.17152#page=7)

## Recommendationにもscaling lawは現れたのか

著者らは、HSTUのlayer数、embedding次元、head数、sequence長、retrievalのnegative数などを変え、学習computeを約3桁の範囲で増やしました。DLRMは約2,000億parameter付近で性能が飽和した一方、GRは1.5兆parameterまで改善が続き、retrievalのHR@100／HR@500とrankingのNEがcomputeに対してpower lawに従ったと報告しています。

最大構成は、sequence長8,192、embedding次元1,024、HSTU 24 layerです。streaming学習なので、computeは365日分へ正規化してGPT-3やLLaMA 2の学習規模と比較されています。著者らは、言語modelと違ってsequence長を他のparameterと一緒に伸ばすことが特に重要だと述べています。

これは「推薦model一般の普遍的なscaling law」が確立したという意味ではありません。観測は1社の非公開data、非公開task、限られたcompute範囲に基づきます。DLRM baselineの正確なproduction設定も機密で、論文では高水準の構成だけが説明されています。再現可能なpublic datasetの表と、production scalingの主張はevidenceの強さを分けて読む必要があります。[論文Figure 7とAppendix E](https://arxiv.org/pdf/2402.17152#page=8)

## 公開実装で試せる範囲

[公式repository](https://github.com/meta-recsys/generative-recommenders)はApache-2.0で、MovieLensとAmazon Reviewsのpublic実験、HSTUのTriton／CUDA kernel、training・inference用のDLRM-v3などを公開しています。READMEの確認環境はUbuntu 22.04、CUDA 12.4、Python 3.10で、public datasetの多くは24GB以上のGPU memoryが目安です。

```sh
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders
pip3 install -r requirements.txt
mkdir -p tmp
python3 preprocess_public_data.py
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345
```

これはML-1Mなどの公開条件を試す手順であり、1.5兆parameterのproduction modelを再現するものではありません。production data、feature定義、A/B test設定、学習cluster、全serving stackは公開されていません。また、repositoryのREADMEが示す一部の再現値はarXiv v3のTable 4とわずかに異なります。比較するときはpaperの数字と現在のcodeを混ぜず、commit、config、data preprocessingを固定する必要があります。

## 導入するなら何を分けて検証するか

この論文を実務へ適用するときは、「HSTUへ置き換える」という1つの変更にまとめず、次の順で効果を切り分けると判断しやすくなります。

1. **評価を固定する**：時系列split、strongなsequential baseline、HR／NDCGに加え、latency、memory、QPSを測る。
2. **encoderだけ比較する**：同じfeature、loss、negative sampling、model規模でSASRec／TransformerとHSTUを比べる。
3. **featureのsequence化を試す**：人手集計featureを一度に消さず、raw actionだけのGRとの差をablationで測る。
4. **学習最適化を分ける**：ragged kernelとStochastic Lengthを別々に導入し、長い履歴での品質劣化を監視する。
5. **servingを検証する**：候補数ごとにM-FALCONのmicrobatch、KV cache、tail latency、HBM／DRAM転送を測る。
6. **小さいonline testから始める**：offline指標だけでなく、主要指標、guardrail、長期的な満足度を確認する。

atomic IDを使うため、新規item、rare item、embedding tableの更新、削除要求への対応も必要です。sequenceへfeatureを統一すれば自動的にprivacyが改善するわけではありません。どの行動を保存するか、保存期間、access control、user consentを別途設計する必要があります。

## Limitation

この研究を読むうえで重要な制約は次の通りです。

- public実験は3 datasetのoffline評価で、productionと異なるmulti-pass・full-shuffle条件である
- productionのtask、data、baseline詳細、A/B test期間、sample数、統計的不確実性が非公開である
- 12.4%は匿名化されたranking E-Taskの最大改善で、売上やCTRなど特定のbusiness metricではない
- 総parameter数には巨大なID embeddingも含まれ、同規模LLMとのparameter数だけの比較は誤解を招く
- HSTU encoderの高速化、M-FALCON、production infrastructureの効果がend-to-end結果では一体になっている
- 新しいcontentや急変する嗜好にatomic ID表現がどうgeneralizeするかは、公開結果だけでは十分に判断できない

論文のImpact Statementは、手作りfeatureの削減がprivacyや長期的なuser valueの改善につながる可能性を述べています。しかし、privacy指標や長期outcomeを直接評価した結果は示していません。ここは実証済みの効果ではなく、将来の方向性です。

## まとめ

Generative Recommenderの本質は、既存の推薦systemへLLMを足すことではありません。異種feature、ranking、retrieval、学習例の作り方を**ユーザー行動のsequential transduction**として再設計し、computeを増やすと品質が伸びる土台を作ることです。

HSTUのpointwise attentionは行動の「相対的な重要度」だけでなく「蓄積量」を残し、Stochastic Lengthは長い履歴のtraining costを抑え、M-FALCONは候補間で同じ履歴計算を共有します。public datasetでの改善は再現可能な入口ですが、1.5兆parameter、12.4%のonline改善、285倍のFLOPsを扱うproduction結果は非公開条件への依存が大きく、system全体のcase studyとして読むのが適切です。

## 参照

- Jiaqi Zhai et al., [Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152), ICML 2024, arXiv v3, 2024-05-06.
- Jiaqi Zhai et al., [PDF全文](https://arxiv.org/pdf/2402.17152), 26 pages.
- Meta RecSys, [generative-recommenders](https://github.com/meta-recsys/generative-recommenders), official implementation, Apache-2.0.
