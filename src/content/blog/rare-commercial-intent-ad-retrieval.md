---
title: Commercial Intentionで広告検索を短くする：Tencentの生成検索RAREを読む
description: EMNLP 2025のRARE論文をもとに、LLMが短い商用意図を生成し、転置インデックスから広告を引く仕組み、60ms以内の推論設計、オフライン評価と本番A/Bテストの読み方を解説します。
publishedAt: 2026-10-04
category: AI
tags:
  - LLM
  - Generative Retrieval
  - Advertising
  - Information Retrieval
  - EMNLP
draft: false
---

> AI利用の明示<br>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。主張と数値は参照元を確認していますが、利用時は原文も確認してください。

Tencentの研究チームがEMNLP 2025で発表した「[Real-time Ad Retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising](https://aclanthology.org/2025.emnlp-main.1473/)」は、LLMによる生成検索を数千万件規模の広告検索へ組み込み、本番の応答時間に収めた事例です。

この論文の価値は、LLMに広告IDを直接覚えさせるのではなく、Commercial Intention（CI）という短い意味表現を生成させ、既存の転置インデックスへ接続した点にあります。モデルが得意な意味理解と、検索システムが得意な大量候補の参照を分担する設計です。

## 広告IDを直接生成すると規模と更新に弱い

検索広告では、利用者のqueryから数百万〜数千万件の広告を絞り、後段のrankerへ渡します。従来のquery→keyword→adという2段構成は、広告主が購入したkeywordと利用者の表現がずれると候補を取りこぼします。queryから広告を直接検索するdual encoderについても、深い商用意図の理解に課題があるというのが論文の見方です。

生成検索では、文書や広告にDocIDを割り当て、LLMがqueryからIDを生成する方法が研究されてきました。しかし数値IDは意味を持たず、一つのIDが少数の候補にしか対応しないため、大規模な候補集合では多くのIDを生成しなければなりません。広告の追加や削除のたびにモデルを再学習したり、FM-indexを更新したりする方法も、本番運用では扱いにくくなります。

RAREは、このDocIDを人が読める短いCIへ置き換えます。たとえば花の広告なら「花をオンライン購入」「近くの花屋」「母の日」といった表現です。一つのCIに多数の広告を結び付ければ、LLMが少数の意味表現を生成するだけで広い候補へ到達できます。

## RAREはLLMと転置インデックスをCIでつなぐ

RAREには、広告側のindexingとquery側のretrievalという二つの流れがあります。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/rare-architecture-mobile.svg">
    <img src="/img/posts/rare-architecture.svg" alt="広告からCI転置インデックスを構築し、queryから生成したCIで広告候補を検索するRAREの処理構成" loading="lazy">
  </picture>
  <figcaption>図1：<a href="https://aclanthology.org/2025.emnlp-main.1473.pdf">原論文Figure 1〜3・§3</a>の事実を基に、オフライン処理とオンライン処理の境界が分かるよう本記事で再構成した独自図です。原図の転載ではありません。原論文は<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>です。</figcaption>
</figure>

### 広告側では約200万種類のCIへ集約する

最初に13BパラメータのLLMが、広告タイトル、landing page、配信素材から複数のCIを生成します。業界単位のclusteringで重複を減らし、分野の専門家による評価も加えます。こうしてCI集合を約200万種類まで絞る流れです。

続いて制約付きdecodingを使い、各広告を平均約30個のCIへ割り当てます。最後にCIをkey、広告をposting listとする転置インデックスを構築します。新しい広告は既存のCI集合へ割り当てればよいため、追加のたびにLLMを再学習する必要はありません。CI→広告のindexは1時間ごと、CI集合全体は1か月ごとに更新すると論文に記載されています。

### query側ではcache missだけ1Bモデルを呼ぶ

queryが届くと、RAREは最初にcacheを調べます。head queryでは、13Bモデルがオフラインで生成したCIをそのまま使用します。cacheにないqueryだけが1Bモデルによるオンライン生成の対象です。そのCIを転置インデックスで引き、得られた広告候補を後段のrankerへ渡します。

論文が報告するquery分布では、約5%のqueryが全リクエストの約60%を占めます。実運用のcache hit率は約65%です。数百万件のhead queryを事前計算することで、オンライン計算資源を70%削減したと報告しています。

## LLMの調整は知識と出力形式を分ける

RAREのLLM調整は2段階です。

Knowledge Injectionでは、queryの意図抽出、広告の意図抽出、広告タイトル生成、query拡張という4タスクを使います。各タスクは2,000件です。公開LLMが生成した推論過程付きの合成データをbase modelへ学習させます。件数を絞った理由として、著者らは大量のfine-tuningによって一般知識や推論能力を失う可能性を挙げています。

Format Fine-Tuningでは、実際のオンラインデータから作ったquery→CIと広告→CIを各2,000件使います。ここでは途中の説明を出さず、CIだけを多様に列挙する形式を覚えさせます。論文の構成では、前段が広告分野の知識と推論方法、後段が短い出力形式を担当します。

ただし、学習データそのもの、data split、学習率、epoch数は公開されていません。Appendixにpromptと件数はありますが、同じモデルを完全に再現できる情報は揃っていません。

## 制約付きbeam searchで存在するCIだけを生成する

自由生成した文字列がCI集合に存在しなければ、転置インデックスを検索できません。RAREは約200万種類のCIからprefix trieを作り、現在のprefixから有効なtokenだけを次の候補にします。確率がしきい値を下回るtokenを落とし、各段でbeam size以内の候補を残します。

```text
allowed = trie.children(current_prefix)
next_tokens = model.next_token_probabilities(query, current_prefix)
candidates = next_tokens
  .filter(token => token in allowed)
  .filter(token => probability(token) >= threshold)
beams = top_k(expand(beams, candidates), beam_size)
```

実装はCUDAでLLM推論に統合され、複数CIを並列生成します。論文で採用した設定は、オフライン13Bモデルがbeam size 256、temperature 0.8、最大出力長6です。オンライン1Bモデルではbeam size 50、temperature 0.7、最大出力長4を使います。CIは平均3 tokenで、オンライン推論は60ms以内です。

本番基盤では数百GPUを使い、学習済みモデルをFP8へ量子化しています。GPU 1基あたり約30 QPS、peak GPU利用率は最大90%との報告です。したがって「1Bモデルなら60msで動く」という一般的なbenchmarkではありません。専用CUDA実装、短い出力、cache、FP8、数百GPUを組み合わせたシステム全体の結果です。

## オフライン評価では広い候補集合ほど差が開く

オフライン評価には、実サービスのhead queryとclick広告を1日分集め、cleaningした5,000 query、150,000広告を使います。各queryには最大1,000件の広告候補が対応します。比較対象はBM25、BERT系、T5、Qwen、Hunyuan、DSIなど10手法です。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/rare-offline-results-mobile.svg">
    <img src="/img/posts/rare-offline-results.svg" alt="RAREと主要baselineのHR@500、MAP、広告coverageを比較したchart" loading="lazy">
  </picture>
  <figcaption>図2：<a href="https://aclanthology.org/2025.emnlp-main.1473.pdf">原論文Table 1</a>の数値から主要な比較対象を本記事でchart化しました。HR@500とMAPは0〜1、ACRは百分率で尺度が異なるため、panelを分けています。誤差や信頼区間は原論文に掲載されていません。</figcaption>
</figure>

RARE-1BはHR@500で0.5134、MAPで0.1845となり、どちらも比較手法中で最高でした。次点のHR@500はBERT-baseの0.4714、MAPはSimBERT-v2-Rの0.1797です。一方、HR@50はRAREが0.0985で、BERT-baseの0.1038やBERT-smallの0.0995を下回ります。上位50件より上位500件で差が大きいことから、後段rankerへ広めの候補を供給する用途で強みが表れています。

ACRは、広告を1件以上取得できたrequestの割合です。RAREは95.05%で高いものの、Qwen-1.8Bの96.13%、DSIとSubstrの96.15%より低い値です。論文の結果は、RAREがすべてのmetricで首位という意味ではありません。

## Ablationが示すのは各部品の異なる役割

Table 3では、Knowledge Injection（KI）とConstrained Beam Search（CBS）を外した構成を比較しています。

| 構成 | HR@500 | MAP | ACR | 平均CI数 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| KIなし | 0.1706 | 0.1540 | 59.51% | 22.78 | 90.4% |
| CBSなし | 0.1868 | 0.1687 | 67.12% | 4.84 | 95.2% |
| KI・CBSなし | 0.1562 | 0.1592 | 48.28% | 9.09 | 94.5% |
| RARE | 0.5134 | 0.1845 | 95.05% | 74.49 | 96.5% |

KIを外すとAccuracyだけでなくACRとHR@500も大きく下がります。CBSを外した場合、平均CI数は74.49から4.84へ減り、候補の多様性を確保しにくくなりました。Appendixの定性例も、zero-shot、KIのみ、format fine-tuningまで、CBSまで加えた構成の順に関連性と多様性を比較しています。

なお本文は、KIなしの59.51%を「recall rate」と記述していますが、Table 3の同じ値はACR列です。記事では表の定義に従い、59.51%をACRとして扱います。

## 本番A/Bテストは売上指標まで改善した

著者らは、WeChat Search、Demand-Side Platform、QQ Browser Searchという3環境へRAREを導入しました。いずれも日次リクエストが数十億規模とされ、1か月、利用者の20%を対象にA/Bテストを行っています。

WeChat Searchでは、広告消費額が5.04%、GMVが6.37%、CTRが1.28%、shallow conversionが5.29%、deep conversionが24.77%増えました。Demand-Side Platformでは消費額が7.18%、GMVが5.03%増えています。QQ Browser Searchでは消費額が4.50%、GMVが5.02%、shallow conversionが17.07%増えた一方、CTRは0.74%低下しました。

これは本番規模でbusiness metricまで測った重要な結果です。ただし論文には、対照群の絶対値、利用者数、分散、信頼区間、統計的有意差の検定方法が掲載されていません。「significant benefits」という記述を統計的有意性の報告とはみなせません。また著者は全員Tencent所属で、評価も同社システム上の結果です。

## 「end-to-end」という表現には境界がある

RAREはqueryからCIを生成し、広告候補を直接引くため、従来のkeyword検索より経路を短くします。しかし論文のLimitationsには、queryと広告の関連性はdownstream processが管理すると明記されています。LLMが生成段階でqueryと広告の関連性まで一貫して判定するわけではありません。

さらに、広告indexの規模は数千万件、CI集合は約200万種類と示されていますが、index容量、更新時間、1 requestあたりの候補数、rankerを含むend-to-end latencyは公開されていません。60msはFigure 5で示された生成推論の安全上限であり、広告配信全体のp99 latencyとは区別する必要があります。

評価データも1日分のhead queryが中心です。long-tail query、新商品、季節変化、別言語への一般化は、この実験だけでは判断できません。新しいCI集合を月次更新し、ブランド名や商品知識を定期的に注入する運用は示されていますが、更新による性能変化やrollback方法は報告されていません。

## 小規模に試すなら固定語彙の意味routerから始める

公開情報だけではRAREを完全再現できません。それでも設計の中心である「LLMが短い意味keyを生成し、既存indexを引く」構造は、小規模な検証に分解できます。

まず、商品や文書から人が確認できる意図語彙を作り、各itemを複数の意図へ割り当てます。次に、prefix trieを使う制約付きdecodingと、自由生成後のnearest-neighbor mappingを比較します。評価対象はquery単位のHR@KとMAPだけではありません。意図を一つも生成できない割合、item coverage、生成時間、平均意図数も記録します。

導入順序は、オフラインで候補生成だけを比較し、shadow trafficでlatencyとcoverageを測り、少量のonline trafficで既存retrieverとの併用から始めるのが安全です。CI生成が失敗した場合は既存検索へ戻し、index revision、model revision、生成CI、cache hit、候補数をlogへ残します。これは論文の構成を一般化した実装案で、Tencentが公開した導入手順ではありません。

広告以外にも、categoryや意図を共有するitemが多く、一つの意味keyから複数候補へ展開できる検索に向いています。反対に、各itemが固有で意味keyを共有しにくい領域や、最新itemを即時に固有名で検索する用途では、一対多の圧縮効果が小さくなります。

## まとめ

RAREは、LLMを巨大な広告indexそのものとして使わず、短いCIを生成する意味routerとして配置しました。CI集合を固定し、転置インデックス、cache、制約付きbeam searchと組み合わせることで、数千万件規模の広告検索を本番の応答時間へ収めています。

結果を読むときは、RARE単体のモデル性能とシステム全体の工夫を分ける必要があります。オフラインではHR@500とMAPが比較手法中で最高でしたが、HR@50やACRは首位ではありません。本番A/Bテストでは複数の売上・conversion指標が改善した一方、統計検定やend-to-end latencyの詳細は非公開です。

再利用しやすい発想は、生成対象をitem IDから共有可能な意味keyへ変え、LLMと従来検索の役割を分担した点です。生成検索を本番へ入れる際は、モデルだけでなく、語彙の更新、cache、index更新、fallbackまでを一つの検索システムとして評価する必要があります。

## 参照

- Tongtong Liuほか, [Real-time Ad Retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising](https://aclanthology.org/2025.emnlp-main.1473/), EMNLP 2025, pp. 28948–28960. [PDF](https://aclanthology.org/2025.emnlp-main.1473.pdf)
