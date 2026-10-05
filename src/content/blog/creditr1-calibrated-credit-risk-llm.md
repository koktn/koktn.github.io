---
title: CreditR1解説：Brier報酬でLLMのデフォルト確率を校正する
description: 中国A株企業の信用リスクを対象に、SFTとGRPO、Brier score、pairwise ranking、根拠検証を組み合わせたCreditR1の設計と評価結果、限界を解説します。
publishedAt: 2026-10-05
category: AI
tags:
  - Large Language Models
  - Credit Risk
  - Reinforcement Learning
  - GRPO
  - Calibration
draft: false
---

> AI利用の明示
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。数値や主張は原論文を確認して記載していますが、利用時は原文も確認してください。

今回取り上げるのは、Yuxuan Wuらの査読論文「[CreditR1: Calibration-Aware Reinforcement Learning for Interpretable Corporate Credit Risk Assessment with Large Language Models](https://doi.org/10.3390/math14152702)」です。MDPIの*Mathematics*に2026年7月28日付で掲載され、[HTML全文](https://www.mdpi.com/2227-7390/14/15/2702)と[PDF](https://www.mdpi.com/2227-7390/14/15/2702/pdf)がCC BY 4.0で公開されています。

この論文の価値は、LLMに企業のデフォルト確率を答えさせるだけでなく、その確率が実際の発生率と合うようにBrier scoreをGRPOの報酬へ組み込んだ点にあります。中国A株企業の2023年データでは、CreditR1のAUCは0.883で、XGBoostの0.891との統計的な差はありませんでした。ECEはXGBoostの0.089から0.047へ低下しています。

ただし、テストデータのデフォルト事例は119件です。学習コードと処理済みデータも公開されていないため、この記事では論文の報告値と、そこから読み取れる実務上の示唆を分けて説明します。

## AUCが高くても、PDが正しいとは限らない

信用リスクモデルは、危険な企業を上位へ並べるだけでは不十分です。自己資本や貸倒引当金の算定に使うPD（Probability of Default）には、予測確率と実際のデフォルト率が対応するcalibration（確率校正）も求められます。

論文はdiscriminationとcalibrationの違いを、極端な例で説明しています。デフォルト企業へ0.99、非デフォルト企業へ0.98を出すモデルは、全企業の順序を正しく並べるためAUCは1.0です。しかし、確率としては過大です。反対に全企業へ母集団のデフォルト率だけを出せば、その集団全体では校正されても、危険度の順位を付けられません。

| 性質 | 問い | 論文の主な指標 |
| --- | --- | --- |
| Discrimination | デフォルト企業を非デフォルト企業より上位へ並べられるか | AUC、KS |
| Calibration | 予測したPDと実際の発生率が対応しているか | ECE、ACE、class-balanced Brier score |
| Faithfulness | 説明中の数値を入力資料までたどれるか | citation precision、fabrication rate |

デフォルト率が約5%のデータで正誤だけを報酬にすると、すべてを非デフォルトと答える方策でも高い報酬を得られます。CreditR1は、確率の正しさ、順位、説明の根拠、出力形式を別々の報酬にして、この崩れ方を抑えます。

## 対象データと時間分割

対象は2015〜2024年の中国A株上場企業です。金融業を除き、CSMARとWINDから23個の指標を集めています。内訳は収益性、レバレッジ、流動性、活動性、支払能力に関する19個の財務比率と、売上成長率、営業キャッシュフロー、監査意見、業種コードです。さらに年次報告書のMD&Aから、文の境界で切った512 tokenの抜粋を入力します。

予測日は年度末ではなく、年次報告書が実際に開示された日です。その日から12か月以内に、ST指定、債券デフォルト、2 notch以上の格下げのいずれかが起きると正例になります。予測時点ですでにST指定を受けていた企業は除外されています。

| Split | 期間 | Firm-years | デフォルト事例 | デフォルト率 |
| --- | ---: | ---: | ---: | ---: |
| Training | 2015〜2021 | 12,847 | 539 | 4.20% |
| Validation | 2022 | 2,156 | 103 | 4.78% |
| Test | 2023 | 2,341 | 119 | 5.08% |
| OOD test | 2024 | 2,089 | 120 | 5.74% |

企業名、銘柄コード、子会社名、地理情報は匿名tokenへ置き換え、MD&AにもNERを適用しました。識別可能なentityは、1抜粋あたり平均4.7個から0.3個へ減っています。データも時間で分割し、事前学習データの記憶を評価結果と取り違えにくくしています。

## SFTから複合報酬のGRPOへ進む

CreditR1の学習は、データ構築、SFT、GRPOの3段階です。次の図では、入力から4つの報酬までの関係を追ってください。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/creditr1-training-pipeline-mobile.svg">
    <img src="/img/posts/creditr1-training-pipeline.svg" alt="財務指標とMD&Aを匿名化し、根拠を検査したSFTを経て、4種類の報酬でGRPO学習するCreditR1の処理図" loading="lazy">
  </picture>
  <figcaption>Wuら「CreditR1」Figure 1・§3.2〜3.5を基に本記事用に再構成。原図の転載ではありません。原論文は<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>です。</figcaption>
</figure>

### Stage 0：入力を一定の形式へ変換する

23個の財務指標には実数値と業種内percentileを付け、MD&A抜粋と指示文を一つのpromptへ入れます。promptは固定templateから自動生成するため、サンプルごとの手作業はありません。モデルはreasoning chainと0〜1のPDをXML tag付きで出力します。

### Stage 1：根拠を検査したSFTで形式を覚える

GPT-4oが教師となり、学習サンプルごとのreasoning chainを生成します。教師には正解ラベルを見せません。その後、結論の方向がラベルと一致し、説明中の数値の80%以上を入力へたどれるchainだけをSFTへ使います。通過率は全体で72%、デフォルト事例で65%、非デフォルト事例で73%でした。

このフィルターは結果と合う説明だけを残すため、outcome-consistent biasを生み得ます。論文もこの点を認めています。一方、フィルターを外したablationではAUCが0.007低下し、ECEが0.011上がりました。後段のGRPOが一部を補っても、初期データの品質は最終結果に影響します。

base modelはQwen2.5-7Bです。3 epochのSFTで得たモデルを、GRPOの初期policyと固定reference policyの両方に使います。

### Stage 2：4種類の報酬でGRPOを学習する

GRPOでは1 promptにつき8個の応答を生成し、グループ内で報酬を正規化します。3000 stepの各stepで64 promptを処理して、学習中のpolicyだけを更新する構成です。別のvalue networkや学習型reward modelは使いません。

複合報酬は次の加重和です。

```text
R = 0.3 × R_Brier
  + 0.3 × R_Rank
  + 0.2 × R_Evid
  + 0.2 × R_Fmt
```

`R_Brier = 1 - (p - y)²`は、正しい条件付き確率を答えたときに期待報酬が最大になるstrictly proper scoring ruleです。ただし、この性質が保証されるのはBrier報酬単体に限られます。順位などのnon-properな項を加えた複合報酬全体には、同じ理論保証がありません。最終的な校正は実験で確かめる必要があります。

`R_Rank`は、デフォルト企業のPDが同業種の非デフォルト企業より高くなるようにするpairwise ranking報酬です。同じCSRC細分類と年度の企業を組にできた割合は91.3%でした。該当企業がなければ上位の業種区分を使い、それでも見つからなければ総資産percentileが近い企業を選びます。Brier報酬だけで全企業のPDが約5%へ集まるのを防ぎ、AUCを維持する役割です。

`R_Evid`はreasoning chainに書かれた数値が入力にもある割合です。相対誤差1%以内を一致とし、日付や序数などは除外します。数値を一つも書かなければ0になるため、根拠のない曖昧な説明でも満点にはなりません。ただし、数値以外の因果説明までは検証しません。

`R_Fmt`はXML構造と0〜1のPDを機械的にparseできるかを判定します。4項を乗算せず加算することで、一つの失敗により他の学習信号まで0になるのを避けています。

## AUCを保ちながらECEを下げた

2023年test setでは、CreditR1のAUCは0.883 ± 0.004、ECEは0.047 ± 0.006でした。`±`は3 seedの標準偏差です。一方、次の図のXGBoost系は決定的な1回の結果です。

<figure class="article-figure">
  <picture>
    <source media="(max-width: 600px)" srcset="/img/posts/creditr1-main-results-mobile.svg">
    <img src="/img/posts/creditr1-main-results.svg" alt="2023年test setにおけるCreditR1とXGBoost系モデルのAUCとECEを比較した棒グラフ" loading="lazy">
  </picture>
  <figcaption>原論文Table 5を基に本記事用に可視化。2023年test setは2,341 firm-years、デフォルト119件です。CreditR1のみ3 seedの平均を表示しています。</figcaption>
</figure>

| Model | AUC ↑ | ECE ↓ | Class-balanced Brier ↓ |
| --- | ---: | ---: | ---: |
| XGBoost | 0.891 | 0.089 | 0.142 |
| XGBoost + isotonic | 0.891 | 0.062 | 0.126 |
| XGBoost + beta | 0.891 | 0.058 | 0.124 |
| XGBoost + text | **0.897** | 0.083 | 0.137 |
| SFT-only | 0.862 | 0.118 | 0.158 |
| CreditR1 structured-only | 0.871 | 0.052 | 0.128 |
| CreditR1 | 0.883 ± 0.004 | **0.047 ± 0.006** | **0.119 ± 0.005** |

CreditR1とXGBoostのAUC差0.008は、DeLong testで有意ではありませんでした（p = 0.31）。一方、ECEは未校正XGBoostより47.2%、isotonic calibration後のXGBoostより24.2%低い値です。CreditR1との差の95% bootstrap CIは、それぞれ0.028〜0.056と0.003〜0.027でした。

事後校正で最良だったbeta calibrationのECEは0.058です。CreditR1との差0.013も、Holm–Bonferroni補正後に有意でした（95% CI 0.002〜0.025、補正後p = 0.042）。ただしデフォルト事例は119件しかなく、相対改善率は不確実性を含む点推定です。

ECEはbinの切り方で値が変わります。論文は5、10、15、20個のequal-frequency binで再計算し、いずれでもCreditR1、再校正したXGBoost、未校正XGBoostの順序が変わらないことを確認しています。5〜20 binでCreditR1のECEは0.041〜0.052でした。

## Ablationで分かった4つの役割

| Variant | AUC ↑ | ECE ↓ | 観測された主な問題 |
| --- | ---: | ---: | --- |
| CreditR1 | 0.883 | 0.047 | — |
| Brier報酬なし | 0.879 | 0.112 | 校正が悪化 |
| Ranking報酬なし | 0.841 | 0.058 | PD分布が圧縮し、順位性能が低下 |
| Evidence報酬なし | 0.881 | 0.051 | 数値のfabricationが増加 |
| Format報酬なし | 0.876 | 0.054 | 無効出力と抽出の不安定さが増加 |
| Binary reward | 0.853 | 0.138 | 順位と校正の両方が悪化 |

Brier報酬を除くとECEは0.047から0.112へ2倍以上になりました。Ranking報酬を除くとAUCが0.042下がります。正誤だけのbinary rewardはAUCが0.030下がり、ECEが0.091上がりました。約5%しか正例がない確率予測で、正誤報酬だけを使う問題が結果にも表れています。

## 説明の数値は追跡しやすくなったが、完全ではない

2023年test setに対するreasoning評価では、CreditR1のcitation precisionは0.94でした。説明中の追跡できない数値を一つ以上含むchainの割合は7%です。SFT-onlyは19%、promptだけのGPT-4oは38%でした。

3人の金融系大学院レベルのraterが、モデル名を伏せた100 chainを評価しました。CreditR1は根拠、論理的一貫性、risk-factor coverageのすべてで最も高い平均値を得ています。ただし、実務のcredit officerを含む評価ではなく、サンプルも100件です。さらに`R_Evid`が検査するのは数値の一致であり、定性的な主張や因果関係が正しいことまでは保証しません。照合先も入力promptであって、元の開示資料全体ではありません。

## 2024年への時間shiftとcontamination probe

2024年のOOD testでは、CreditR1のAUCが0.883から0.869へ、ECEが0.047から0.056へ変化しました。XGBoostのECEは0.089から0.102、isotonic calibration版は0.062から0.078です。CreditR1も悪化していますが、論文の設定では事後校正より小さな変化に収まりました。

事前学習データの記憶を調べるため、論文は4種類のprobeも実施しています。

- 匿名化した企業名の復元率は2.3%で、random baselineの1.8%と有意差がない（p = 0.42）
- Qwen2.5の事前学習cutoff前後に起きた事例のAUC差は0.008で、有意差がない（p = 0.68）
- 財務数値を±2%揺らしたとき、元のPDとのSpearman相関は0.974、結論が反転した割合は2.1%
- 実名をpromptへ戻してもAUCの増加は0.003で、有意差がない（p = 0.52）

これらの結果からは、企業固有の結果を単純に記憶していた証拠は見つかりませんでした。ただし、有限個のprobeでcontaminationを完全に否定することはできません。論文も同じ留保を置いています。

## 実務ではGBDTを置き換えず、second readerにする

学習には4台のNVIDIA H200を使い、SFTが約4.5時間、GRPOが約13.7時間かかりました。CreditR1の推論は1社あたり2.8秒です。XGBoostの0.3msと比べると約4桁遅いため、リアルタイムの大量審査には向きません。

論文が想定するのは、GBDTで全体をscreeningし、高risk層と判断の境界付近だけをCreditR1へ回す構成です。validation splitでrisk上位20%と境界帯を処理した試算では、LLMの推論量を約72%減らしながら、実際のデフォルトの86%を対象にできました。CreditR1は既存のscorecardを置き換えるモデルではありません。確認可能な説明と校正済みPDが必要な案件のsecond readerと考える方が現実的です。

導入時には、少なくとも次を分けて検証する必要があります。

1. 時間で分けたholdoutでAUCとcalibrationを同時に測る
2. ECEだけに依存せず、bin数、reliability diagram、Brier scoreも確認する
3. model version、prompt、checkpoint、decoding設定を固定し、出力の再現性を記録する
4. 入力分布のdriftと実現したデフォルトを定期的にbacktestする
5. LLM停止時は従来モデルへ戻せるようにし、両modelの不一致を人手確認へ回す

この手順は、公開情報から本記事で整理した導入案です。論文は四半期ごとのcalibration backtestやdrift alarmを提案していますが、具体的なしきい値や運用コードは公開していません。

## 再現性と適用範囲

Qwen2.5-7B、veRL、主要なhyperparameterは論文に記載されています。GRPOは3000 step、group sizeは8、KL係数は0.04、最大rolloutは2048 tokenです。weightには、BrierとRankingへ各0.3、EvidenceとFormatへ各0.2を割り当てています。

raw dataは商用databaseにあり、処理済みdatasetは非公開です。著者は合理的な要請があれば提供するとしていますが、学習コード、固定promptの全文、filterやverifierの実装も公開していません。このため、公開情報だけでTable 5を完全再現することはできません。

適用範囲にも注意が必要です。

- 実証対象は中国A株企業だけで、ST指定や暗黙の政府保証は市場固有である
- 2023年と2024年のデフォルト事例は119〜120件で、tail binのcalibration評価が不安定になりやすい
- 7Bより大きいmodelの効果は検証していない
- 報酬weight以外の主要hyperparameterは系統的な感度分析をしていない
- 複合報酬全体はstrictly proper scoring ruleではない
- 同じ2022年validation splitを、報酬weight選択とcheckpoint選択へ順番に2回使っている
- 人手評価は3人、100 chainで、実務のcredit officerを含まない

Basel IIIやIFRS 9に沿った本番modelには、景気循環をまたぐlong-run PD、rating migration、backtesting、model governanceなども必要です。CreditR1の結果だけで規制用途のPD modelとして導入できるわけではありません。

## まとめ

CreditR1は、信用リスクのLLMを「当たったか」だけで学習せず、確率、順位、根拠、形式を別々に評価しました。Brier報酬がcalibrationを、同業種内のpairwise rankingがdiscriminationを主に支えています。2023年の中国A株データではXGBoostに近いAUCを保ち、post-hoc calibrationを含む比較対象より低いECEを報告しました。

一方、結果は単一市場と119件のデフォルト事例に基づきます。データとコードも完全には公開されていません。実務への示唆は、LLMでGBDTを全面的に置き換えることではなく、校正済みPDと検証可能な説明が必要な案件へ限定し、人が確認するsecond readerとして使うことです。

## 参考文献

- Yuxuan Wu, Haowen Dai, Yiheng Zhang, Jinping Ma, [CreditR1: Calibration-Aware Reinforcement Learning for Interpretable Corporate Credit Risk Assessment with Large Language Models](https://doi.org/10.3390/math14152702), *Mathematics* 2026, 14(15), 2702.
- 同論文の[HTML全文](https://www.mdpi.com/2227-7390/14/15/2702)、[PDF](https://www.mdpi.com/2227-7390/14/15/2702/pdf)、[version notes](https://www.mdpi.com/2227-7390/14/15/2702/notes)
