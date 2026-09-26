---
title: 中国製二足歩行ロボットの学習ツリー――構造・制御・センシングをつなげて理解する
description: AgiBot X1、OpenLoong、Humanoid-Gym、状態推定と最新の視覚歩行研究を軸に、急速な進歩を支えた基盤技術と、構造・制御・センシングを学ぶ順序を整理します。
publishedAt: 2026-09-22
category: AI
tags:
  - Robotics
  - Humanoid Robot
  - Reinforcement Learning
  - State Estimation
  - Sim-to-Real
draft: false
---

> **AI利用の明示**
>
> 本記事の構成と本文は、OpenAIのコーディングエージェント「Codex」が作成しました。2026年9月22日時点の公開資料を確認していますが、実機で試す前に原資料と安全手順も確認してください。

二足歩行ロボットを体系的に理解する近道は、**AgiBot X1の設計資料で身体を見る、OpenLoongとHumanoid-Gymで2系統の制御を比べる、状態推定から視覚歩行へ進む**という順序です。

本稿は、中国企業・研究機関のロボットと、中国製の機体を用いた海外大学の研究を対象にします。CAD、公式仕様、論文、公開実装を「何を理解するために読むか」という学習ツリーに並べます。

## 最初に押さえる資料の性格

同じ機体名が出てきても、メーカーの標準機能と研究者が実装したアルゴリズムは別です。公開資料から確認できる範囲を先に分けておきます。

| 資料・プロジェクト | 機体 | 公開資料の性格 | ここから言えること |
| --- | --- | --- | --- |
| [AgiBot X1 hardware](https://github.com/AgibotTech/agibot_x1_hardware) | AgiBot X1 | メーカー公開のハードウェア資料 | CAD、BOM、組立手順を使って構造を調べられる |
| [Unitree G1公式仕様](https://www.unitree.com/g1/) | Unitree G1 | メーカーの製品仕様 | 自由度、モーター、エンコーダー、ベアリング、搭載センサーの構成を確認できる |
| [OpenLoong Dynamics Control](https://github.com/loongOpen/OpenLoong-Dyn-Control) | 青龍 | 企業・研究機関によるオープンな制御リファレンス | MuJoCo上のMPC、WBC、歩容生成と、実機デモの報告を追える |
| [Humanoid-Gym](https://github.com/roboterax/humanoid-gym) | RobotEra XBot-S／XBot-L | RobotEra関係者を含む研究実装 | RL歩行の学習、sim-to-sim、zero-shot sim-to-realの構成を追える |
| ASAP／FootQuery | Unitree G1 | 大学などによる研究実装 | G1で実証された手法であり、G1の標準歩行アルゴリズムとは限らない |
| ARMOR | Fourier GR1 | 研究実装 | GR1へ分散センサーと衝突回避方策を追加した例であり、標準装備の説明ではない |
| UniPoint | Deep Robotics DR02 | 浙江大学・Deep Robotics関係者による研究 | DR02で点群融合歩行を検証した結果であり、市販機の標準機能とは断定できない |
| PRIMO | AgiBot A3 Ultra | 武漢大学・AgiBotによる研究 | A3 Ultraで自己運動推定を評価した研究であり、製品全体の標準構成を示すものではない |

とくに注意したいのは、製品仕様にカメラやLiDARが載っていても、ある歩行方策がそのセンサーを入力に使っているとは限らないことです。逆に、研究では市販機へ追加センサーを取り付けている場合もあります。

## 学習ツリーの全体像

```text
0. 機体と資料の関係を区別する
   ├─ メーカー公式仕様・CAD
   ├─ メーカー／共同体の公開ツールチェーン
   └─ 研究者が特定機体へ実装したアルゴリズム
        ↓
基盤技術：接触に強いactuator → actuatorを含むSim-to-Real → GPU並列RL
        ↓
1. 構造
   ├─ X1のCAD・BOM・組立SOP
   ├─ 関節配置、質量、可動範囲、足裏
   └─ 並列リンクとmotor space／joint spaceの対応
        ↓
2. モデルベース制御                  3. 学習ベース制御
   ├─ 状態推定                         ├─ 観測・行動・報酬
   ├─ 歩容・足先軌道                   ├─ PPO・domain randomization
   ├─ MPC → 接地力                     ├─ policy → 関節目標
   └─ WBC → 全身の関節指令             └─ PD制御 → motor指令
        └──────────┬──────────┘
                   ↓
4. Sim-to-Realと全身運動
   ├─ RMA／DWL：履歴から環境・内部状態へ適応
   ├─ DeepMimic／OmniH2O：motion imitationとretargeting
   ├─ ASAP：実機dataでsimとの差を補正
   └─ BeyondMimic：motion trackingからskill合成へ
                   ↓
5. センシング
   ├─ proprioception：IMU・encoder・接触
   ├─ state estimation：InEKF／PRIMO
   └─ exteroception：depth・LiDAR・terrain表現
                   ↓
6. 視覚歩行
   ├─ FootQuery：将来の着地点から過去のdepthを検索
   └─ UniPoint：LiDARとdepth cameraをpoint levelで融合
```

左右に分かれたモデルベース制御と学習ベース制御は、競合するというより比較対象です。どちらも機体状態、接触、目標運動を扱い、最終的には高速な関節制御へ指令を渡します。異なるのは、その途中を明示的な力学モデルと最適化で解くか、シミュレーションで方策として学ぶかです。

## なぜ近年、急速に進歩したのか

中国系二足歩行ロボットの進歩を技術面から見ると、単一のbreakthroughよりも、**接触に耐えるactuator、実機差を扱う学習、試行錯誤を高速化するGPU simulation**が順に実用域へ入り、同時に使えるようになった影響が大きいと考えられます。

ここで挙げる基盤研究の多くは、中国企業によるものでも二足robotだけを対象にしたものでもありません。これらは現在の中国製robotが個別に採用した設計を証明する資料ではなく、近年の性能向上を可能にした世界的な技術stackを理解するための資料です。また、供給網、量産、投資、政策など産業面の因果は本稿の技術資料だけでは検証できないため、ここでは論じません。

### 3つのbottleneckが続けて小さくなった

| 技術上のbottleneck | 基盤資料 | 何が変わったか | 読む際の注意 |
| --- | --- | --- | --- |
| 接触・衝撃を扱える駆動系 | [Proprioceptive Actuator Design in the MIT Cheetah](https://dspace.mit.edu/entities/publication/8b15bcc8-c288-43aa-b923-69ce3c13b818)（2017） | torque densityだけでなく、force control bandwidth、backdrivability、impact mitigationを一体で設計する視点を示した | MIT Cheetahの設計であり、Unitreeなどの採用を示す資料ではない |
| simulationと実機のactuator応答の差 | [Learning Agile and Dynamic Motor Skills for Legged Robots](https://arxiv.org/abs/1901.08652)（2019） | 実機dataからactuator networkを学び、delayや低level制御を含む応答をsimulationへ入れてANYmalへ転送した | 四足robotでの結果。形状と質量だけ合わせれば十分ではないことを学ぶ資料 |
| 学習の試行回数と待ち時間 | [Learning to Walk in Minutes](https://arxiv.org/abs/2109.11978)（2021） | 1枚のGPU上で数千体を並列simulationし、curriculumとPPOで方策開発の反復を大幅に短縮した | 並列数を増やせば常に良いわけではなく、on-policy更新に必要な時間方向のsampleとのtrade-offがある |

MIT Cheetahの論文が強調するのは、最大torqueの大きさだけではありません。着地時の外力で関節が動きやすいbackdrivability、高bandwidthなforce control、衝撃を機械・制御の両方で扱えることが、dynamic locomotionの前提になります。構造を見るときは、motor specに加えて減速比、rotor inertia、伝達効率、制御帯域まで追う必要があります。

Hwangboらの2019年の研究は、そのhardwareをsimulation上でどう再現するかという次の問題を扱います。ANYmalの12 actuatorを並列に動かして4分未満で100万件超のsampleを集め、joint position errorやvelocityの履歴からtorqueを予測するnetworkを学習しました。論文では、理想的または解析的なactuator modelで学んだ方策は実機で1歩も進めず、delayやbandwidthのずれが原因と考察されています。これは後述するASAPと同じく実機差をdataで扱いますが、補正場所が異なります。

```text
Hwangbo et al. (2019)
  └─ actuatorの入出力modelを実機dataから学び、training simulatorへ入れる

ASAP (2025)
  └─ policyを実機で動かしたtrajectoryからdelta action modelを学び、policyをfine-tuneする
```

Rudinらの研究は、学習algorithmそのものより、**試行錯誤を回すsystem**が開発速度を変えたことを示します。実験では4096体のANYmalを並列に動かし、不整地用policyを単一のRTX A6000で20分未満に学習したと報告しています。terrainごとに成功すれば難しく、失敗すれば易しくするcurriculumも、広い条件を一度に学ばせる際の重要な要素です。個々の数値は同論文のhardwareと実装条件に依存しますが、「rewardやrandomizationを変えて翌日を待つ」状態から、「短いiterationで比較する」状態への変化が本質です。

以上から本稿が導く技術的な解釈は、次のとおりです。

```text
高性能なactuatorと低level control
  × actuator・delayを含むSim-to-Real
  × GPU上の大量並列simulation
  × curriculum・domain randomization
  → 実機を壊す前に多数の案を絞り、短いcycleでrobotへ移せる
```

Humanoid-Gym、UnitreeのRL toolchain、ASAPなどは、この世界的な基盤が中国製humanoidのmodel、実機interface、公開codeと結びついた例として読めます。ただし、公開論文から確認できるのは技術的な接続であり、「中国勢だけが伸びた理由」を国別に因果推定した結果ではありません。

### 模倣、適応、学習しやすい機体へ発展した

追加で読む3本は、現在のhumanoidへつながる別の枝を補います。

- [DeepMimic](https://arxiv.org/abs/1804.02717)（2018）は、reference motionを追う模倣目的とtask目的を組み合わせました。ASAPやBeyondMimicへ進む前に、人間らしい運動とtask達成をreward上でどう分担するかを学べます。ただし、結果は物理simulation内のcharacterとAtlas modelであり、実機転送を示した研究ではありません。
- [RMA](https://arxiv.org/abs/2107.04034)（2021）は、training時に得られるfrictionやpayloadなどのprivileged informationをlatent表現へ圧縮し、実行時には直近0.5秒のstate・action履歴からその表現を推定します。Unitree A1へfine-tuningなしで展開しました。ここでいうonline adaptationは、実機上でnetwork weightを再学習することではなく、学習済みadaptation moduleが環境に応じたlatentを更新することです。
- [Berkeley Humanoid](https://arxiv.org/abs/2407.21781)（2024）は、複雑な閉linkやelastic elementを避け、通信delayを抑え、転倒に耐える小型機を作ることで、simulationを単純にし、軽いdomain randomizationと基本的なMLP policyでも実機転送しやすくする考え方を示します。中国製機体ではありませんが、「高度な学習器」だけでなく「学習しやすいhardware」を設計する比較対象になります。

「なぜ進歩が速くなったか」を先に知りたい場合は、次の順で読むと論点がつながります。

1. Learning to Walk in Minutes：開発cycleを変えた大量並列学習
2. MIT Cheetahのactuator論文：接触を扱うhardware条件
3. Hwangboらの2019年論文：actuatorを含むSim-to-Real
4. Humanoid-Gym：これらを二足歩行のtraining pipelineへ落とす方法
5. ASAP：特定実機で観測した差を使ってさらに合わせる方法

各論文では、結果だけでなく「機体を変えたか、modelを変えたか、sample数を増やしたか」「ablationで何を外すと崩れるか」を確認します。これにより、hardware、algorithm、計算資源の寄与を分けて読めます。

## 1. 構造：X1の設計資料から「制御される身体」を読む

構造の学習では、自由度の数だけでなく、モーターの回転がどの経路で関節へ伝わるかを見ます。最低限、次の5点を対応づけます。

| 着眼点 | 制御・性能へ現れる影響 |
| --- | --- |
| 関節配置と可動範囲 | しゃがみ、脚の振り出し、到達可能な着地点 |
| モーターと減速・伝達機構 | 最大トルク、速度、backdrivability、衝撃時の応答 |
| 脚の質量分布 | swing legの慣性、加減速のしやすさ、消費energy |
| 膝・足首のlinkage | motor角とjoint角の非線形な対応、姿勢ごとの出力特性 |
| 足裏と接触部 | 支持領域、摩擦、衝撃、接触状態の観測方法 |

### X1でCAD、BOM、組立を往復する

[AgiBot公式のX1設計資料ページ](https://www.agibot.com.cn/DOCS/OS/X1-PDG)とGitHub repositoryには、日付別のdirectoryがあり、2025年3月7日版には部品単位のSTEP、SolidWorks 2022のsource、全体図面、BOM、工具list、組立SOP、組立動画へのlinkが含まれます。

学び方は、完成した3D modelを眺めるだけでは不十分です。

1. 全体STEPでhip、knee、ankleの回転軸とlink長を確認する。
2. BOMでactuator、bearing、fastenerを特定する。
3. 部品STEPでmotorからjointまでの伝達経路を追う。
4. 組立SOPで軸受、配線、締結の順序を確認する。
5. 同じ部分をURDFで探し、collision形状、質量、慣性、joint limitと照合する。

URDFは制御・simulationに必要な抽象化ですが、製造形状や閉linkの内部まで必ず表現するわけではありません。CADとURDFの差を見ることが、「実機」と「制御model」の差を理解する最初の演習になります。

### G1の仕様は部品選定の比較軸に使う

UnitreeのG1公式仕様には、片脚6自由度、低慣性・高速の内転子PMSM、dual encoder、joint output部のcross roller bearingなどが掲載されています。標準G1とG1 EDUでは腰や腕などの構成が異なるため、論文に書かれた自由度数を製品ページの一つの列だけで判断してはいけません。

ここでの目的はG1を詳細設計の代わりに使うことではなく、X1のBOMやCADを読むときに「actuator、encoder、bearing、joint limitをどの粒度で比較するか」という観点を得ることです。

### 並列機構を制御modelへつなぐ

近年のhumanoidでは、脚先側の質量を減らすため、motorをjoint軸から離し、膝のfour-bar linkageや2自由度のparallel ankleを使う構成があります。この場合、単純なserial joint modelではmotor角、joint角、torque、impedance gainの関係を正しく扱えないことがあります。

[Control of Humanoid Robots with Parallel Mechanisms using Differential Actuation Models](https://arxiv.org/abs/2503.22459)は、膝と足首の非線形な伝達を解析的なactuation modelとして表し、DDPによる最適化とPPOによる学習へ組み込んでいます。図中にはG1、H1、GR1などが近年のparallel architectureの例として登場しますが、実験機はそれらの市販機ではありません。中国製ロボットの標準制御を説明した論文ではなく、共通する機構上の問題を理解する補助資料です。

さらに設計側へ進むなら、[A Framework for Optimal Ankle Design of Humanoid Robots](https://arxiv.org/abs/2509.16469)が、parallel ankleのworkspace、actuator、性能指標をmulti-objective optimizationで比較しています。CADの形を制御性能やtask requirementへ結びつけたい段階で読む資料です。

この段階の到達目標は、任意の脚について次の変換を説明できることです。

```text
motor位置・速度・torque
  → transmission／linkage
  → joint位置・速度・torque
  → foot poseと接触力
  → robot全体の重心・運動量
```

## 2. 制御：OpenLoongとHumanoid-Gymを並行して読む

### モデルベース制御：OpenLoongでdata flowを追う

[OpenLoong Dynamics Control](https://github.com/loongOpen/OpenLoong-Dyn-Control)は、青龍のMuJoCo modelと、歩行、jump、視覚を使わない障害物踏破のdemoを公開しています。repositoryは実機で歩行とblind obstacle steppingを実現したと報告していますが、公開手順の中心はMuJoCo simulationです。

最初に追うdata flowは次のとおりです。

```text
sensor／MuJoCo state
  → state estimationとforward kinematics
  → gait schedulerがsupport legとswing legを決定
  → foot placementが着地点とswing trajectoryを生成
  → MPCが重心運動とcontact forceを予測・最適化
  → WBCがbase、foot、hand、contactのtaskを全身へ配分
  → PVT／joint controllerがmotor指令を生成
```

MPC（Model Predictive Control）は、少し先までのbase姿勢、位置、角速度、速度と、左右足の力・momentを扱います。WBC（Whole-Body Control）は、静止接触、胴体姿勢、水平位置、swing leg、hand trackingなどのtask priorityを、joint accelerationやcontact forceの制約のもとで調整します。

repository内で読む順序は、`demo/walk_mpc_wbc.cpp`から各moduleへの呼び出しを追い、次に`GaitScheduler`、`FootPlacement`、`MPC`、`WBC`、`PVT_ctrl`へ進む形が分かりやすいです。parameterを変える前に、各値がworld frame、local frame、support foot frameのどれで表現されるかを確認します。

### 学習ベース制御：Humanoid-Gymで最小構成を分解する

[Humanoid-Gymの論文](https://arxiv.org/abs/2404.05695)と[実装](https://github.com/roboterax/humanoid-gym)は、RobotEra XBot-S／XBot-Lでzero-shot sim-to-realを検証したRL locomotionの入口です。学習時はIsaac Gym、sim-to-sim検証にはMuJoCoを使います。

主要componentを対応づけると次のようになります。

| Component | Humanoid-Gymでの役割 |
| --- | --- |
| Observation | gait clock、速度command、joint position／velocity、base角速度・姿勢、前回action |
| Privileged state | friction、mass、base linear velocity、外力、contactなどをcritic側で利用 |
| Action | 12 jointのtarget position |
| Low-level control | target positionをPD controllerが追従 |
| Reward | 速度、姿勢、高さ、contact pattern、joint追従、energy、smoothnessなど |
| Sim-to-Real | system delay、friction、motor strength、payload、sensor noiseなどをrandomize |

論文の構成ではpolicyが100 Hz、内部PD controllerが1,000 Hzで動きます。つまり、neural networkが毎millisecondのmotor出力を直接すべて決めるのではありません。低速側の方策が関節目標を更新し、その間を高速なfeedback loopが支えます。この周波数はHumanoid-Gymの設定であり、全機体に共通する仕様ではありません。

OpenLoongとHumanoid-Gymの対応を並べると、違いが見えやすくなります。

| 問い | OpenLoong | Humanoid-Gym |
| --- | --- | --- |
| 次の運動をどう決めるか | gait、foot placement、MPC、WBCを明示的に計算 | policyが観測からjoint targetを出す |
| 接触をどう扱うか | contact制約とforceをmodel内で扱う | contactをreward、privileged state、simulation dynamicsで学ぶ |
| 調整箇所 | cost weight、task priority、gain、step parameter | observation、reward、randomization、network、curriculum |
| 失敗の調べ方 | model、constraint、solver、frame、gainを追う | reward、data distribution、value、randomization、sim差を追う |
| 強み | 中間量の意味を追いやすい | 高次元・非線形な対応をsimulationから獲得できる |
| 主な弱点 | model誤差と最適化cost | 学習分布外の挙動と原因説明の難しさ |

## 3. RL歩行から全身運動へ枝を伸ばす

Humanoid-Gymの後は、解こうとしている問題の違いを意識して読むと整理しやすくなります。

### DeepMimic：motion imitationの出発点を押さえる

[DeepMimic](https://arxiv.org/abs/1804.02717)は、motion captureなどのreferenceを追うimitation objectiveと、目標方向へ歩くといったtask objectiveを組み合わせました。単にposeを再生するのではなく、物理simulation内で外乱から回復し、目的に応じてreferenceから外れる余地を方策に与えます。

本稿では実機humanoidの成果としてではなく、OmniH2O、ASAP、BeyondMimicへ続く「reference motionを物理的に成立するcontrolへ変える」という発想の基盤として位置づけます。

### RMA：観測履歴から環境変化へ適応する

[Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034)（RMA）は、base policyとadaptation moduleを分け、直近のstate・action履歴からfriction、payload、motor strengthなどに応じたlatentを推定します。trainingは全てsimulationで行い、Unitree A1へpolicyのfine-tuningなしで展開しています。

RMAを読むときは、実行時の適応とnetworkのonline学習を区別します。実機上では、学習済みadaptation moduleが10 Hzでlatentを更新し、base policyが100 Hzでjoint targetを出します。weightをその場で更新する方式ではありません。また四足robotの研究なので、二足のbalanceや全身運動へ同じ性能が自動的に移るわけではありません。

### DWL：観測できない状態を内部で推定する

[Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning](https://arxiv.org/abs/2408.14472)は、Denoising World Model Learning（DWL）によってstate estimationとsystem identificationをRL frameworkへ組み込みます。proprioceptionからbase velocity、foot contact、terrainの大まかな高さを推定し、同じnetworkで階段、斜面、不整地などへ対応します。

これはcameraなしで地形を完全復元する手法ではありません。論文自身もpreciseな形状推定は難しいと述べています。足が触れた結果や身体の応答から、制御に役立つlatent stateを得る研究として読みます。

### OmniH2O：人間の運動をrobotへ移す

[OmniH2O](https://arxiv.org/abs/2406.08858)は、人間motionのretargeting、privileged teacher、sparse sensor inputを使うdeployable student policyという流れを示します。人間とrobotでは骨格、link長、joint range、接触条件が異なるため、人間のposeをそのままjoint commandにはできません。

学ぶべきなのは、次の2段階です。

```text
human motion
  → kinematic retargetingでrobotのreference motionを作る
  → physics simulation内のtracking policyで実現可能な運動にする
```

### ASAP：実機dataでsimulationとの差を学ぶ

[ASAP](https://arxiv.org/abs/2502.01143)は、Unitree G1でpretrained policyを動かして実機trajectoryを集め、simulationと実機のずれを補うdelta action modelを学びます。そのmodelをsimulationへ組み込み、元のtracking policyをfine-tuneした後、delta modelなしで実機へ戻します。

重要なのは、domain randomizationを広げ続けるだけでなく、実際に現れた誤差をdataとして使う点です。ただし実機data収集には安全な初期policyが必要で、激しい失敗を含む探索を無制限に行えるわけではありません。

### BeyondMimic：追従からskillの合成へ進む

[BeyondMimic](https://arxiv.org/abs/2508.08241)は、まずdynamicなmotion trackingを成立させ、そのmotion primitiveをguided diffusionでtaskに合わせて組み合わせます。waypoint navigation、joystick control、obstacle avoidanceなどを実機で示しています。

読む順序は、Humanoid-Gymでvelocity-command locomotionを理解し、OmniH2O／ASAPでmotion trackingとsim-to-realを学び、最後にBeyondMimicでtask-conditionedなmotion synthesisを見るのが自然です。

## 4. センシング：身体状態と外界形状を分ける

センシングは、sensor名ではなく「何を推定してcontrolへ渡すか」で分類します。

| 分類 | 主な入力 | 推定・認識するもの | 主な失敗 |
| --- | --- | --- | --- |
| Proprioception | joint encoder、IMU、motor電流・torque関連値 | 姿勢、速度、joint state、接触、slip | bias、impact、model誤差、接触仮定の破れ |
| Contact sensing | 足裏force／pressure、joint torque | 荷重、contact位置、support状態 | 衝撃、飽和、足裏の局所接触 |
| Exteroception | RGB、depth camera、LiDAR | 段差、障害物、foothold、周囲との位置関係 | occlusion、画角、照明、反射、latency |

### 状態推定の基礎：contact-aided InEKF

[Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation](https://arxiv.org/abs/1904.09251)は、IMUでstateをpropagateし、forward kinematicsと接触情報でpose、velocity、contact pointをcorrectする枠組みです。Cassieでの実験で、通常のquaternion-based EKFと比較しています。

この論文から学ぶべきなのはfilterの式だけではありません。

- IMU単独ではbiasと積分誤差が蓄積する
- 接地している足はworldに対して静止している、という仮定が観測を与える
- 足がslipすると、その仮定が破れて推定も崩れる
- yawやglobal positionなど、sensor構成だけでは観測しにくいstateがある
- contactの追加・削除とIMU biasをstateにどう含めるか

つまりstate estimatorはcontrolの前処理ではなく、接触modelとsensorの信頼度を管理するcomponentです。

### 外界表現の基礎：確率的elevation map

[Probabilistic Terrain Mapping for Mobile Robots with Uncertain Localization](https://www.research-collection.ethz.ch/items/563227f2-bb05-434b-8aef-1b001a9fdebc)は、range sensorのnoiseだけでなく、robot自身のlocalization uncertaintyも含めてgrid-based elevation mapを作ります。

歩行policyへdepthやpoint cloudを入れる前に、次のdata flowを理解するのに適しています。

```text
range measurement
  + sensor extrinsics
  + robot poseとそのuncertainty
  → world／robot-centricなterrain表現
  → foothold selectionやlocomotion policy
```

一方、aggressive motionではodometry driftがmapへ入り、thin barrierのような垂直構造は2.5D elevation表現から失われやすいという制約があります。この弱点が、後述するpoint-level fusionの動機につながります。

### Sensor配置を学ぶ：ARMOR

[ARMOR](https://arxiv.org/abs/2412.00396)は、Fourier GR1の腕へ分散配置したToF sensorを使い、頭部cameraだけではocclusionする領域を補います。実機では28個のToF LiDARと15 Hzの更新loopを使い、主に上半身のcollision avoidanceとmotion planningを扱います。

したがって、ARMORは二足歩行controller全体の論文ではありません。ここから学ぶのは、sensor性能だけでなく、**どの身体部位に置けばtaskに必要な空間が見えるか**というco-designです。

## 5. 2026年9月の視覚歩行・自己運動推定

2026年9月22日時点で、とくに新しい3件は次のとおりです。いずれも公開直後のarXiv v1で、長期運用や第三者再現が蓄積した手法ではありません。UniPointはRA-Lへ投稿中、PRIMOはunder reviewと明記されています。

| 研究 | 公開日・機体 | 入力と出力 | 解いている問題 |
| --- | --- | --- | --- |
| [FootQuery](https://arxiv.org/abs/2609.21447) | 9月18日／Unitree G1 | proprioception＋過去のdepth → joint target | 接近時には見えたが、着地時には画角外・自己遮蔽になる足場を記憶から取り出す |
| [UniPoint](https://arxiv.org/abs/2609.23666) | 9月20日／Deep Robotics DR02 | 360° LiDAR＋前後depth camera＋proprioception → joint target | 複数sensorを固定数のpoint tokenへまとめ、広い視野、局所精度、sensor故障時の冗長性を両立する |
| [PRIMO](https://arxiv.org/abs/2609.23610) | 9月20日／AgiBot A3 Ultra | IMU＋下肢・腰のjoint state → velocity・rotation | deployment policyが変わっても再利用しやすいproprioceptive odometryを学ぶ |

### FootQuery：将来の着地点をqueryにする

FootQueryは、各足の次のtouchdown locationとuncertaintyをproprioceptionから予測し、その位置が写っていた過去のdepth frameを検索します。79,537件のstair sampleを使った論文内分析では、現在frameの対象領域が見える割合は7.82%に対し、保持した履歴のいずれかで見える割合は67.89%でした。

ここでの新しさは、visual memoryを時間順に一様処理するのではなく、「次にどこへ触れるか」をqueryにする点です。ただし論文は、retrieval pathだけの効果をさらに切り分けるcontrol実験が必要だとも議論しています。

### UniPoint：sensorごとの画像ではなく3D pointで統合する

UniPointは、head-mountedの360° LiDARとtorso前後のdepth cameraをbase frameのpoint setへearly fusionし、voxel化後に各frame 80点、5 frameで400 tokenへ固定します。proprioceptionをqueryとするcross-attentionで、歩行に必要な点を選びます。

DR02での実機評価は、7種類・9設定を各20 trial実施し、70 cmのplatform、100 cmのgap、thin barrier、stepping stone、balance beamなどを対象にしています。actorは50 Hzで23 jointのposition targetを出し、onboard RK3588上のnetwork forwardは100回のONNX Runtime計測で平均1.5 msと報告されています。これらは同論文のhardware・前処理条件での値で、別機体へそのまま一般化できるbenchmarkではありません。

### PRIMO：controllerと推定器のdata分布を切り離す

PRIMOは、特定のlocomotion policyのrolloutだけでodometry estimatorを学ぶと、policy更新後のmotionを覆えない問題を扱います。約64時間のretargeted human motionをtrackingするsimulation rolloutからtraining dataを作り、physics・左右対称性のpriorを入れたestimatorを学習します。

実機は31 body DoFのAgiBot A3 Ultraで、入力には両脚12 jointと腰3 jointのposition／velocity、pelvis IMUを使います。cameraとLiDARを使わない自己運動推定の研究ですが、評価referenceにはLiDAR map localizationやmotion captureを用いています。つまり「推定時に外界sensorを使わない」ことと「ground truth作成にも使わない」ことは別です。

この3本は同じ方向を向いているようで、役割が異なります。

```text
PRIMO
  └─ robot自身がどう動いたかをproprioceptionから推定

FootQuery
  └─ 次に触れる場所を予測し、必要な過去のdepthを検索

UniPoint
  └─ 現在と直近の複数range sensorを共通の3D表現へ融合
```

最近の流れは、単に「cameraで地形を見る」ことではありません。**接触予定に必要な記憶を選ぶ、sensor数によらない共通表現を作る、controller更新に耐える自己状態推定を作る**という、controlとの接続部分へ焦点が移っています。

## 6. 実際に手を動かす順序

### Stage 1：X1の片脚を図解する

X1の最新公開directoryを使い、hipからfootまでのlink、joint axis、actuator、bearing、cable routeを1枚にまとめます。次に、CADで見える部品とURDFで見えるparameterを2列で対応づけます。

完了条件は、「このmotorを回すと、どのlinkを介してどのjointが動くか」「simulationでは何が省略されているか」を説明できることです。

### Stage 2：OpenLoongの1歩をtraceする

公式READMEが示す環境はUbuntu 22.04.4、g++ 11.4.0です。まずparameterを変えず、`walk_wbc`と`walk_mpc_wbc`の違いを観察します。

```bash
git clone https://github.com/loongOpen/OpenLoong-Dyn-Control.git
cd OpenLoong-Dyn-Control
mkdir build && cd build
cmake ..
make
./walk_mpc_wbc
```

logへbase pose、desired foot pose、support leg、MPCのcontact force、WBCのjoint acceleration、final torqueを出し、1歩の時系列を並べます。値を調整する前に、単位とcoordinate frameを記録します。

### Stage 3：Humanoid-Gymの観測・action・rewardを対応づける

Humanoid-GymのREADMEが固定しているPython 3.8、PyTorch 1.13.1、CUDA 11.7、Isaac Gym Preview 4は、論文実装を再現するための古いstackです。新規projectの一般的な推奨versionではありません。

```bash
python humanoid/scripts/train.py --task=humanoid_ppo \
  --run_name baseline --headless --num_envs 4096
python humanoid/scripts/play.py --task=humanoid_ppo \
  --run_name baseline
python humanoid/scripts/sim2sim.py \
  --load_model /path/to/exported/policies/policy.pt
```

最初の実験ではrewardを増やさず、次を記録します。

- commandに対するbase velocity error
- left／right footのcontact timing
- action、joint target、実joint positionの差
- torque／energy cost
- Isaac GymとMuJoCoでのtrajectory差
- randomizationを1項ずつ外したときの変化

G1／H1で現在の公式toolchainを使いたい場合は、Isaac Gym系の[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)と、Isaac Sim 5.1.0・Isaac Lab 2.3.0を対象とする[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)を別々に確認します。どちらも`Train → Play → Sim2Sim → Sim2Real`の順を明示しています。実機展開はsimulationの延長ではなく、emergency stop、吊り治具、可動範囲、通信断時の挙動を含む別の安全工程です。

### Stage 4：同じlogをmodel-based estimatorとlearned estimatorで見る

IMU、joint position／velocity、contact判定を保存し、まずInEKF系の予測・更新を可視化します。その後、DWLやPRIMOのようなlearned estimatorが何を追加で推定するかを比較します。

比較では平均誤差だけでなく、足滑り、着地impact、急旋回、controller変更後に誤差がどう増えるかを分けます。PRIMOの数値を再現するには、同論文のtraining corpus、real-robot protocol、ground truth条件まで揃える必要があります。

### Stage 5：terrain表現を比較する

同じdepth／point cloudから、次の3表現を作って可視化します。

1. robot周囲のheight sample
2. uncertainty付きelevation map
3. voxel化したpoint token

stairs、gap、thin vertical barrier、自己遮蔽した足場で、何が失われるかを比較します。最初からpolicyの成功率だけを見ると、perceptionの失敗とcontrolの失敗を区別できません。

### Stage 6：最新研究はablationから読む

FootQuery、UniPoint、PRIMOは、demo動画の見た目ではなく、次の比較を確認します。

- memoryを外すと何が落ちるか
- sensorを1種類遮蔽するとどうdegradeするか
- trainingに使ったpolicyとdeployment policyが変わるとどうなるか
- simulationで使えるprivileged informationが実機policyへ漏れていないか
- 実機trial数、terrain数、ground truth、失敗の定義は何か

この読み方をすると、「歩けた」という結果を、構造、推定、知覚、方策、低level controlのどこが支えたのか分解できます。

## 限界と注意点

- MIT Cheetah、ANYmal、Unitree A1の研究は、actuator、Sim-to-Real、並列学習、適応の基盤を理解する資料です。四足での成功が二足humanoidの性能を直接保証するわけではありません。
- 本稿は技術資料から進歩の条件を整理したもので、中国に固有の供給網、製造cost、投資、政策の寄与を比較・実証した産業分析ではありません。
- X1のCADが公開されていても、材料特性、公差、製造条件、firmware、低level制御の全情報が揃うわけではありません。
- G1の製品仕様と、G1を使うASAPやFootQueryの研究実装を混同してはいけません。
- OpenLoongは学習しやすい一体的なcodebaseですが、公開されているsimulationと実機controllerの全条件が同一とは限りません。
- Humanoid-Gymのzero-shot transferはXBot-S／XBot-Lでの研究結果です。任意のURDFを追加すれば同じ結果になるわけではありません。
- InEKFは接触・運動学の仮定が明確な一方、slipや衝撃で仮定が破れます。learned estimatorもtraining分布とsim-to-real gapから自由ではありません。
- ARMORは主に上半身のcollision avoidance、FootQueryとUniPointはterrain locomotion、PRIMOはodometryを扱います。目的の違う数値を横並びの性能rankingにはできません。
- 2026年9月の3論文はarXiv v1です。公開直後のため、査読、code公開、第三者再現、長期耐久の状況を継続して確認する必要があります。

## まとめ

二足歩行ロボットは、mechanism、controller、estimator、perceptionを別々に読むだけではつながりません。X1のCADでmotorからcontactまでの物理経路を見て、OpenLoongで明示的な力学計算を追い、Humanoid-Gymで同じ問題を観測・action・rewardへ写像すると、両者の共通部分と違いが見えます。

近年の急速な進歩は、RLだけの成果でも、motorだけの成果でもありません。接触に適したactuator、実機の応答を含むsimulation、大量並列学習、環境適応、そして学習しやすい機体設計が積み重なり、実機へ移すまでのiterationが短くなりました。中国系projectの公開model、toolchain、実機研究は、その技術stackが具体的なplatform上で結びついたものとして理解できます。

その上でInEKFからDWL／PRIMOへ進み、elevation mapからFootQuery／UniPointへ進むと、最近の研究が「もっと大きなnetwork」ではなく、**接触に必要なstateと外界情報を、いつ、どのsensorから、どの表現でcontrolへ渡すか**を改善していることが分かります。

## 参照

### 構造・機体資料

- AgiBot, [AgiBot X1 Product Design Guide](https://www.agibot.com.cn/DOCS/OS/X1-PDG).
- AgiBot, [agibot_x1_hardware](https://github.com/AgibotTech/agibot_x1_hardware).
- Unitree Robotics, [G1 Humanoid Robot](https://www.unitree.com/g1/).
- Victor Lutz et al., [Control of Humanoid Robots with Parallel Mechanisms using Differential Actuation Models](https://arxiv.org/abs/2503.22459), arXiv:2503.22459v2, 2025.
- Guglielmo Cervettini et al., [A Framework for Optimal Ankle Design of Humanoid Robots](https://arxiv.org/abs/2509.16469), arXiv:2509.16469, 2025.

### 制御・Sim-to-Real

- Patrick M. Wensing et al., [Proprioceptive Actuator Design in the MIT Cheetah: Impact Mitigation and High-Bandwidth Physical Interaction for Dynamic Legged Robots](https://dspace.mit.edu/entities/publication/8b15bcc8-c288-43aa-b923-69ce3c13b818), IEEE T-RO, 2017.
- Xue Bin Peng et al., [DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills](https://arxiv.org/abs/1804.02717), ACM TOG, 2018.
- Jemin Hwangbo et al., [Learning Agile and Dynamic Motor Skills for Legged Robots](https://arxiv.org/abs/1901.08652), Science Robotics, 2019.
- Ashish Kumar et al., [RMA: Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034), RSS, 2021.
- Nikita Rudin et al., [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978), CoRL, 2021.
- Qiayuan Liao et al., [Berkeley Humanoid: A Research Platform for Learning-based Control](https://arxiv.org/abs/2407.21781), arXiv:2407.21781, 2024.
- Humanoid Robot (Shanghai) Co., Ltd., [OpenLoong Dynamics Control](https://github.com/loongOpen/OpenLoong-Dyn-Control), 2024–.
- Xinyang Gu et al., [Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer](https://arxiv.org/abs/2404.05695), arXiv:2404.05695v2, 2024; [code](https://github.com/roboterax/humanoid-gym).
- Xinyang Gu et al., [Advancing Humanoid Locomotion: Mastering Challenging Terrains with Denoising World Model Learning](https://arxiv.org/abs/2408.14472), RSS 2024.
- Tairan He et al., [OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning](https://arxiv.org/abs/2406.08858), 2024.
- Tairan He et al., [ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills](https://arxiv.org/abs/2502.01143), 2025.
- Takara E. Truong et al., [BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion](https://arxiv.org/abs/2508.08241), 2025.
- Unitree Robotics, [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) and [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab).

### 状態推定・perception

- Ross Hartley et al., [Contact-Aided Invariant Extended Kalman Filtering for Robot State Estimation](https://arxiv.org/abs/1904.09251), 2019. RSS 2018 conference paperの拡張版。
- Péter Fankhauser et al., [Probabilistic Terrain Mapping for Mobile Robots with Uncertain Localization](https://www.research-collection.ethz.ch/items/563227f2-bb05-434b-8aef-1b001a9fdebc), IEEE RA-L, 2018.
- Daehwa Kim et al., [ARMOR: Egocentric Perception for Humanoid Robot Collision Avoidance and Motion Planning](https://arxiv.org/abs/2412.00396), 2024.
- Tao Dong et al., [FootQuery: Future-Touchdown-Guided Retrieval from Depth History for Perceptive Humanoid Locomotion](https://arxiv.org/abs/2609.21447), arXiv:2609.21447v1, 2026-09-18.
- Sicen Li et al., [UniPoint: Unified Point-Level Sensor Fusion for Humanoid Locomotion Across Challenging Terrains](https://arxiv.org/abs/2609.23666), arXiv:2609.23666v1, 2026-09-20.
- Xu Han et al., [PRIMO: Prior-Informed Odometry from Human-Motion Tracking for Humanoid Robots](https://arxiv.org/abs/2609.23610), arXiv:2609.23610v1, 2026-09-20; [code](https://github.com/Agibot-Spatial-Intelligence/PRIMO).
