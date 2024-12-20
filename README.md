
# hottoSNS-BERT：大規模日本語SNSコーパスによる文分散表現モデル

## 概要
* 大規模日本語SNSコーパスによる文分散表現モデル（以下，大規模SNSコーパス）から作成したbertによる文分散表現を構築した
* 本文分散表現モデル(以下，hottoSNS-BERT）は下記登録フォームから登録した方のみに配布する
  * [利用規約](#利用規約)は本README.mdの末尾に記載されている．またLICENSE.mdにも同じ内容が記載されている．

[登録フォーム](https://forms.office.com/Pages/ResponsePage.aspx?id=Zpu1Ffmdi02AfxgH3uo25PxaMnBWkvJLsXoQLeuzhoBUNU0zN01BR1VFNk9RSEUxWVRNSzAyWThZNSQlQCN0PWcu)


<img src="https://github.com/hottolink/hottoSNS-bert/blob/master/images/QR_hottoSNS-bert.png" width="128">

[言語資源を利用した発表の登録フォーム](https://forms.office.com/r/EQizJpFBJg)
※PDFファイルの提出は不要です。

### 引用について
* 本モデルに関する論文発表は未公開です．引用される方は，以下のbibtexをご利用ください．
```
@article{hottoSNS-bert,
  author = Sakaki， Takeshi and Mizuki, Sakae and Gunji, Naoyuki},
  title = {BERT Pre-trained model Trained on Large-scale Japanese Social Media Corpus},
  year = {2019},
  howpublished = {\url{https://github.com/hottolink/hottoSNS-bert}}
}
```

<!-- TOC -->

- [hottoSNS-BERT：大規模日本語SNSコーパスによる文分散表現モデル](#hottosns-bert大規模日本語snsコーパスによる文分散表現モデル)
    - [概要](#概要)
    	- [引用について](#引用について)
    - [配布リソースに関する説明](#配布リソースに関する説明)
        - [大規模日本語SNSコーパス](#大規模日本語snsコーパス)
        - [ファイル構成](#ファイル構成)
        - [利用方法](#利用方法)
            - [実行確認環境](#実行確認環境)
            - [付属評価コードの利用準備](#付属評価コードの利用準備)
            - [付属評価コードの利用方法](#付属評価コードの利用方法)
            - [モデルの読み込み方法](#モデルの読み込み方法)
    - [配布リソースの構築手順](#配布リソースの構築手順)
        - [コーパス・単語分散表現の構築方法](#コーパス・単語分散表現の構築方法)
            - [平文コーパスの収集・構築](#平文コーパスの収集・構築)
            - [前処理](#前処理)
            - [分かち書きコーパスの構築](#分かち書きコーパスの構築)
            - [後処理](#後処理)
            - [既存モデルとの前処理・分かち書きの比較](#既存モデルとの前処理・分かち書きの比較)
            - [統計量](#統計量)
        - [文分散表現の学習](#文分散表現の学習)
            - [pre-training](#pre-training)
            - [neuralnet architectureの比較](#neuralnet-architectureの比較)
            - [学習環境](#学習環境)
    - [配布リソースの性能評価](#配布リソースの性能評価)
        - [評価用データセット](#評価用データセット)
        - [downstream task: fine-tuning](#downstream-task-fine-tuning)
        - [実験結果](#実験結果)
    - [pytorch-transformersからの利用](#pytorch-transformersからの利用)
    - [利用規約](#利用規約)

<!-- /TOC -->

## 配布リソースに関する説明
### 大規模日本語SNSコーパス
* BERT自家版を学習するために，大規模な日本語ツイートのコーパスを構築した
* 収集文の多様性が大きくなるように工夫している．具体的には，bot投稿・リツイートの除外，重複ツイート文の除外といった工夫を施している
* 構築されたコーパスのツイート数は8,500万である
    * 本家BERTが用いたコーパス(=En Wikipedia + BookCorpus)と比較すると，35%程度の大きさである

### ファイル構成

| モデル |  ファイル名 |
|-----------|----------------------|
| hottoSNS-BERTモデル  |bert_config.json|
| | graph.pbtxt |
| | model.ckpt-1000000.meta |
| | model.ckpt-1000000.index |
| | model.ckpt-1000000.data-00000-of-00001 |
| sentencepieceモデル | tokenizer_spm_32K.model  |
| | tokenizer_spm_32K.vocab.to.bert |
| |  tokenizer_spm_32K.vocab  



### 利用方法
#### 実行確認環境
* Python 3.6.6
* tensorflow==1.11.0
* sentencepiece==0.1.8


#### 付属評価コードの利用準備
`./hottoSNS-bert/evaluation_dataset/twitter_sentiment/`以下に[Twitter日本語評判分析データセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/)からツイートを再現し，BERTモデル評価用に加工したデータが必要．詳細は[hottoSNS-bert/evaluation_dataset/twitter_sentiment/](https://github.com/hottolink/hottoSNS-bert/tree/master/evaluation_dataset/twitter_sentiment)参照．
#### 付属評価コードの利用方法
```
# リポジトリのClone
git clone https://github.com/hottolink/hottoSNS-bert.git
cd hottoSNS-bert

# 取得した各BERTモデルファイルを `trained_model/` 以下に配置
cp -r [hottoSNS-bert dir]/* ./trained_model/masked_lm_only_L-12_H-768_A-12/ 
cp -r [日本語wikipedia model dir]/* ./trained_model/wikipedia_ja_L-12_H-768_A-12/
cp -r [Multilingual model dir]/* ./trained_model/multi_cased_L-12_H-768_A-12/


# 評価環境の構築・評価実行
# ※テキストファイルから分散表現を読み込むため、実行に時間がかかります。
sh setup.sh
cd src
sh run_classifier.sh 

```

#### モデルの読み込み方法
サンプルコードを参照してください。

## 配布リソースの構築手順
### コーパス・単語分散表現の構築方法

####  平文コーパスの収集・構築
* 期間：2017年〜2018年に投稿されたツイートから一部を抽出
* 投稿クライアント：人間用のクライアントのみ
    * 実質的に，botによる投稿を除外
* ツイート種別：オーガニックおよびメンション

#### 前処理
* 文字フィルタ：ReTweet記号(RT)・省略記号(...)の除外
* 正規化：NFKC正規化，小文字化
* 特殊トークン化：mention, url
* 除外：正規化された本文が重複するツイートを削除


* サンプルデータは以下の通り

```
ゆめさんが、ファボしてくるあたり、世代だなって思いました(   ̇- ̇  )笑
<mention> 90秒に250円かけるかどうかは、まぁ個人の自由だしね()
<mention> それでは聞いてください  rainy <url>
```

#### 分かち書きコーパスの構築
* sentencepieceを採用
* 設定は以下の通り


|argument|value|
|--|--|
|vocab_size|32,000 |
|character_coverage|0.9995|
|model_type|unigram|
|add_dummy_prefix|FALSE|
|user_defined_symbols|\<url\>,\<mention\>|
|control_symbols|[CLS],[SEP],[MASK]|
|input_sentence_size|20,000,000 |
|pad_id|0|
|unk_id|1|
|bos_id|-1|
|eos_id|-1|

* サンプルデータは以下の通り

```
ゆめ さんが 、 ファボ してくる あたり 、 世代 だ なって思いました ( ▁̇ - ▁̇ ▁ ) 笑

<mention> ▁ 90 秒 に 250 円 かける かどうかは 、 まぁ 個人の 自由 だしね ()

<mention> ▁ それでは 聞いてください ▁ rain y ▁ <url>
```

#### 後処理
* 短すぎる・少なすぎるツイートを除外
* 具体的には，以下に示すしきい値を下回るユーザおよびツイートを除外

|limitation|value|
|--|--|
|トークン長さ|5|
|ユーザあたりのツイート数|5|


#### 既存モデルとの前処理・分かち書きの比較
| 前処理                         |                         |                                | 分かち書き       |         |               |          |
|-------------------------------------|-------------------------|--------------------------------|----------------|---------|---------------|----------|
| モデル名                            | 文字正規化 | 特殊トークン化 |  小文字化 | 単語数 | 分かち書き     | 言語 |
| BERT MultiLingual                   | None                    | no                             | yes            | 119,547 | WordPiece     | multi※ |
| BERT 日本語Wikipedia                | NFKC                    | no                             | no             | 32,000  | SentencePiece | ja       |
| hottoSNS-BERT | NFKC                    | yes                         | no             | 32,000  | SentencePiece | ja       |






#### 統計量
構築されたコーパスの統計量は，以下の通り

*  コーパス全体

|metric|value|
|--|--|
|n_user|1,872,623 |
|n_post|85,925,384 |
|n_token|1,441,078,317 |

*  トークン数・1ユーザあたりの投稿数

|metric|n_token|n_post.per.user|
|--|--|--|
|min|5|5|
|mean|16.77|45.89|
|std|13.06|14.83|
|q(0.99)|64|76|
|max|227|781|


### 文分散表現の学習
#### pre-training
next sentence predictionはツイートに適用することが難しいため、masked language model のみを適用する.
また、事前学習のタスク設定について，各サンプルのtoken数を最大64に制限した．

#### neuralnet architectureの比較

| neuralnet architecture                    |         |         |             |         | pre-training  |             |         |           |
|-------------------------------------------|---------|---------|-------------|---------|---------------|-------------|---------|-----------|
| モデル名                                  | n_dim_e | n_dim_h | n_attn_head | n_layer | max_pos_embed | max_seq_len | n_batch | n_step    |
| BERT MultiLingual                         | 768     | 3072    | 12          | 12      | 512           | 512         | 256     | 1,000,000 |
| BERT 日本語Wikipedia                      | 768     | 3072    | 12          | 12      | 512           | 512      | 256  | 1,400,000 |
| hottoSNS-BERT        | 768     | 3072    | 12          | 12      | 512           | 64          | 512     | 1,000,000 |



#### 学習環境
* Google Computing Platform の Cloud TPU を使用．詳細は以下の通り
* neuralnet framework は TensorFlow 1.12.0 を使用
* 詳細は以下の通り
    * CPU：n1-standard-2（vCPU x 2、メモリ 7.5 GB）
    * ストレージ：Cloud Storage
    * TPU：v2-8



## 配布リソースの性能評価
### 評価用データセット
* ツイート評判分析をdownstreamタスクとして、構築したBERTモデルの評価を行う．
* 評判分析タスクは，2種類のデータセットを用いて評価する
	1. [Twitter日本語評判分析データセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/)[芥子+, 2017]
		* サンプル数：161K
	2. 内製のデータセット；Twitter大規模トピックコーパス
		* サンプル数：12K

* 統計量は以下の通り

|データセット名|トピック|positive|negative|neutral|total|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Twitter大規模トピックコーパス|指定なし|4,162 |3,031 |4,807 |12,000 |
|Twitter日本語評判分析データセット|家電・携帯端末|10,249 |15,920 |134,928 |161,097 |

### downstream task: fine-tuning
* downstream task の詳細は，以下の通りである
	* task type：日本語ツイートの評判分析；Positive/Negative/Neutral の3値分類
	* task dataset
		1. Twitter日本語評判分析データセット[芥子+, 2017]
		2. 内製のデータセット
	* methodology
		* task dataset を train:test = 9:1 に分割
		* hyper-parameter は，BERT論文[Devlin+, 2018] に準拠
		* 学習および評価を7回繰り返して，平均値を報告
	* evaluation metric：accuracy および macro F-value

```
芥子 育雄, 鈴木 優, 吉野 幸一郎, グラム ニュービッグ, 大原 一人, 向井 理朗, 中村 哲: 「単語意味ベクトル辞書を用いたTwitterからの日本語評判情報抽出」， 電子情報通信学会論文誌, Vol.J100-D, No.4, pp.530-543, 2017.4.
Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", arXiv preprint arXiv:1810.04805,2018
```


### 実験結果
実験結果は以下の通り．

|       | Twitter大規模カテゴリコーパス          | | Twitter日本語評判分析データセット          |         |
|-------------------------------------|----------|-----------------------------------|----------|---------|
| モデル名                            | accuracy | F-value                           | accuracy | F-value |
| BERT MultiLingual                   | 0.7019   | 0.7011                            | 0.8776   | 0.7225  |
| BERT 日本語Wikipedia                | 0.7237   | 0.7239                            | 0.8790   | 0.7359  |
| hottoSNS-BERT| 0.7387   | 0.7396                            | 0.8880   | 0.7503  |

下記のような結果であると言える．

* Twitter評判分析タスクに対する性能は以下のようになった。
	* Multilingual < 日本語Wikipedia < 日本語SNS 
* Multilingual < 日本語Wikipediaであることから日本語を対象としたdownstreamタスクでは，日本語(の語彙)に特化した分かち書き方法および，日本語のコーパスを用いた事前学習の方が適していると考えられる
* 日本語Wikipedia < 日本語SNS であることから，Twitterを対象としたdownstreamタスクでは、日本語Wikipediaよりもドメインに特化した大規模日本語SNSコーパスで学習したBERTモデルの方が良い性能が得られると考えられる

## pytorch-transformersからの利用
* pytorch 1.8.1+cu102, tensorflow 2.4.1での動作例
1. hottoSNS-bertの読み込み
    ```
    # pytorch-transformers, tensorflowの読み込み
    import os, sys
    from transformers import BertForPreTraining, BertTokenizer
    import tensorflow as tf

    # hottoSNS-bertの読み込み
    sys.path.append("./hottoSNS-bert/src/")
    import tokenization
    from preprocess import normalizer
    ```
1. 必要なファイルの指定
    ```
    bert_model_dir = "./hottoSNS-bert/trained_model/masked_lm_only_L-12_H-768_A-12/"
    config_file = os.path.join(bert_model_dir, "bert_config.json")
    vocab_file = os.path.join(bert_model_dir, "tokenizer_spm_32K.vocab.to.bert")
    sp_model_file = os.path.join(bert_model_dir, "tokenizer_spm_32K.model")
    bert_model_file = os.path.join(bert_model_dir, "model.ckpt-1000000.index")
    ```
1. tokenizerのインスタンス化
    ```
    tokenizer = tokenization.JapaneseTweetTokenizer(
        vocab_file=vocab_file,
        model_file=model_file,
        normalizer= normalizer.twitter_normalizer_for_bert_encoder,
        do_lower_case=False)
    ```
1. tokenizeの実行例
    ```
    # 例文
    text = '@test ゆめさんが、ファボしてくるあたり、世代だなって思いました(   ̇- ̇  )笑 http://test.com/test.html'
    # tokenize
    words = tokenizer.tokenize(text)
    print(words)
    # ['<mention>', '▁', 'ゆめ', 'さんが', '、', 'ファボ', 'してくる', 'あたり', '、', '世代', 'だ', 'なって思いました', '(', '▁̇', '-', '▁̇', '▁', ')', '笑', '▁', '<url>']
    # idへ変換
    tokenizer.convert_tokens_to_ids(["[CLS]"]+words+["[SEP]"])
    # [2, 6, 7, 6372, 348, 8, 5249, 2135, 1438, 8, 3785, 63, 28146, 12, 112, 93, 112, 7, 13, 31, 7, 5, 3]

    ```

1. pretrainモデルの読み込み
    ```
    model = BertForPreTraining.from_pretrained(bert_model_file, from_tf=True, config=config_file)
    ```


## 利用規約
同一の内容をLICENSE.mdに記述
```
第１条（定義）
本契約で用いられる用語の意味は、以下に定めるとおりとする。
（１）「本規約」とは、本利用規約をいう。
（２）「甲」とは、 株式会社ホットリンク（以下「甲」という）をいう。
（３）「乙」とは、本規約に同意し、甲の承認を得て、甲が配布する文分散表現データを利用する個人をいう。
（４）「本データ」とは、甲が作成した文分散表現データおよびそれに付随する全部をいう。


第２条（利用許諾）
甲は、乙が本規約に従って本データを利用することを非独占的に許諾する。なお、甲及び乙は、本規約に明示的に定める以外に、乙に本データに関していかなる権利も付与するものではないことを確認する。


第４条（許諾の条件）
甲が乙に本データの利用を許諾する条件は、以下の通りとする。 
（１）利用目的： 日本語に関する学術研究・産業研究（以下「本研究」という）を遂行するため。
（２）利用の範囲： 乙及び乙が所属する研究グループ
（３）利用方法： 本研究のために本データを乙が管理するコンピューター端末またはサーバーに複製し、本データを分析・研究しデータベース等に保存した解析データ（以下「本解析データ」という）を得る。


第５条（利用申込）
１．乙は、甲が指定するウェブ上の入力フォーム（以下、入力フォーム）を通じて、乙の名前や所属、連絡先等、甲が指定する項目を甲に送信し、本データの利用について甲の承認を得るものとする。 なお、甲が承認しなかった場合、甲はその理由を開示する義務を負わない。
２．前項に基づき甲に申告した内容に変更が生じる場合、乙は遅滞なくこれを甲に報告し、改めて甲の承認を得るものとする。
３．乙が入力フォームを送信した時点で、乙は本規約に同意したものとみなされる。

第６条（禁止事項）
乙は、本データの利用にあたり、以下に定める行為をしてはならない。 
（１）本データ及びその複製物（それらを復元できるデータを含む）を譲渡、貸与、販売すること。また、書面による甲の事前許諾なくこれらを配布、公衆送信、刊行物に転載するなど前項に定める範囲を超えて利用し、甲または第三者の権利を侵害すること。  
（２）本データを用いて甲又は第三者の名誉を毀損し、あるいはプライバシーを侵害するなどの権利侵害を行うこと。
（３）乙及び乙が所属する研究グループ以外の第三者に本データを利用させること。
（４）本規約で明示的に許諾された目的及び手段以外にデータを利用 すること。

第７条（対価） 
本規約に基づく本データの利用許諾の対価は発生しない。

第８条（公表）
１．乙は、学術研究の目的に限り、本データを使用して得られた研究成果や知見を公表することができる。これらの公表には、本解析データや処理プログラムの公表を含む。
２．乙は、公表にあたっては、本データをもとにした成果であることを明記し、成果の公表の前にその概要を書面やメール等で甲に報告する。
３．乙は、論文発表の際も、本データを利用した旨を明記し、提出先の学会、発表年月日を所定のフォームから甲に提出するものとする。



第９条（乙の責任）
１．乙は、本データをダウンロードする為に必要な通信機器やソフトウェア、通信回線等の全てを乙の責任と費用で準備し、操作、接続等をする。
２．乙は、本データを本研究の遂行のみに使用する。
３．乙は、本データが漏洩しないよう善良な管理者の注意義務をもって管理し、乙のコンピューター端末等に適切な対策を施すものとする。
４．乙が、本研究を乙が所属するグループのメンバーと共同で遂行する場合、乙は、本規約の内容を当該グループの他のメンバーに遵守させるものとし、万一、当該他のメンバーが本規約に違反し甲又は第三者に損害を与えた場合は、乙はこれを自らの行為として連帯して責任を負うものとする。
５．甲が必要と判断する場合、乙に対して、本データの利用状況の開示を求めることができるものとし、乙はこれに応じなければならない。


第１０条（知的財産権の帰属）
甲及び乙は、本データに関する一切の知的財産権、本データの利用に関連して甲が提供する書類やデータ等に関する全ての知的財産権について、甲に帰属することを確認する。ただし、本データ作成の素材となった各文書の著作権は正当な権利を有する第三者に帰属する。

第１１条（非保証等）
１．甲は、本データが、第三者の著作権、特許権、その他の無体財産権、営業秘密、ノウハウその他の権利を侵害していないこと、法令に違反していないこと、本データ作成に利用したアルゴリズムに誤り、エラー、バグがないことについて一切保証せず、また、それらの信頼性、正確性、速報性、完全性、及び有効性、特定目的への適合性について一切保証しないものとし、瑕疵担保責任も負わない。
２．本データに関し万一、第三者から知的財産権侵害等の主張がなされた場合には、乙はただちに甲に対しその旨を通知し、甲に対する情報提供等、当該紛争の解決に最大限協力するものとする。


第１２条（違反時の措置） 
１．甲は、乙が次の各号の一つにでも該当した場合、甲は乙に対して本データの利用を差止めることができる。
（１）本規約に違反した場合
（２）法令に違反した場合
（３）虚偽の申告等の不正を行った場合
（４）信頼関係を破壊するような行為を行った場合
（５）その他甲が不適当と認めた場合
２．前項の規定は甲から乙に対する損害賠償請求を妨げるものではない。 
３．第１項に基づき、甲が乙に対して本データの利用の差し止めを求めた場合、乙は、乙が管理する設備から、本データ、本解析データ及びその複製物の一切を消去するものとする。

第１３条（甲の事情による利用許諾の取り消し）
１．甲は、その理由の如何を問わず、なんらの催告なしに、本データの利用許諾を停止することができるものとする。その際は、第１５条に基づき、乙は速やかに本データおよびその複製物の一切を消去または破棄する。 
２．前項の破棄、消去の対象に本解析データは含まない。


第１４条（利用期間）
１．乙による本データの利用可能期間は、第５条にもとづく甲の承認日より１年間とする。
２．乙が１年間を超えて本データの利用継続を希望する場合、第５条に基づく方法で再度利用申請を行うこととする。


第１５条（本契約終了後の措置等）
１．理由の如何を問わず、第１４条に定める利用期間が終了したとき、もしくは、本データの利用許諾が取り消しとなった場合、乙は本データおよびその複製物の一切を消去または破棄する。  
２．前項の破棄、消去の対象に本解析データは含まない。ただし、乙は、本解析データから本データを復元して再利用することはできないものとする。
３．第１０条、第１１条、第１５条から第１９条は、本契約の終了後も有効に存続する。

第１６条（権利義務譲渡の禁止）
乙は、相手方の書面による事前の承諾なき限り、本契約上の地位及び本契約から生じる権利義務を第三者に譲渡又は担保に供してはならない。

第１７条 （個人情報等の保護および法令遵守）
１．甲が取得した乙の個人情報は、別途定める甲２のプライバシーポリシーに従って取り扱われる。
２．甲は、サーバー設備の故障その他のトラブル等に対処するため、乙の個人情報を他のサーバーに複写することがある。

第１８条（準拠法）
本契約の準拠法は、日本法とする。

第１９条（管轄裁判所）
本契約に起因し又は関連して生じた一切の紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とする。

第２０条（協　議）
本契約に定めのない事項及び疑義を生じた事項は、甲乙誠意をもって協議し、円満にその解決にあたる。

第２１条（本規約の効力）
本規約は、本データの利用の関する一切について適用される。なお、本規約は随時変更されることがあるが、変更後の規約は特別に定める場合を除き、ウェブ上で表示された時点から効力を生じるものとする。
```
