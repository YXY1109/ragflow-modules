# TABLEFORMER: Robust Transformer Modeling for Table-Text Encoding

Jingfeng Yang Aditya Gupta† Shyam Upadhyay†Luheng He Rahul Goel† Shachi Paul †?Georgia Institute of Technology†Google Assistantjingfengyangpku@gmail.comtableformer@google.com

# Abstract

Understanding tables is an important aspect ofnatural language understanding. Existing mod-els for table understanding require lineariza-tion of the table structure, where row or col-umn order is encoded as an unwanted bias.Such spurious biases make the model vulner-able to row and column order perturbations.Additionally, prior work has not thoroughlymodeled the table structures or table-text align-ments, hindering the table-text understandingability. In this work, we propose a robust andstructurally aware table-text encoding architec-ture TABLEFORMER, where tabular structuralbiases are incorporated completely throughlearnable attention biases. TABLEFORMER is(1) strictly invariant to row and column or-ders, and, (2) could understand tables betterdue to its tabular inductive biases. Our eval-uations showed that TABLEFORMER outper-forms strong baselines in all settings on SQA,WTQ and TABFACT table reasoning datasets,and achieves state-of-the-art performance onSQA, especially when facing answer-invariantrow and column order perturbations $6 \%$ im-provement over the best baseline), because pre-vious SOTA models’ performance drops by$4 \% - 6 \%$ when facing such perturbations whileTABLEFORMER is not affected.1

# 1 Introduction

Recently, semi-structured data (e.g. variable lengthtables without a fixed data schema) has attractedmore attention because of its ubiquitous presenceon the web. On a wide range of various table rea-soning tasks, Transformer based architecture alongwith pretraining has shown to perform well (Eisen-schlos et al., 2021; Liu et al., 2021).

In a nutshell, prior work used the Transformerarchitecture in a BERT like fashion by serializing

<table><tr><td>Title</td><td>Length</td></tr><tr><td>Screwed Up Ghetto Queen</td><td>5:02 5:00</td></tr></table>

Question: Of all song lengths, which one is the longest?Gold Answer: 5:02TAPAS: 5:00TAPAS after row order perturbation: 5:02TABLEFORMER: 5:02

(a) TAPAS predicts incorrect answer based on the original table,while it gives the correct answer if the first row is moved tothe end of the table.

<table><tr><td>Nation</td><td>Gold</td><td>Silver</td><td>Bronze</td></tr><tr><td>Great Britain</td><td>2</td><td>1</td><td>2</td></tr><tr><td>Spain</td><td>1</td><td>2</td><td>0</td></tr><tr><td>Ukraine</td><td>0</td><td>2</td><td>0</td></tr></table>

Question: Which nation received 2 silver medals?Gold Answer: Spain, UkraineTAPAS: SpainTABLEFORMER: Spain, UkraineTABLEFORMER w/o a proposed structural bias: Spain

(b) TAPAS gives incomplete answer due to its limited cellgrounding ability.

Figure 1: Examples showing the limitations of exist-ing models (a) vulnerable to perturbations, and (b) lack-ing structural biases. In contrast, our proposed TABLE-FORMER predicts correct answers for both questions.

tables or rows into word sequences (Yu et al., 2020;Liu et al., 2021), where original position ids areused as positional information. Due to the usageof row/column ids and global position ids, priorstrategies to linearize table structures introducedspurious row and column order biases (Herzig et al.,2020; Eisenschlos et al., 2020, 2021; Zhang et al.,2020; Yin et al., 2020). Therefore, those models arevulnerable to row or column order perturbations.But, ideally, the model should make consistent pre-dictions regardless of the row or column orderingfor all practical purposes. For instance, in Figure 1,the predicted answer of TAPAS model (Herzig et al.,2020) for Question (a) “Of all song lengths, whichone is the longest?” based on the original table is$5 . 0 0 ^ { \prime }$ , which is incorrect. However, if the first rowis adjusted to the end of the table during inference,the model gives the correct length $" 5 . 0 2 "$ as an-swer. This probing example shows that the modelbeing aware of row order information is inclinedto select length values to the end of the table dueto spurious training data bias. In our experimentson the SQA dataset, TAPAS models exhibit a $4 \%$ -$6 \%$ (Section 5.2) absolute performance drop whenfacing such answer-invariant perturbations.

Besides, most prior work (Chen et al., 2020; Yinet al., 2020) did not incorporate enough structuralbiases to models to address the limitation of sequen-tial Transformer architecture, while others induc-tive biases which are either too strict (Zhang et al.,2020; Eisenschlos et al., 2021) or computationallyexpensive (Yin et al., 2020).

To this end, we propose TABLEFORMER, aTransformer architecture that is robust to row andcolumn order perturbations, by incorporating struc-tural biases more naturally. TABLEFORMER re-lies on 13 types of task-independent table $$ textattention biases that respect the table structure andtable-text relations. For Question (a) in Figure 1,TABLEFORMER could predict the correct answerregardless of perturbation, because the model couldidentify the same row information with our “samerow” bias, avoiding spurious biases introduced byrow and global positional embeddings. For Ques-tion (b), TAPAS predicted only partially correctanswer, while TABLEFORMER could correctly pre-dict “Spain, Ukraine” as answers. That’s becauseour “cell to sentence” bias could help table cellsground to the paired sentence. Detailed attentionbias types are discussed in Section 5.2.

Experiments on 3 table reasoning datasets showthat TABLEFORMER consistently outperforms orig-inal TAPAS in all pretraining and intermediatepretraining settings with fewer parameters. Also,TABLEFORMER’s invariance to row and columnperturbations, leads to even larger improvementover those strong baselines when tested on pertur-bations. Our contributions are as follows:

• We identified the limitation of current table-text encoding models when facing row or col-umn perturbation.• We propose TABLEFORMER, which is guaran-teed to be invariant to row and column orderperturbations, unlike current models.• TABLEFORMER encodes table-text structuresbetter, leading to SoTA performance on SQAdataset, and ablation studies show the effec-tiveness of the introduced inductive biases.

# 2 Preliminaries: TAPAS for TableEncoding

In this section, we discuss TAPAS which servesas the backbone of the recent state-of-the-art table-text encoding architectures. TAPAS (Herzig et al.,2020) uses Transformer architecture in a BERTlike fashion to pretrain and finetune on tabulardata for table-text understanding tasks. This isachieved by using linearized table and texts formasked language model pre-training. In the fine-tuning stage, texts in the linearized table and textpairs are queries or statements in table QA or table-text entailment tasks, respectively.

Specifically, TAPAS uses the tokenized and flat-tened text and table as input, separated by [SEP]token, and prefixed by [CLS]. Besides token, seg-ment, and global positional embedding introducedin BERT (Devlin et al., 2019), it also uses rank em-bedding for better numerical understanding. More-over, it uses column and row embedding to encodetable structures.

Concretely, for any table-text linearized se-quence $\textit { S } = \ \{ v _ { 1 } , v _ { 2 } , \cdot \cdot \cdot , v _ { n } \}$ , where $n$ is thelength of table-text sequence, the input to TAPASis summation of embedding of the following:

token ids $\left( W \right) = \{ w _ { v _ { 1 } } , w _ { v _ { 2 } } , \cdot \cdot \cdot , w _ { v _ { n } } \}$positional ids $( B ) = \{ b _ { 1 } , b _ { 2 } , \cdot \cdot \cdot , b _ { n } \}$segment ids $( G ) = \{ g _ { s e g _ { 1 } } , g _ { s e g _ { 2 } } , \cdot \cdot \cdot , g _ { s e g _ { n } } \}$column ids $( C ) = \{ c _ { c o l _ { 1 } } , c _ { c o l _ { 2 } } , \cdot \cdot \cdot , c _ { c o l _ { n } } \}$row ids $( R ) = \{ r _ { r o w _ { 1 } } , r _ { r o w _ { 2 } } , \cdot \cdot \cdot , r _ { r o w _ { n } } \}$rank ids $( Z ) = \{ z _ { r a n k _ { 1 } } , z _ { r a n k _ { 2 } } , \cdot \cdot \cdot , z _ { r a n k _ { n } } \}$

where $s e g _ { i }$ , $c o l _ { i }$ , $r o w _ { i }$ , $r a n k _ { i }$ correspond tothe segment, column, row, and rank id for the ithtoken, respectively.

As for the model, TAPAS uses BERT’s self-attention architecture (Vaswani et al., 2017) off-the-shelf. Each Transformer layer includes a multi-head self-attention sub-layer, where each tokenattends to all the tokens. Let the layer input$H = [ h _ { 1 } , h _ { 2 } , \cdot \cdot \cdot , h _ { n } ] ^ { \top } \in \mathbb { R } ^ { n \times d }$ corresponding to$S$ , where $d$ is the hidden dimension, and $h _ { i } \in \mathbb { R } ^ { d \times 1 }$is the hidden representation at position $i$ . Fora single-head self-attention sub-layer, the input$H$ is projected by three matrices $\bar { W } ^ { Q } \in \mathbb { R } ^ { d \times \bar { d } _ { K } }$$W ^ { K } \in \mathbb { R } ^ { d \times d _ { K } }$ , and $W ^ { V } \in \mathbb { R } ^ { d \times d _ { V } }$ to the corre-sponding representations $Q , K$ , and $V$ :

![](http://127.0.0.1:9000/mineru/images/216f92e813522ce05fcbc2fc99f0aefbb781c5d71042fffbf7abccef3e20fc64.jpg)

![](http://127.0.0.1:9000/mineru/images/44a1b90e659bf679fecb1be2fa3b8d3b4eddf3f6185bc5b45bb01204cf2e8f5c.jpg)  
Figure 2: TABLEFORMER input and attention biases in the self attention module. This example corresponds totable (a) in Figure 1 and its paired question “query”. Different colors in the attention bias matrix denote differenttypes of task independent biases derived based on the table structure and the associated text.

$$
Q = H W ^ { Q } , \quad V = H W ^ { V } , \quad K = H W ^ { K }
$$

Then, the output of this single-head self-attention sub-layer is calculated as:

$$
\mathrm { A t t n } ( H ) = \mathrm { s o f t m a x } ( \frac { Q K ^ { \top } } { \sqrt { d _ { K } } } ) V
$$

# 3 TABLEFORMER: Robust StructuralTable Encoding

As shown in Figure 2, TABLEFORMER encodes thegeneral table structure along with the associatedtext by introducing task-independent relative atten-tion biases for table-text encoding to facilitate thefollowing: (a) structural inductive bias for bettertable understanding and table-text alignment, (b)robustness to table row/column perturbation.

Input of TABLEFORMER. TABLEFORMERuses the same token embeddings $W$ , segmentembeddings $G$ , and rank embeddings $Z$ as TAPAS.However, we make 2 major modifications:

1) No row or column ids. We do not use row em-beddings $R$ or column embeddings $C$ to avoid anypotential spurious row and column order biases.

2) Per cell positional ids. To further remove anyinter-cell order information, global positional em-beddings $B$ are replaced by per cell positional em-beddings $P = \{ p _ { p o s _ { 1 } } , p _ { p o s _ { 2 } } , \cdot \cdot \cdot , p _ { p o s _ { n } } \}$ , wherewe follow Eisenschlos et al. (2021) to reset theindex of positional embeddings at the beginningof each cell, and $p o s _ { i }$ correspond to the per cellpositional id for the ith token.

Positional Encoding in TABLEFORMER. Notethat the Transformer model either needs to spec-ify different positions in the input (i.e. absolutepositional encoding of Vaswani et al. (2017)) orencode the positional dependency in the layers (i.e.relative positional encoding of Shaw et al. (2018)).

TABLEFORMER does not consume any sort ofcolumn and row order information in the input. Themain intuition is that, for cells in the table, the onlyuseful positional information is whether two cellsare in the same row or column and the columnheader of each cell, instead of the absolute orderof the row and column containing them. Thus, in-spired by relative positional encoding (Shaw et al.,2018) and graph encoding (Ying et al., 2021), wecapture this with a same column/row relation asone kind of relative position between two linearizedtokens. Similarly, we uses 12 such table-text struc-ture relevant relations (including same cell, cellto header and so on) and one extra type represent-ing all other relations not explicitly defined. Allof them are introduced in the form of learnableattention bias scalars.

Formally, we consider a function $\phi ( v _ { i } , v _ { j } ) : V \times$$V  \mathbb { N }$ , which measures the relation between $v _ { i }$and $v _ { j }$ in the sequence $( v _ { i } , v _ { j } \in S )$ . The function $\phi$can be defined by any relations between the tokensin the table-text pair.

Attention Biases in TABLEFORMER. In ourwork, $\phi ( v _ { i } , v _ { j } )$ is chosen from 13 bias types, cor-responding to 13 table-text structural biases. Theattention biases are applicable to any table-text pairand can be used for any downstream task:

• “same row” identifies the same row infor-mation without ordered row id embedding orglobal positional embedding, which help themodel to be invariant to row perturbations,“same column”, “header to column cell”, and“cell to column header” incorporates the samecolumn information without ordered columnid embedding,• “cell to column header” makes each cellaware of its column header without repeatedcolumn header as features,• “header to sentence” and “cell to sentence”help column grounding and cell grounding ofthe paired text,• “sentence to header”, “sentence to cell”, and“sentence to sentence” helps to understand thesentence with the table as context,• “header to same header” and “header toother header” for better understanding of ta-ble schema, and “same cell bias” for cell con-tent understanding.

Note that, each cell can still attend to other cellsin the different columns or rows through “others”instead of masking them out strictly.

We assign each bias type a learnable scalar,which will serve as a bias term in the self-attentionmodule. Specifically, each self-attention headin each layer have a set of learnable scalars$\{ b _ { 1 } , b _ { 2 } , \cdots , b _ { 1 3 } \}$ corresponding to all types of in-troduced biases. For one head in one self-attentionsub-layer of TABLEFORMER, Equation 2 in theTransformer is replaced by:

$$
\bar { A } = \frac { Q K ^ { \top } } { \sqrt { d _ { K } } } , \quad A = \bar { A } + \hat { A }
$$

$$
\operatorname { A t t n } ( H ) = \operatorname { s o f t m a x } ( A ) V
$$

where $\bar { A }$ is a matrix capturing the similarity be-tween queries and keys, $\hat { A }$ is the Attention BiasMatrix, and Aˆi,j = bφ(vi,vj ).

Relation between TABLEFORMER and ETC.ETC (Ainslie et al., 2020) uses vectors to repre-sent relative position labels, although not directlyapplied to table-text pairs due to its large computa-tional overhead (Eisenschlos et al., 2021). TABLE-FORMER differs from ETC in the following as-pects (1) ETC uses relative positional embeddingswhile TABLEFORMER uses attention bias scalars.In practice, we observed that using relative posi-tional embeddings increases training time by morethan $7 \mathbf { x }$ , (2) ETC uses global memory and local at-tention, while TABLEFORMER uses pairwise atten-tion without any global memory overhead, (3) ETCuses local sparse attention with masking, limitingits ability to attend to all tokens, (4) ETC did notexplore table-text attention bias types exhaustively.Another table encoding model MATE (Eisensch-los et al., 2021) is vulnerable to row and columnperturbations, and shares limitation (3) and (4).

# 4 Experimental Setup

# 4.1 Datasets and Evaluation

We use the following datasets in our experiments.

Table Question Answering. For the table QAtask, we conducted experiments on WikiTableQues-tions (WTQ) (Pasupat and Liang, 2015) and Se-quential QA (SQA) (Iyyer et al., 2017) datasets.WTQ was crowd-sourced based on complex ques-tions on Wikipedia tables. SQA is composed of6, 066 question sequences (2.9 question per se-quence on average), constructed by decomposing asubset of highly compositional WTQ questions.

Table-Text Entailment. For the table-text en-tailment task, we used TABFACT dataset (Chenet al., 2020), where the tables were extracted fromWikipedia and the sentences were written by crowdworkers. Among total 118, 000 sentences, eachone is a positive (entailed) or negative sentence.

Perturbation Evaluation Set. For SQA andTABFACT, we also created new test sets to measuremodels’ robustness to answer-invariant row and col-umn perturbations during inference. Specifically,row and column orders are randomly perturbed forall tables in the standard test sets.2

Pre-training All the models are first tuned onthe Wikipidia text-table pretraining dataset (Herziget al., 2020), optionally tuned on synthetic datasetat an intermediate stage (“inter”) (Eisenschlos et al.,2020), and finally fine-tuned on the target dataset.To get better performance on WTQ, we followHerzig et al. (2020) to further pretrain on SQAdataset after the intermediate pretraining stage inthe “inter-sqa” setting.

Evaluation For SQA, we report the cell selectionaccuracy for all questions (ALL) using the officialevaluation script, cell selection accuracy for all se-quences (SEQ), and the denotation accuracy for allquestions $( \mathrm { A L L } \mathrm { L } _ { \mathrm { d } } )$ . To evaluate the models’ robust-ness in the instance level after perturbations, wealso report a lower bound of example predictionvariation percentage:

$$
V P = { \frac { \left( { \mathrm { t } } 2 { \mathrm { f } } + { \mathrm { f } } 2 { \mathrm { t } } \right) } { \left( { \mathrm { t } } 2 { \mathrm { t } } + { \mathrm { t } } 2 { \mathrm { f } } + { \mathrm { f } } 2 { \mathrm { t } } + { \mathrm { f } } 2 { \mathrm { f } } \right) } }
$$

where t2t, t2f, f2t, and f2f represents how many ex-ample predictions turning from correct to correct,from correct to incorrect, from incorrect to correctand from incorrect to incorrect, respectively, afterperturbation. We report denotation accuracy onWTQ and binary classification accuracy on TAB-FACT respectively.

# 4.2 Baselines

We use TAPASBASE and TAPASLARGE as base-lines, where Transformer architectures are exactlysame as BERTBASE and BERTLARGE (Devlinet al., 2019), and parameters are initialized fromBERTBASE and BERTLARGE respectively. Cor-respondingly, we have our TABLEFORMERBASEand TABLEFORMERLARGE, where attention biasscalars are initialized to zero, and all other pa-rameters are initialized from BERTBASE andBERTLARGE.

# 4.3 Perturbing Tables as Augmented Data

Could we alleviate the spurious ordering biasesby data augmentation alone, without making anymodeling changes? To answer this, we train an-other set of models by augmenting the training datafor TAPAS through random row and column orderperturbations.3

Table 1: Results on SQA test set before and after per-turbation during inference (median of 5 runs). ALL iscell selection accuracy, SEQ is cell selection accuracyfor all question sequences, $\mathrm { { A L L } _ { d } }$ is denotation accu-racy for all questions (reported to compare with Liuet al. (2021)). $V P$ is model prediction variation per-centage after perturbation. Missing values are those notreported in the original paper.  

<table><tr><td></td><td colspan="2">Before Perturb</td><td colspan="2">After Perturb</td></tr><tr><td></td><td>ALL</td><td>SEQ ALLd</td><td>ALL</td><td>VP</td></tr><tr><td>Herzig et al. (2020)</td><td>67.2</td><td>40.4</td><td></td><td>1</td></tr><tr><td>Eisenschlos et al. (2020)</td><td>71.0</td><td>44.8</td><td>1</td><td>1</td></tr><tr><td>Eisenschlos et al.(2021)</td><td>71.7</td><td>46.1</td><td>1</td><td>1</td></tr><tr><td>Liu et al. (2021)</td><td>1</td><td>1 74.5</td><td>1</td><td>1</td></tr><tr><td>TAPASBASE</td><td>61.1</td><td>31.3</td><td>57.4</td><td>14.0%</td></tr><tr><td>TABLEFORMERBASE</td><td>66.7</td><td>39.7</td><td>66.7</td><td>0.2%</td></tr><tr><td>TAPASLARGE</td><td>66.8</td><td>39.9</td><td>60.5</td><td>15.1%</td></tr><tr><td>TABLEFORMERLARGE</td><td>70.3</td><td>44.8</td><td>70.3</td><td>0.1%</td></tr><tr><td>TAPASBASE inter</td><td>67.5</td><td>38.8</td><td>61.0</td><td>14.3%</td></tr><tr><td>TABLEFORMERBASE inter</td><td>69.4</td><td>43.5</td><td>69.3</td><td>0.1%</td></tr><tr><td>TAPASLARGE inter</td><td>70.6</td><td>43.9</td><td>66.1</td><td>10.8%</td></tr><tr><td>TABLEFORMERLARGE inter</td><td>72.4</td><td>47.5</td><td>75.9 72.3</td><td>0.1%</td></tr></table>

For each table in the training set, we randomlyshuffle all rows and columns (including corre-sponding column headers), creating a new tablewith the same content but different orders of rowsand columns. Multiple perturbed versions of thesame table were created by repeating this process$\{ 1 , 2 , 4 , 8 , 1 6 \}$ times with different random seeds.For table QA tasks, selected cell positions are alsoadjusted as final answers according to the perturbedtable. The perturbed table-text pairs are then usedto augment the data used to train the model. Duringtraining, the model takes data created by one spe-cific random seed in one epoch in a cyclic manner.

# 5 Experiments and Results

Besides standard testing results to compare TABLE-FORMER and baselines, we also answer the follow-ing questions through experiments:

• How robust are existing (near) state-of-the-art table-text encoding models to semanticpreserving perturbations in the input?• How does TABLEFORMER compare with ex-isting table-text encoding models when testedon similar perturbations, both in terms of per-formance and robustness?

Table 2: Binary classification accuracy on TABFACT development and 4 splits of test set, as well as performanceon test sets with our perturbation evaluation. Median of 5 independent runs are reported. Missing values are thosenot reported in the original paper.  

<table><tr><td></td><td colspan="5">Before Perturb</td><td colspan="4">After Perturb</td></tr><tr><td></td><td>dev</td><td>test</td><td>testsimple</td><td>testcomplex</td><td>testsmall</td><td>test</td><td>testsimple</td><td>testcomplex</td><td>testsmall</td></tr><tr><td>Eisenschlos et al. (2020)</td><td>81.0</td><td>81.0</td><td>92.3</td><td>75.6</td><td>83.9</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Eisenschlos et al. (2021)</td><td>1</td><td>81.4</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>TABLEFORMERBASE TAPASBASE</td><td>72.8 75.1</td><td>72.3 75.0</td><td>84.8</td><td>66.2</td><td>74.4 77.1</td><td>71.2 75.0</td><td>83.4 88.2</td><td>65.2 68.5</td><td>72.5 77.1</td></tr><tr><td></td><td></td><td></td><td>88.2</td><td>68.5</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TAPASLARGE</td><td>74.7</td><td>74.5</td><td>86.6</td><td>68.6</td><td>76.8</td><td>73.7</td><td>86.0</td><td>67.7</td><td>76.1</td></tr><tr><td>TABLEFORMERLARGE</td><td>77.2</td><td>77.0</td><td>90.2</td><td>70.5</td><td>80.3</td><td>77.0</td><td>90.2</td><td>70.5</td><td>80.3</td></tr><tr><td>TAPASBASE inter</td><td>78.4</td><td>77.9</td><td>90.1</td><td>71.9</td><td>80.5</td><td>76.8</td><td>89.5</td><td>70.5</td><td>79.7</td></tr><tr><td>TABLEFORMERBASE inter</td><td>79.7</td><td>79.2</td><td>91.6</td><td>73.1</td><td>81.7</td><td>79.2</td><td>91.6</td><td>73.1</td><td>81.7</td></tr><tr><td>TAPASLARGE inter</td><td>80.6</td><td>80.6</td><td>92.0</td><td>74.9</td><td>83.1</td><td>79.2</td><td>91.7</td><td>73.0</td><td>83.0</td></tr><tr><td>TABLEFORMERLARGE inter</td><td>82.0</td><td>81.6</td><td>93.3</td><td>75.9</td><td>84.6</td><td>81.6</td><td>93.3</td><td>75.9</td><td>84.6</td></tr></table>

Table 3: Denotation accuracy on WTQ developmentand test set. Median of 5 independent runs are reported.  

<table><tr><td>Model</td><td>dev</td><td>test</td></tr><tr><td>Herzig et al. (2020)</td><td>1</td><td>48.8</td></tr><tr><td>Eisenschlos et al. (2021)</td><td>1</td><td>51.5</td></tr><tr><td>TAPASBASE</td><td>23.6</td><td>24.1</td></tr><tr><td>TABLEFORMERBASE</td><td>34.4</td><td>34.8</td></tr><tr><td>TAPASLARGE</td><td>40.8</td><td>41.7</td></tr><tr><td>TABLEFORMERLARGE</td><td>42.5</td><td>43.9</td></tr><tr><td>TAPASBASE inter-sqa</td><td>44.8</td><td>45.1</td></tr><tr><td>TABLEFORMERBASE inter-sqa</td><td>46.7</td><td>46.5</td></tr><tr><td> TAPASLARGE inter-sqa</td><td>49.9</td><td>50.4</td></tr><tr><td>TABLEFORMERLARGE inter-sqa</td><td>51.3</td><td>52.6</td></tr></table>

• Can we use perturbation based data augmen-tation to achieve robustness at test time?

• Which attention biases in TABLEFORMERcontribute the most to performance?

# 5.1 Main Results

Table 1, 2, and 3 shows TABLEFORMER perfor-mance on SQA, TABFACT, and WTQ, respec-tively. As can be seen, TABLEFORMER outper-forms corresponding TAPAS baseline models in allsettings on SQA and WTQ datasets, which showsthe general effectiveness of TABLEFORMER’sstructural biases in Table QA datasets. Specifi-cally, TABLEFORMERLARGE combined with inter-mediate pretraining achieves new state-of-the-artperformance on SQA dataset.

Similarly, Table 2 shows that TABLEFORMERalso outperforms TAPAS baseline models in all set-tings, which shows the effectiveness of TABLE-FORMER in the table entailment task. Note that,Liu et al. (2021) is not comparable to our results, be-cause they used different pretraining data, differentpretraining objectives, and BART NLG model in-stead of BERT NLU model. But TABLEFORMERattention bias is compatible with BART model.

# 5.2 Perturbation Results

One of our major contributions is to systematicallyevaluate models’ performance when facing row andcolumn order perturbation in the testing stage.

Ideally, model predictions should be consistenton table QA and entailment tasks when facing suchperturbation, because the table semantics remainsthe same after perturbation.

However, in Table 1 and 2, we can see that in ourperturbed test set, performance of all TAPAS mod-els drops significantly in both tasks. TAPAS modelsdrops by at least $3 . 7 \%$ and up to $6 . 5 \%$ in all settingson SQA dataset in terms of ALL accuracy, whileour TABLEFORMER being strictly invariant to suchrow and column order perturbation leads to no dropin performance.4 Thus, in the perturbation setting,TABLEFORMER outperforms all TAPAS baselineseven more significantly, with at least $6 . 2 \%$ and$2 . 4 \%$ improvements on SQA and TABFACT dataset,respectively. In the instance level, we can see that,with TAPAS, there are many example predictionschanged due to high $V P$ , while there is nearly noexample predictions changed with TABLEFORMER(around zero $V P$ ).

Table 4: Model size comparison.  

<table><tr><td>Model</td><td>Number of parameters</td></tr><tr><td>TAPASBASE</td><td>110M</td></tr><tr><td rowspan="2">TABLEFORMERBASE</td><td>110 M - 2*512*768 + 12*12*13=</td></tr><tr><td>110 M-0.8M+0.002M</td></tr><tr><td>TAPASLARGE</td><td>340M</td></tr><tr><td>TABLEFORMERLARGE</td><td>340 M - 2*512*1024 +24*16*13 = 340M-1.0M+0.005M</td></tr></table>

# 5.3 Model Size Comparison

We compare the model sizes of TABLEFORMERand TAPAS in Table 4. We added only a few atten-tion bias scalar parameters (13 parameters per headper layer) in TABLEFORMER, which is negligiblecompared with the BERT model size. Meanwhile,we delete two large embedding metrics (512 rowids and 512 column ids). Thus, TABLEFORMERoutperforms TAPAS with fewer parameters.

# 5.4 Analysis of TABLEFORMER Submodules

In this section, we experiment with several variantsof TABLEFORMER to understand the effectivenessof its submodules. The performance of all variantsof TAPAS and TABLEFORMER that we tried on theSQA development set is shown in Table 5.

Learnable Attention Biases v/s Masking. In-stead of adding learnable bias scalars, we mask outsome attention scores to restrict attention to thosetokens in the same columns and rows, as well asthe paired sentence, similar to Zhang et al. (2020)(SAT). We can see that TAPASBASE-SAT performsworse than TAPASBASE, which means that restrict-ing attention to only same columns and rows bymasking reduce the modeling capacity. This led tochoosing soft bias addition over hard masking.

Attention Bias Scaling. Unlike TABLE-FORMER, we also tried to add attention biasesbefore the scaling operation in the self-attentionmodule (SO). Specifically, we compute pair-wiseattention score by:

$$
A _ { i j } = \frac { ( h _ { i } ^ { \top } W ^ { Q } ) ( h _ { j } ^ { \top } W ^ { K } ) ^ { \top } + \hat { A } _ { i j } } { \sqrt { d _ { K } } }
$$

<table><tr><td></td><td>rc-gp</td><td>c-gp</td><td>gp</td><td>pcp</td></tr><tr><td>TAPASBASE</td><td>57.6</td><td>47.4</td><td>46.4</td><td>29.1</td></tr><tr><td>TAPASBASE-SAT</td><td>45.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TABLEFORMERBASE-SO</td><td>60.0</td><td>60.2</td><td>59.8</td><td>60.7</td></tr><tr><td>TABLEFORMERBASE</td><td>62.2</td><td>61.5</td><td>61.7</td><td>61.9</td></tr></table>

Table 5: ALL questions’ cell selection accuracy ofTABLEFORMER variants on SQA development set. rc-$g p$ represents the setting including row ids, columnids and global positional ids, $c { \scriptscriptstyle - } g p$ represents columnids and global positional ids, $g p$ represents global po-sitional ids, and pcp represents per-cell positional ids.“SAT” represents masking out some attention scores.“SO” represents adding attention bias before scaling.

instead of using:

$$
A _ { i j } = \frac { ( { h _ { i } ^ { \top } W ^ { Q } } ) ( { h _ { j } ^ { \top } W ^ { K } } ) ^ { \top } } { \sqrt { d _ { K } } } + \hat { A } _ { i j } ,
$$

which is the element-wise version of Equa-tion 1 and 3. However, Table 5 showsthat TABLEFORMERBASE-SO performs worse thanTABLEFORMERBASE, showing the necessity ofadding attention biases after the scaling operation.We think the reason is that the attention bias termdoes not require scaling, because attention biasscalar magnitude is independent of $d _ { K }$ , while thedot products grow large in magnitude for large val-ues of $d _ { K }$ . Thus, such bias term could play anmore important role without scaling, which helpseach attention head know clearly what to pay moreattention to according to stronger inductive biases.

Row, Column, & Global Positional IDs.With TAPASBASE, TABLEFORMERBASE-SO, andTABLEFORMERBASE, we first tried the full-versionwhere row ids, column ids, and global positionalids exist as input $( r c { - } g p )$ . Then, we deleted rowids $( c { - } g p )$ , and column ids $( g p )$ sequentially. Fi-nally, we changed global positional ids in $g p$ toper-cell positional ids $( p c p )$ . Table 5 shows thatTAPASBASE performs significantly worse from rc-$g p  c \cdot g p  g p  p c p$ , because table structure in-formation are deleted sequentially during such pro-cess. However, with TABLEFORMERBASE, there isno obvious performance drop during the same pro-cess. That shows the structural inductive biases inTABLEFORMER can provide complete table struc-ture information. Thus, row ids, column ids andglobal positional ids are not necessary in TABLE-FORMER. We pick TABLEFORMER pcp setting asour final version to conduct all other experiments inthis paper. In this way, TABLEFORMER is strictlyinvariant to row and column order perturbation byavoiding spurious biases in those original ids.

Table 6: Comparison of TABLEFORMER and perturbeddata augmentation on SQA test set, where $V P$ repre-sents model prediction variation percentage after per-turbation. Median of 5 independent runs are reported.  

<table><tr><td></td><td>Befor Perturb</td><td>After Perturb</td></tr><tr><td></td><td>ALL SEQ</td><td>ALL VP</td></tr><tr><td>TAPASBASE</td><td>61.1 31.3</td><td>57.4 14.0%</td></tr><tr><td>TAPASBASE 1p</td><td>63.4 34.6 64.6</td><td>63.4 9.9%</td></tr><tr><td>TAPASBASE 2p</td><td>35.6</td><td>64.5 8.4%</td></tr><tr><td>TAPASBASE 4p</td><td>37.0</td><td>65.0 8.1%</td></tr><tr><td>TAPASBASE 8p</td><td>37.3</td><td>64.3 7.2%</td></tr><tr><td>TAPASBASE 16p</td><td>33.6</td><td>62.2 7.0%</td></tr><tr><td>TABLEFORMERBASE</td><td>39.7</td><td>66.7</td><td>0.1%</td></tr></table>

# 5.5 Comparison of TABLEFORMER andPerturbed Data Augmentation

As stated in Section 4.3, perturbing row and col-umn orders as augmented data during training canserve as another possible solution to alleviate thespurious row/column ids bias. Table 6 shows theperformance of TABPASBASE model trained withadditional $\{ 1 , 2 , 4 , 8 , 1 6 \}$ perturbed versions ofeach table as augmented data.

We can see that the performance of TAPASBASEon SQA dataset improves with such augmentation.Also, as the number of perturbed versions of eachtable increases, model performance first increasesand then decreases, reaching the best results with8 perturbed versions. We suspect that too manyversions of the same table confuse the model aboutdifferent row and column ids for the same table,leading to decreased performance from 8p to 16p.Despite its usefulness, such data perturbation isstill worse than TABLEFORMER, because it couldnot incorporate other relevant text-table structuralinductive biases like TABLEFORMER.

Although, such data augmentation makes themodel more robust to row and column order per-turbation with smaller $V P$ compared to standardTAPASBASE, there is still a significant predictiondrift after perturbation. As shown in Table 6, $V P$decreases from 1p to 16p, however, the best $V P$$( 7 . 0 \% )$ is still much higher than (nearly) no varia-tion $( 0 . 1 \% )$ of TABLEFORMER.

To sum up, TABLEFORMER is superior to rowand column order perturbation augmentation, be-cause of its additional structural biases and strictlyconsistent predictions after perturbation.

Table 7: Ablation study of proposed attention biases.  

<table><tr><td></td><td>ALL</td><td>SEQ</td></tr><tr><td>TABLEFORMERBASE</td><td>62.1</td><td>38.4</td></tr><tr><td>- Same Row</td><td>32.1 62.1</td><td>2.8 37.7</td></tr><tr><td>- Same Column - Same Cell</td><td>61.8</td><td>38.4</td></tr><tr><td>- Cell to Column Header</td><td>60.7</td><td>36.6</td></tr><tr><td>- Cell to Sentence</td><td>60.5</td><td>36.4</td></tr><tr><td>- Header to Column Cell</td><td>60.5</td><td>35.8</td></tr><tr><td>- Header to Other Header</td><td>60.6</td><td>35.8</td></tr><tr><td>- Header to Same Header</td><td>61.0</td><td>36.9</td></tr><tr><td>- Header to Sentence</td><td>61.1</td><td>36.3</td></tr><tr><td>- Sentence to Cell</td><td></td><td>36.2</td></tr><tr><td>- Sentence to Header</td><td>60.8</td><td>37.3</td></tr><tr><td></td><td>61.0</td><td></td></tr><tr><td>- Sentence to Sentence</td><td>60.0</td><td>35.3</td></tr><tr><td>- All Column Related (# 2,4,6)</td><td>54.5</td><td>29.3</td></tr></table>

# 5.6 Attention Bias Ablation Study

We conduct ablation study to demonstrate the util-ity of all 12 types of defined attention biases. Foreach ablation, we set the corresponding attentionbias type id to “others” bias id. Table 7 showsTAPASBASE’s performance SQA dev set. Over-all, all types of attention biases help the TABLE-FORMER performance to some extent, due to cer-tain performance drop after deleting each bias type.

Amongst all the attention biases, deleting “samerow” bias leads to most significant performancedrop, showing its crucial role for encoding tablerow structures. There is little performance dropafter deleting “same column” bias, that’s becauseTABLEFORMER could still infer the same columninformation through “cell to its column header”and “header to its column cell” biases. Afterdeleting all same column information (“same col-umn”, “cell to column header” and “header to col-umn cell” biases), TABLEFORMER performs signif-icantly worse without encoding column structures.Similarly, there is little performance drop afterdeleting “same cell” bias, because TABLEFORMERcan still infer same cell information through “samerow” and “same column” biases.

# 5.7 Limitations of TABLEFORMER

TABLEFORMER increases the training time byaround $2 0 \%$ , which might not be ideal for verylong tables and would require a scoped approach.Secondly, with the strict row and column order in-variant property, TABLEFORMER cannot deal withquestions based on absolute orders of rows in ta-bles. This however is not a practical requirementbased on the current dataset. Doing a manual studyof 1800 questions in SQA dataset, we found thatthere are 4 questions5 $0 . 2 \%$ percentage) whoseanswers depend on orders of rows. Three of themasked “which one is at the top of the table”, an-other asks “which one is listed first”. However,these questions could be potentially answered byadding back row and column order informationbased on TABLEFORMER.

# 6 Other Related Work

Transformers for Tabular Data. Yin et al.(2020) prepended corresponding column headersto cells contents, and Chen et al. (2020) used cor-responding column headers as features for cells.However, such methods encode each table headermultiple times, leading to duplicated computingoverhead. Also, tabular structures (e.g. same rowinformation) are not fully incorporated to such mod-els. Meanwhile, Yin et al. (2020) leveraged rowencoder and column encoder sequentially, whichintroduced much computational overhead, thus re-quiring retrieving some rows as a preprocessingstep. Finally, SAT (Zhang et al., 2020), Denget al. (2021) and Wang et al. (2021) restricted atten-tion to same row or columns with attention mask,where such inductive bias is too strict that cellscould not directly attend to those cells in differentrow and columns, hindering the modeling abilityaccording to Table 5. Liu et al. (2021) used theseq2seq BART generation model with a standardTransformer encoder-decoder architecture. In allmodels mentioned above, spurious inter-cell or-der biases still exist due to global positional idsof Transformer, leading to the vulnerability to rowor column order perturbations, while our TABLE-FORMER could avoid such problem. Mueller et al.(2019) and Wang et al. (2020) also used relativepositional encoding to encode table structures, butthey modeled the relations as learnable relation vec-tors, whose large overhead prevented pretrainingand led to poor performance without pretraining,similarly to ETC (Ainslie et al., 2020) explained inSection 3.

Structural and Relative Attention. Modifiedattention scores has been used to model relativepositions (Shaw et al., 2018), long documents (Daiet al., 2019; Beltagy et al., 2020; Ainslie et al.,2020), and graphs (Ying et al., 2021). But addinglearnable attention biases to model tabular struc-tures has been under-explored.

# 7 Conclusion

In this paper, we identified the vulnerability ofprior table encoding models along two axes: (a)capturing the structural bias, and (b) robustnessto row and column perturbations. To tacklethis, we propose TABLEFORMER, where learnabletask-independent learnable structural attention bi-ases are introduced, while making it invariant torow/column order at the same time. Experimentalresults showed that TABLEFORMER outperformsstrong baselines in 3 table reasoning tasks, achiev-ing state-of-the-art performance on SQA dataset,especially when facing row and column order per-turbations, because of its invariance to row andcolumn orders.

# Acknowledgments

We thank Julian Eisenschlos, Ankur Parikh, andthe anonymous reviewers for their feedbacks inimproving this paper.

# Ethical Considerations

The authors foresee no ethical concerns with theresearch presented in this paper.

# References

Joshua Ainslie, Santiago Ontanon, Chris Alberti, Va-clav Cvicek, Zachary Fisher, Philip Pham, AnirudhRavula, Sumit Sanghai, Qifan Wang, and Li Yang.2020. ETC: Encoding long and structured inputsin transformers. In Proceedings of the 2020 Con-ference on Empirical Methods in Natural LanguageProcessing (EMNLP), pages 268–284, Online. Asso-ciation for Computational Linguistics.Iz Beltagy, Matthew E. Peters, and Arman Cohan.2020. Longformer: The long-document transformer.arXiv:2004.05150.Wenhu Chen, Hongmin Wang, Jianshu Chen, YunkaiZhang, Hong Wang, Shiyang Li, Xiyou Zhou, andWilliam Yang Wang. 2020. Tabfact : A large-scaledataset for table-based fact verification. In Inter-national Conference on Learning Representations(ICLR), Addis Ababa, Ethiopia.Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Car-bonell, Quoc Le, and Ruslan Salakhutdinov. 2019.Transformer-XL: Attentive language models beyonda fixed-length context. In Proceedings of the 57thAnnual Meeting of the Association for Computa-tional Linguistics, pages 2978–2988, Florence, Italy.Association for Computational Linguistics.

Xiang Deng, Huan Sun, Alyssa Lees, You Wu, andCong Yu. 2021. TURL: Table Understandingthrough Representation Learning. In VLDB.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, andKristina Toutanova. 2019. BERT: Pre-training ofdeep bidirectional transformers for language under-standing. In Proceedings of the 2019 Conferenceof the North American Chapter of the Associationfor Computational Linguistics: Human LanguageTechnologies, Volume 1 (Long and Short Papers),pages 4171–4186, Minneapolis, Minnesota. Associ-ation for Computational Linguistics.

Julian Eisenschlos, Maharshi Gor, Thomas Müller, andWilliam Cohen. 2021. MATE: Multi-view attentionfor table transformer efficiency. In Proceedings ofthe 2021 Conference on Empirical Methods in Natu-ral Language Processing, pages 7606–7619, Onlineand Punta Cana, Dominican Republic. Associationfor Computational Linguistics.

Julian Eisenschlos, Syrine Krichene, and ThomasMüller. 2020. Understanding tables with interme-diate pre-training. In Findings of the Associationfor Computational Linguistics: EMNLP 2020, pages281–296, Online. Association for ComputationalLinguistics.

Jonathan Herzig, Pawel Krzysztof Nowak, ThomasMüller, Francesco Piccinno, and Julian Eisenschlos.2020. TaPas: Weakly supervised table parsing viapre-training. In Proceedings of the 58th AnnualMeeting of the Association for Computational Lin-guistics, pages 4320–4333, Online. Association forComputational Linguistics.

Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang. 2017.Search-based neural structured learning for sequen-tial question answering. In Proceedings of the55th Annual Meeting of the Association for Com-putational Linguistics (Volume 1: Long Papers),pages 1821–1831, Vancouver, Canada. Associationfor Computational Linguistics.

Qian Liu, Bei Chen, Jiaqi Guo, Zeqi Lin, and Jian-guang Lou. 2021. Tapex: Table pre-training vialearning a neural sql executor. arXiv preprintarXiv:2107.07653.

Thomas Mueller, Francesco Piccinno, Peter Shaw,Massimo Nicosia, and Yasemin Altun. 2019. An-swering conversational questions on structured datawithout logical forms. In Proceedings of the2019 Conference on Empirical Methods in Natu-ral Language Processing and the 9th InternationalJoint Conference on Natural Language Processing(EMNLP-IJCNLP), pages 5902–5910, Hong Kong,China. Association for Computational Linguistics.

Panupong Pasupat and Percy Liang. 2015. Compo-sitional semantic parsing on semi-structured tables.In Proceedings of the 53rd Annual Meeting of theAssociation for Computational Linguistics and the7th International Joint Conference on Natural Lan-guage Processing (Volume 1: Long Papers), pages

1470–1480, Beijing, China. Association for Compu-tational Linguistics.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani.2018. Self-attention with relative position represen-tations. In Proceedings of the 2018 Conference ofthe North American Chapter of the Association forComputational Linguistics: Human Language Tech-nologies, Volume 2 (Short Papers), pages 464–468,New Orleans, Louisiana. Association for Computa-tional Linguistics.

Ashish Vaswani, Noam Shazeer, Niki Parmar, JakobUszkoreit, Llion Jones, Aidan N Gomez, ŁukaszKaiser, and Illia Polosukhin. 2017. Attention is allyou need. In Advances in neural information pro-cessing systems, pages 5998–6008.

Bailin Wang, Richard Shin, Xiaodong Liu, OleksandrPolozov, and Matthew Richardson. 2020. RAT-SQL:Relation-aware schema encoding and linking fortext-to-SQL parsers. In Proceedings of the 58th An-nual Meeting of the Association for ComputationalLinguistics, pages 7567–7578, Online. Associationfor Computational Linguistics.

Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, ZhiyiFu, Shi Han, and Dongmei Zhang. 2021. TUTA:Tree-based Transformers for Generally StructuredTable Pre-training. In Proceedings of the 27th ACMSIGKDD Conference on Knowledge Discovery &Data Mining, pages 1780–1790.

Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Se-bastian Riedel. 2020. TaBERT: Pretraining for jointunderstanding of textual and tabular data. In Pro-ceedings of the 58th Annual Meeting of the Asso-ciation for Computational Linguistics, pages 8413–8426, Online. Association for Computational Lin-guistics.

Chengxuan Ying, Tianle Cai, Shengjie Luo, ShuxinZheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu. 2021. Do Transformers Really PerformBad for Graph Representation? arXiv preprintarXiv:2106.05234.

Tao Yu, Chien-Sheng Wu, Xi Victoria Lin, BailinWang, Yi Chern Tan, Xinyi Yang, Dragomir Radev,Richard Socher, and Caiming Xiong. 2020. GraPPa:Grammar-Augmented Pre-Training for Table Se-mantic Parsing. arXiv preprint arXiv:2009.13845.

Hongzhi Zhang, Yingyao Wang, Sirui Wang, XuezhiCao, Fuzheng Zhang, and Zhongyuan Wang. 2020.Table fact verification with structure-aware trans-former. In Proceedings of the 2020 Conference onEmpirical Methods in Natural Language Process-ing (EMNLP), pages 1624–1629, Online. Associa-tion for Computational Linguistics.