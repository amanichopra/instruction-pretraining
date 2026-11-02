# Does instruction pretraining change what models can learn next?

Research brief and experiment proposal. Literature review date: **2026-11-01**.

**Recommendation:** investigate whether instruction-formatted pretraining changes subsequent learning because of its task structure, its selection of information, or its interaction with optimization. Use controlled synthetic experiments to separate these effects, then test their relevance on paired natural-language corpora.

The strongest prospective contribution is a causal account of **transfer to withheld facts, tasks, and compositions**, rather than another demonstration that instruction pretraining improves immediate instruction-following scores. Whether the effect favors raw text, instructions, or a mixture is an experimental question.

Agreed scope: training from scratch, later supervised learning, substantial GPU access, and a causal/mechanistic paper angle. RL, long-context extension, and deployment are outside the initial study. This document proposes experiments; it does not report new experimental results.

## 1. What is the actual scientific question?

### Same loss formula does not imply the same learning process

A useful common description is

$$
L_t(\theta) = -\mathbb{E}_{z \sim D_t}\left[\frac{1}{Z(z)}\sum_i w_i\log p_\theta(z_i\mid z_{<i})\right].
$$

Here, $D_t$ is the distribution sampled at training step $t$, $w_i$ is the token weight, and $Z$ specifies loss normalization. With the same causal attention convention, unit weights on all non-padding tokens, and the same normalization, raw-text pretraining and unmasked SFT use the same loss formula. They generally optimize different expected losses because their data distributions differ.

For an instruction $x$ and response $y$, full-sequence training models $p(x,y)=p(x)p(y\mid x)$; response-only training models the conditional response likelihood. Masking is therefore meaningful, but removing masking does not remove differences in information content, sequence structure, repetition, or gradient statistics.

The training trajectory also depends on data ordering, learning rates, weight decay, optimizer state, packing, and context length. A sharp stage boundary is one possible data/optimization schedule. There is no mathematical requirement to restart training merely because examples become instructions. There can still be practical reasons for a separate stage: late-arriving data, scarce high-quality demonstrations, a changed target distribution, or adapting one shared base model to many users.

Likewise, training a randomly initialized model on instruction data is still pretraining. The question is whether we need a **raw-document component**, not whether initial learning can be eliminated. Synthetic instructions also inherit information from source documents and previously trained generators. A successful instruction-only student would not show that raw text was unnecessary throughout that larger system.

### What should “better future learning” mean?

Keep the following outcomes separate:

| Outcome | Measurement | What it does not establish by itself |
| --- | --- | --- |
| Better starting performance | Accuracy before target SFT | Faster acquisition of new information |
| Better sample efficiency | Target examples/tokens needed to reach a shared score | Better attainable performance |
| Better optimization efficiency | Performance versus target training steps/FLOPs | Better efficiency per unique example |
| Better final performance | Score after equal-budget, validation-tuned adaptation | Less forgetting |
| Better retention | Change in previously measured capabilities after adaptation | Easier acquisition of new capabilities |
| Different curriculum | Skill trajectories and acquisition order across checkpoints | A causal internal dependency between those skills |

Do not call a larger post-SFT gain “greater plasticity” solely because a model started lower. Report starting scores and absolute trajectories alongside gains.

## 2. What the existing evidence does—and does not—establish

### FineInstructions is a strong starting point, with several bundled interventions

[FineInstructions v3](https://arxiv.org/html/2601.22146v3) constructs roughly 18M user-derived instruction templates and grounds responses largely in source-document excerpts. Its main experiments compare 1.8B-parameter models on matched training-token budgets. The 23B-token IPT-derived corpus is repeated four times, while the 300B-token Nemotron-derived corpus is used for one epoch; corpus size and training exposure must not be conflated.

Its main evaluation uses MixEval, MT-Bench-101, and AlpacaEval, and reports stronger immediate response performance. These results support the effectiveness of the complete pipeline. They do not isolate instruction syntax from selection of educational content, answer extraction, template matching, judging, or changes in context and repetition. The paper explicitly does not optimize for raw-text perplexity.

Its diversity evidence includes millions of instantiated templates and low concentration in individual templates. This is valuable evidence of template diversity, but is not a measurement of every fact retained from the original documents, independent computational skills, or support for future tasks. Many distinct templates can exercise similar operations. Conversely, a small set of operations can cover many useful facts.

The main reported comparisons do not settle our proposed transfer question. However, **do not claim that the authors never tried subsequent SFT**: their [released experiments](https://huggingface.co/fineinstructions/pretraining_experiments/tree/main) include raw-text and FineInstructions checkpoints labeled with SFT variants, with model cards and training arguments. Their existence does not establish the results of a systematic held-out-task transfer study, but it rules out treating any subsequent SFT experiment as obviously new. The inspected pretraining configuration alone also does not conclusively establish the historical prompt-masking implementation; verify trainer code before claiming an exact reproduction.

### Three nearby results make the novelty boundary narrower

1. **Instruction pretraining followed by SFT already exists.** [Instruction Pre-Training](https://arxiv.org/abs/2406.14491) reports improved subsequent instruction-tuning trajectories, including MMLU evaluation during tuning. Its raw-plus-instruction construction differs from FineInstructions, but it is a direct predecessor.
2. **Early exposure to future data already helps in some settings.** [The Finetuner’s Fallacy](https://arxiv.org/abs/2603.16177) repeats specialized data during pretraining and evaluates later fine-tuning. Our central test should withhold future task data, rather than merely introduce that same data earlier.
3. **Unification and future-learning benefits are already studied.** [Training Foundation Models with a Single Algorithm](https://arxiv.org/html/2610.40564v1) formulates an evolving likelihood-training distribution and evaluates subsequent learning, including coding RL after mathematical training. Its RL results are related work, not part of our experiment scope.

**Prospective gap:** a study separating presentation from information coverage, testing future tasks absent from the controlled pretraining support, and explaining differences through interventions on skills and training dynamics. This is a promising distinction from the inspected literature, not an exhaustive proof of novelty.

## 3. Annotated reading list

Reading-depth labels refer to this review: **sections inspected** means relevant full-text sections were read, not that every appendix or implementation was audited; **abstract screened** means detailed methodological claims still require reading. Paper titles link to primary sources. Priority is relative to this project. Years follow the initial arXiv identifier unless a version is specified.

### A. Instruction pretraining and restructuring

**[R01] [FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale](https://arxiv.org/html/2601.22146v3) — 2026, v3.** Priority: essential. Depth: sections inspected; dataset schema and selected model artifacts inspected.

Introduces large-scale user-derived templates, document matching, excerpt-grounded answers, and judging. Controlled token-budget comparisons favor the pipeline on the reported response-quality benchmarks. Read Sections 3–6 to identify which interventions are bundled and what diversity means operationally. Reuse its paired source/answer data rather than rebuilding the entire generation system initially. [Resources](https://huggingface.co/fineinstructions); [dataset](https://huggingface.co/datasets/fineinstructions/fineinstructions_nemotron).

**[R02] [Instruction Pre-Training: Language Models are Supervised Multitask Learners](https://arxiv.org/abs/2406.14491) — 2024.** Priority: essential. Depth: sections inspected.

Augments raw corpora with synthesized instruction–response pairs and reports both stronger base models and better subsequent instruction tuning. Figure 4 is a particularly important novelty check: faster downstream learning is already part of its claim. Its mixtures include raw documents and synthesizer-training data, so it does not directly settle the consequences of replacing raw documents with FineInstructions-style standalone pairs. Reproduce its conceptual baseline and compare the support of pretraining and downstream tasks carefully.

**[R03] [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380) — 2024.** Priority: high. Depth: abstract screened.

WRAP mixes original web text with model-generated rephrasings and reports improvements in training efficiency and downstream performance. Its discussion of both style diversity and quality is directly relevant: an apparent instruction-format advantage may partly be a rewriting advantage. Use this as motivation for a content-preserving rephrasing baseline if the initial natural-language results justify another pretraining arm. Do not attribute its gains specifically to future learning.

**[R04] [Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset](https://arxiv.org/abs/2412.02595) — 2024.** Priority: high. Depth: abstract screened, with its use in FineInstructions inspected.

Combines filtering and synthetic transformations to improve the trade-off between data quantity and quality, including long training horizons. It is the source of the main FineInstructions corpus and several relevant baselines. Read the data construction before deciding which raw documents constitute a fair control: quality filtering and source inclusion should not silently differ across arms.

**[R05] [Recycling the Web: A Method to Enhance Pre-training Data Quality and Quantity for Language Models](https://arxiv.org/abs/2506.04689) — 2025.** Priority: supporting. Depth: abstract screened.

REWIRE transforms documents that would otherwise be discarded and reports gains from mixing rewritten and high-quality raw text. It motivates distinguishing formatting from recovered information and improved source quality. Its results also caution against assuming all synthetic transformations reduce coverage: some can make previously unusable material useful. It is a transformation baseline, not direct evidence about future SFT efficiency.

### B. What instruction tuning changes

**[R06] [Instruction Tuning With Loss Over Instructions](https://arxiv.org/html/2405.14394v2) — 2024, v2.** Priority: high. Depth: abstract and introductory analysis inspected.

Studies instruction modelling, which includes prompt tokens in the loss. Reports benefits particularly for small datasets and long prompts paired with short outputs, and offers reduced overfitting as an explanation. This supports an unmasked main experiment, but not a universal claim that masking is inferior. Separate prompt/response lengths, repetition, and loss normalization when interpreting the masking ablation. [Code](https://github.com/ZhengxiangShi/InstructionModelling).

**[R07] [Instruction Fine-Tuning: Does Prompt Loss Matter?](https://arxiv.org/html/2401.13586v3) — 2024, v3.** Priority: high. Depth: methods and introductory results inspected.

Varying prompt-loss weights produces different behavior for short- and long-completion data and for different evaluation types. Intermediate weights can be favorable; the conclusion is more nuanced than a binary masked/unmasked rule. Read the dataset construction and regularization analysis before treating response length as a causal explanation. Our first ablation uses weights 0 and 1; a fractional-weight sweep belongs in follow-up work if masking changes the main result.

**[R08] [Instruction Following without Instruction Tuning](https://arxiv.org/abs/2409.14254) — 2024.** Priority: high. Depth: abstract screened.

Shows that response-only training and other restricted adaptations can elicit broad instruction following from pretrained models. It motivates the answers-alone and generic-format-adaptation controls: a model can improve at expressing existing capabilities without learning the intended instruction–response mapping. These findings concern adaptation of pretrained models; they should not be assumed to hold identically when training from scratch.

**[R09] [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — 2023.** Priority: supporting. Depth: abstract screened.

A small, carefully curated demonstration set produces strong instruction-following behavior from a large pretrained model. Its interpretation motivates asking how much SFT changes knowledge versus output behavior. Treat the superficial-alignment interpretation as a hypothesis supported in its setting, not a theorem that SFT cannot teach new information. Our new-fact branch is designed to measure that distinction directly.

### C. Training order, distribution transitions, and replay

**[R10] [Training Foundation Models with a Single Algorithm](https://arxiv.org/html/2610.40564v1) — 2026, v1.** Priority: essential for positioning. Depth: formulation and relevant empirical sections inspected.

Expresses static data and suitable RL targets within a common likelihood framework, with staged training as a special case. Its later-learning experiments show that prior training mixtures can affect subsequent coding RL. It blocks novelty claims based merely on one trainer, one loss, or evaluating another training stage. Our distinction is supervised transfer under independently controlled format and coverage; read Sections 5.2.3 and 7 alongside the formulation.

**[R11] [The Finetuner’s Fallacy: When to Pretrain with Your Finetuning Data](https://arxiv.org/abs/2603.16177) — 2026.** Priority: essential. Depth: relevant experimental sections inspected.

Repeats small specialized corpora during pretraining and reports better domain performance after fine-tuning, with less forgetting and different overfitting behavior. Its central treatment exposes the model to future domain data early. Our withheld-task experiments must distinguish this useful specialization effect from broadly improved capacity to learn later. Its replay comparisons and overfitting analysis are especially relevant baselines for project 4.

**[R12] [Replaying pre-training data improves fine-tuning](https://arxiv.org/abs/2603.04964) — 2026.** Priority: high. Depth: abstract screened.

Reports that generic-data replay can improve the target task, beyond preserving generic capabilities. It also studies how replay interacts with target-data exposure during pretraining. This motivates a bounded replay control if one pretraining condition seems harder to adapt. Interpret replay gains with R13: whether replay helps depends on the schedule and budget accounting.

**[R13] [Data Schedules and Learning Rate Schedules Should Be Designed Together](https://arxiv.org/html/2610.29348v2) — 2026, v2.** Priority: essential for experimental controls. Depth: introduction and principal findings inspected.

With fixed total compute and midtraining-token budgets, the ranking of data schedules can reverse when the learning-rate schedule changes. Earlier mixing helps in the reported full-decay setting, while a final pure-midtraining stage can win under constant LR. The paper also studies delayed decay. Its main target is midtraining loss, so neither the winning schedule nor the mechanism should be assumed to transfer unchanged to our held-out SFT evaluations.

**[R14] [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](https://arxiv.org/abs/2511.18903) — 2025 preprint.** Priority: high. Depth: abstract screened.

Finds that an ascending data-quality curriculum interacts unfavorably with standard LR decay and studies alternatives. It motivates crossing data order with optimization rather than declaring an intrinsically optimal order. The studied curriculum is based on data quality, not an independently controlled instruction-format intervention. Use it to design schedule controls, not to predict which format must appear first.

**[R15] [Midtraining Bridges Pretraining and Posttraining Distributions](https://arxiv.org/abs/2510.14865) — 2025.** Priority: high. Depth: abstract screened.

Frames midtraining as a distributional bridge and reports interactions between domain proximity, entry time, and mixture weight. This is an alternative to an intrinsic-plasticity explanation: a model may adapt better simply because its training distribution is closer to the target. Evaluate both near and distant task families and avoid defining distance only from the candidate models’ own representations.

**[R16] [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763) — 2024.** Priority: supporting. Depth: abstract screened.

Studies learning-rate rewarming, redecay, and replay when updating models under distribution shift, including comparisons with retraining. It provides practical context for the treatment of optimizer transitions and old-data retention. Its continued-pretraining setting differs from a from-scratch format comparison followed by target SFT; borrow the controls without treating its conclusions as direct evidence for instruction-only pretraining.

### D. Adaptability versus current performance

**[R17] [Overtrained Language Models Are Harder to Fine-Tune](https://arxiv.org/abs/2503.19206) — 2025.** Priority: essential. Depth: abstract screened.

Reports catastrophic overtraining: additional pretraining can yield worse models after fine-tuning, and links this to sensitivity to parameter changes. This is direct motivation for evaluating intermediate checkpoints and later adaptation, rather than selecting pretraining recipes exclusively by their final base scores. It does not establish that instruction data prevents or causes this effect.

**[R18] [(How) Learning Rates Regulate Catastrophic Overtraining](https://arxiv.org/abs/2604.13627) — 2026.** Priority: essential. Depth: abstract screened; related references inspected.

Connects fine-tuning learning rates, forgetting, pretraining decay, and sharpness. It reports qualitatively different fine-tuned models even at the same SFT loss. Consequently, a shared learning rate is a useful controlled comparison but insufficient evidence that one initialization is fundamentally less adaptable: also provide equal-budget hyperparameter tuning. Sharpness measurements can be supporting diagnostics, not a standalone causal explanation.

**[R19] [Weight Decay Improves Language Model Plasticity](https://arxiv.org/abs/2602.11137) — 2026.** Priority: essential. Depth: abstract screened.

Studies pretraining weight decay through downstream adaptability and reports cases where worse base models become better models after fine-tuning. It motivates holding regularization fixed initially and testing sensitivity if a format effect appears. Proposed mechanistic links include representations and attention, but importing that explanation into our study would require measuring and intervening on the relevant behavior.

**[R20] [Critical Learning Periods in Deep Neural Networks](https://arxiv.org/abs/1711.08856) — 2017 preprint.** Priority: supporting conceptual background. Depth: abstract screened.

Studies how an early information deficit can have effects that later exposure does not fully repair, in settings outside our LLM experiments. It inspires early-versus-late restoration of omitted facts or prerequisite operations. A persistent deficit at one late learning rate is not enough to establish a critical period; rule out insufficient adaptation budget and unsuitable optimization first.

### E. Skill acquisition and composition

**[R21] [What do Language Models Learn and When? The Implicit Curriculum Hypothesis](https://arxiv.org/abs/2604.08510) — 2026.** Priority: essential. Depth: task design and methodology inspected; repository README inspected.

Tracks 53 simple and 38 composite tasks across pretrained model checkpoints, finding substantial stability in acquisition order and relationships with task representations. The released [ElementalTask code and data](https://github.com/KaiserWhoLearns/ElementalTask) are a concrete evaluation starting point. Its compositional dependencies are design-level relationships, not proven internal prerequisites. Its original evaluation is largely an ICL probe; converting it into a supervised-transfer benchmark requires new splits, fresh instances, and an explicitly documented protocol.

**[R22] [Skill-It! A Data-Driven Skills Framework for Understanding and Training Language Models](https://arxiv.org/abs/2307.14430) — 2023.** Priority: high. Depth: abstract screened.

Uses skill relationships to study and select training data, with synthetic and natural-language experiments. It suggests a stronger test than correlated emergence: train on a putative prerequisite and measure whether the target becomes easier to learn. It also means that discovering a useful ordering or proposing a skill-aware sampler is not sufficient novelty on its own. Our question is how presentation and retained content alter those relationships.

**[R23] [Irreducible Curriculum for Language Model Pretraining](https://arxiv.org/abs/2310.15389) — 2023.** Priority: supporting. Depth: abstract screened.

Selects data using estimated learnability, with a smaller proxy model limiting selection overhead. It offers an alternative explanation for instruction-data gains: the transformation may change which examples are easy or useful to learn at a given time. A sampler based on learnability is a possible later baseline, but adding one in the initial study would introduce another moving component.

### F. Scaling and resource accounting

**[R24] [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293) — 2021.** Priority: high. Depth: abstract screened.

Measures effective data transferred from pretraining using the amount of target data a from-scratch model would need to achieve comparable loss. It provides a useful language for sample-efficiency results and emphasizes distribution proximity. Our initial study should report observed curves and reachable common thresholds before fitting a scaling law; extrapolated equivalent-data claims need separate validation.

**[R25] [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177) — 2024 preprint.** Priority: high. Depth: abstract screened.

Studies how pretraining data size and alignment affect downstream translation, including cases where downstream quality metrics behave differently from cross-entropy. It supports measuring task outcomes rather than assuming a language-modeling loss advantage transfers. Its translation evidence is relevant background, not a guarantee of a universal scaling relationship for instruction learning.

**[R26] [Rewriting Scaling Laws for Synthetic Data with Compute and Data Constraints](https://arxiv.org/html/2610.31867v1) — 2026, v1.** Priority: high. Depth: abstract, introduction, and study outline inspected.

Fits rewriting scaling laws while charging rewrite-generation compute, in both compute-limited and data-limited settings. The preferred synthetic fraction varies with budget and evaluation. This motivates explicit cost accounting and discourages assuming that the best mixture is always all-synthetic or all-natural. Its targets primarily concern rewriting performance; project 4 would instead select mixtures for performance after a future SFT budget.

**Supplementary explainer:** [Sebastian Raschka: when to mask prompt tokens](https://sebastianraschka.com/faq/docs/when-mask-prompt-tokens.html). Direct retrieval was blocked during this review. Keep it as a user-supplied reading link; the masking discussion above relies on R06 and R07, not an independently verified reading of this page.

### Suggested reading order

1. **Establish the novelty boundary:** R01, R02, R10, R11. Write down precisely which later-learning comparisons each already makes.
2. **Define what could explain an apparent effect:** R08, R17–R19, R06–R07. Separate output behavior, optimization, and actual acquisition.
3. **Design the diagnostic task suite:** R21 and R22, then R20. Inspect the code and make small-model feasibility checks before allocating the main runs.
4. **Design schedules and resource comparisons:** R12–R16 and R24–R26. Read the remaining transformation and curriculum papers as alternatives become relevant.

Discovery used the supplied papers, citation-following, and additional OpenAlex searches. Search-service access was uneven, and the bibliography is a focused starting map rather than an exhaustive review through the review date. Before submitting a paper, repeat forward-citation and recent-paper searches for the final, narrower claim.

## 4. Four ranked project ideas

### Project 1 — Transferable skills or selective compression?

**Question.** Does instruction pretraining change later learning because of presentation, or because it selects a different subset of the information in raw documents?

**Competing hypotheses.** Instructions may provide reusable task structure and improve future acquisition. Alternatively, they may mainly improve access to selected knowledge while omitting information useful for future tasks. A third possibility is that apparent differences vanish after controlling formatting and SFT optimization. None is assumed true.

**Minimum convincing experiment.** A controlled 2×2 crossing presentation with information coverage, followed by independent SFT on withheld targets; then paired natural-language comparisons and answer-content controls. Use a coverage-restoration intervention to ask whether omitted information explains a deficit.

**Novelty over nearby work.** R02 already measures further instruction tuning; R11 already introduces future data early. The distinguishing evidence would be a causal decomposition under known support, followed by transfer to new targets rather than simple reuse of pretraining tasks.

**Expected figures.** Transfer curves split by novelty type; a presentation×coverage interaction plot; accuracy before/after the restoration intervention; a retention-versus-acquisition frontier.

**Main costs and failure modes.** Matched data construction, several pretraining seeds, and many downstream branches. Natural-language controls alone cannot fully isolate semantics. A null presentation effect after coverage control would still answer the motivating question, but would not by itself guarantee a publishable contribution.

**Rank: first.** Best combination of a sharp causal question, usable artifacts, and a connection to the original FineInstructions result.

### Project 2 — Does instruction pretraining change the implicit curriculum?

**Question.** Does instructionization merely accelerate a common skill-acquisition sequence, or change which skills and compositions are learned first?

**Minimum convincing experiment.** Evaluate simple and composite skills at dense checkpoints under paired prose/instruction treatments. Then intervene on the exposure of a proposed prerequisite while holding total exposure fixed. Distinguish a global speedup from pairwise ordering changes and altered transfer into composite tasks.

**Novelty over nearby work.** R21 measures stable emergence relationships across existing model families, and R22 studies skill-aware training. Our contribution would be a controlled presentation intervention that changes—or robustly preserves—those relationships, with evidence about later SFT.

**Expected figures.** Skill-by-checkpoint heatmaps, ordering agreement after accounting for overall progress, and prerequisite-intervention effects on composite learning curves.

**Main costs and failure modes.** Dense evaluation and careful handling of floor effects. Threshold crossings depend on prompt format, task difficulty, and checkpoint spacing. Function-vector similarity alone does not show a mechanism. A common ordering with uniformly earlier acquisition favors acceleration, not a changed dependency structure.

**Rank: second.** Use a bounded version as project 1’s mechanistic component, not a second full paper embedded in the initial experiment matrix.

### Project 3 — Which kind of diversity buys future learning?

**Question.** Do many surface templates improve transfer, or do distinct operations and retained knowledge matter more?

**Minimum convincing experiment.** In a generated world, cross low/high surface-template diversity, low/high operation diversity, and narrow/broad fact coverage. Keep training tokens and repetition accounting explicit. Evaluate fresh templates, withheld operations, and new compositions separately. Confirm only the strongest contrast on natural-language subsets.

**Novelty over nearby work.** FineInstructions measures template diversity; R03 studies useful rewriting diversity. The contribution would be evidence separating forms of diversity and predicting which one helps a specified future-learning regime. Template count must not serve as a proxy for semantic diversity without validation.

**Expected figures.** A transfer matrix by novelty type, factorial effects of the three diversity axes, and marginal transfer improvement per additional distinct operation or template family.

**Main costs and failure modes.** The full factorial grows quickly; surface variants can accidentally change difficulty or introduce facts. Natural-language semantic categories require an audited sample and should not be taken as ground truth from an LLM classifier. A null surface-diversity effect would narrow the interpretation of large template banks.

**Rank: third.** Strong follow-up if project 1 suggests that support or task structure explains transfer.

### Project 4 — Optimize pretraining for the model after its next SFT

**Question.** Is the mixture that maximizes immediate performance different from the mixture that maximizes performance after learning a new task?

**Minimum convincing experiment.** Sweep instruction-token fractions 0, 0.25, 0.5, 0.75, and 1 at fixed pretraining budgets. Evaluate held-out target families across several SFT budgets. At the 0.5 fraction, compare constant mixing, raw-then-instruction, and instruction-then-raw schedules with identical total exposures. Cross those orderings with cosine and warmup-stable-decay schedules.

**Novelty over nearby work.** R11 already optimizes early domain exposure, R13 already studies data/LR scheduling, and R26 already predicts rewrite fractions. The distinguishing target would be transfer to unseen tasks after a specified future SFT budget, including out-of-family validation of any prediction.

**Expected figures.** Mixture optima versus future SFT budget; reversals between base and adapted model rankings; predicted versus observed transfer on held-out families.

**Main costs and failure modes.** The most expensive proposal because mixture, schedule, scale, and adaptation budget interact. A smooth curve is not enough evidence for a scaling law. If the same mixture wins everywhere, report that result within tested budgets rather than forcing a crossover narrative.

**Rank: fourth for the initial paper.** Potentially valuable as a later practical/scaling study once the causal comparisons are understood.

## 5. First-study specification

### 5.1 Primary question and estimands

Primary outcome: held-out supervised-transfer performance as a function of unique target examples and target training tokens, after matched pretraining compute. The primary controlled contrasts are the effect of presentation within each coverage condition and the presentation×coverage interaction.

Keep two experiment tracks separate. The generated-world track identifies effects under known semantic support. The natural-language track tests whether a related pattern survives on realistic data. A natural-language FI-versus-raw contrast estimates the effect of the full data treatment, not pure instruction formatting.

### 5.2 Track A: controlled semantic support

Use generated fictional entities and attributes, finite relations, and deterministic operations on strings or retrieved values. Include primitives such as attribute lookup, case conversion, reversal, and relation traversal, with explicit composition graphs. Avoid real-world entity names whose associations could enter through a teacher.

Build each example from an immutable latent record and render it as either declarative prose or an instruction–answer exchange. Within a coverage level, both renderers must communicate the same propositions and transformation examples. For example, “The code of entity zaf is lum” and “What is the code of entity zaf? lum” expose the same fact. Record the latent facts and operations for every rendered sample so equivalence can be checked programmatically. This tests presentation broadly, including conditional ordering; it does not isolate punctuation or role labels alone.

| Arm | Presentation | Retained support |
| --- | --- | --- |
| A1 | Prose | Full eligible fact/operation support |
| A2 | Instruction–answer | Same full support |
| A3 | Prose | Fixed restricted support |
| A4 | Instruction–answer | Exactly the same restricted support as A3 |

Default restricted coverage: retain 50% of eligible fact groups, stratified by relation and operation, using a fixed latent split shared across paired runs. Reserve a targeted omission of one prerequisite family for the restoration experiment. Future-only facts and target compositions are excluded from **all** pretraining arms; “full” means full eligible pretraining support, not access to the future test set.

Use two comparisons rather than pretending that all quantities can be equal simultaneously:

- **Primary:** same total training tokens, model, and updates. Log fact frequencies and unique support, since text lengths may differ.
- **Diagnostic:** same latent-record exposures, with actual tokens and FLOPs reported. This checks whether extra repetitions or serialization length explain the primary result.

Default pilot: approximately 100M parameters and 1B tokens, four arms and three paired seeds. First validate that a single small run can learn the primitives and that train/test generators are disjoint. The task generator, not the natural-language ElementalTask suite, is the primary evaluation at this scale.

**Restoration intervention:** give A3/A4 the omitted prerequisite support in an early or late window containing the same number of tokens; replace an equal amount of background training so total compute is fixed. Measure later composite-task SFT. Compare the timing effect under both constant and decaying LR before calling it evidence of a critical period. Restoration of old omitted facts is a separate diagnostic from acquisition of future-only facts.

### 5.3 Track B: paired natural-language corpora

Start from [fineinstructions/fineinstructions_nemotron](https://huggingface.co/datasets/fineinstructions/fineinstructions_nemotron). The inspected schema includes `warc_record_id`, `text`, `token_count`, `template_id`, `instantiated_instruction`, `answer`, and `synthetic_token_count`. The dataset card describes companion judging files. Verify score alignment before applying filters; do not assume the first visible rows represent the final judged training subset.

Construct four treatments:

| Arm | Training text | Main comparison it supports |
| --- | --- | --- |
| B1 | Original source documents | Conventional raw-document baseline |
| B2 | Instruction and answer pairs | Complete FI-style data treatment |
| B3 | The same answer texts without instructions | Contribution of selected response content and style |
| B4 | 50% raw-document / 50% instruction tokens | Whether preserving some raw coverage helps |

Use the same eligible source-document pool for every arm. If judging excludes a document entirely from B2, exclude it from the paired pool for B1, B3, and B4. Within B1, deduplicate documents by source ID; do not repeat a raw document once for every generated pair unless repetition is an explicitly measured treatment. B3 retains the original answer text rather than introducing a new teacher-written prose paraphrase. Instructions can themselves contain facts, so B2–B3 is not a perfectly content-matched causal contrast.

Partition by source document and near-duplicate group before sampling; keep all renderings of one source together. Group template paraphrases into families before defining held-out template sets. Report that naturally occurring raw documents can already contain questions and answers: “raw” does not mean “instruction-free.”

Default pilot: a shared approximately 500M-parameter decoder, shared tokenizer, 2,048-token context, 20B training tokens per arm, three paired seeds. Share initialization within each seed where architectures match. Sample proportions by tokens, not examples. Use the same loss, packing boundaries, batch-token budget, optimizer, and pretraining schedule; retain configurations and source manifests. Use a common stable recipe selected on a disjoint development corpus before comparing arms.

The first milestone is a one-seed feasibility run and evaluation audit. Do not allocate all 12 runs before checking that transfer evaluations are above floor and below saturation. The original curriculum study excluded models below 410M for poor performance, and 500M at our token budget may still struggle. If natural-language probes remain at floor, revise the development probe set or increase scale before freezing the main study, and document the change.

### 5.4 Independent downstream learning branches

Do not run the following tasks sequentially on one checkpoint; branch from the same starting model for each target. Sequential continual learning would add a separate source of interference.

| Branch | Held-out material | Intended question |
| --- | --- | --- |
| Familiar family | New instances/entities in a known task family | Does the model exploit familiar task structure? |
| New family | A withheld instruction/operation family | Can it learn a different operation or behavior? |
| New composition | Withheld combinations of seen primitives | Do learned skills support new combinations? |
| New facts | Facts absent from all pretraining treatments | How efficiently can it acquire information through SFT? |

The generated-world track can certify absence by construction. In natural language, source/family separation and deduplication reduce overlap but do not prove that an underlying concept was never encountered. Use newly generated fictional facts for a clean new-information branch, and label broader domain-transfer evidence accordingly.

For each selected family, use nested datasets of 128, 512, 2,048, and 8,192 examples, plus the zero-SFT checkpoint. Use the same target examples across pretraining arms and keep separate development and test instances. Report prompt and response lengths, unique examples, actual processed tokens, repetitions, and updates. Equal examples and equal tokens answer different questions when family lengths differ.

Use full-parameter SFT as the primary adaptation method. Reset optimizer state consistently at the start of every downstream branch. Start with a common learning-rate sweep of 1e-5, 3e-5, 1e-4, and 3e-4, with checkpoints over 1, 2, 4, and 8 passes. Select on development data, using the same search allowance for every pretraining arm. If every condition’s optimum is at a grid edge, expand the grid uniformly before final test evaluation. Report both a common recipe chosen on development tasks and per-arm validation-selected results.

Run SFT first from the final pretrained checkpoints. For the primary transfer contrast, additionally branch from checkpoints at 25%, 50%, and 75% of pretraining. This separates a persistent format effect from a difference in the stage of learning at which models were compared.

### 5.5 Evaluation and mechanistic evidence

Use exact-match or executable grading where possible. The generated-world evaluator knows the correct latent answers. For natural-language tasks, report answer-extraction failures separately from content errors. Include a small general instruction-following evaluation and original-domain held-out loss, but do not use judge preference or raw-text perplexity alone as the primary evidence.

Adapt the [ElementalTask suite](https://github.com/KaiserWhoLearns/ElementalTask) for checkpoint diagnostics. Preserve the original distinction between an ICL probe and a supervised-transfer task. Use fresh instances, independent composition splits, and a fixed demonstration protocol for the ICL measurements; do not accidentally count demonstration exposure as SFT.

Save model checkpoints at 0%, 1%, 2%, 5%, and then every 5% of pretraining for the curriculum diagnostics. Treat acquisition times as interval-censored between evaluated checkpoints. For the generated exact-match tasks, use 80% accuracy sustained across two checkpoints as the primary acquisition criterion, with 50% and 90% sensitivity analyses. For heterogeneous natural-language tasks, show complete curves; compare threshold orderings only where tasks attain those thresholds. Tasks that never cross remain censored, not assigned an arbitrary last-place time.

Primary reported statistics:

- Per-family adaptation curves and macro-averages across families, with pretraining-seed variability visible.
- Observed examples/tokens to common accuracy thresholds; label thresholds never reached and avoid extrapolating beyond observed support.
- Area under accuracy versus log processed-SFT-tokens over a common observed range, accompanied by starting scores and full curves. This is a secondary summary, not a replacement for them.
- Retention changes on fixed old-task test sets, including per-example transitions from correct to incorrect and vice versa.
- In Track A, paired presentation contrasts and their interaction with coverage. Bootstrap seed/family units hierarchically where meaningful; individual test examples are not independent training replicates. Three seeds are a pilot minimum, not a guarantee of statistical power.

Mechanistic evidence should first come from support-removal and restoration interventions. Function-vector similarity, representation probes, gradient alignment, and parameter-sensitivity measurements are optional supporting analyses. Do not use successful-example-only representation selection to compare models without reporting how their success subsets differ. Similar trajectories or vectors do not establish that one learned skill causally implements another.

### 5.6 Controls that determine whether the result is interpretable

**Output-format familiarity.** Add a branch in which all models receive the same generic format-adaptation data before target SFT. Use 2M non-target training tokens, disjoint from target facts, families, and test templates. Count these tokens in total adaptation cost and report direct-SFT and format-adapted results separately. A disappearing gap supports an output-access explanation; it does not prove identical internal knowledge.

**Masking and length.** Main training uses unit weights on all non-padding tokens. On the main B1/B2 contrast, test prompt weights 0 and 1 during target SFT for both short- and long-response tasks. Keep the normalization convention explicit: divide weighted token loss by total non-padding tokens for this controlled ablation, so response-token coefficients are unchanged. Conventional active-token normalization changes the effective scale; if also tested, label it separately and retune LR with the same budget. Do not expand to every fractional weight initially.

**Optimization.** A single shared SFT learning rate can favor one initialization. Report it, but require equal-budget tuned comparisons too. If the primary effect survives, repeat the strongest contrast under a second pretraining schedule, warmup-stable-decay, with the same peak LR and a declared decay window; also test a development-selected weight-decay sensitivity. Do not run the full format×schedule×regularization factorial before there is a signal.

**Coverage and quality.** Log source support and exposure frequency rather than treating matched token counts as matched information. Audit sampled instructions and answers for grounding, omissions, instruction-only facts, and truncation. Keep filtering criteria identical across paired treatments. Explicitly separate results from judged and unjudged data if both are used.

**Contamination and model selection.** Freeze source partitions, target families, generators, and evaluation templates before the main runs. Never select mixtures, checkpoints, or learning rates using the final target test set. For a cross-family prediction, hold out entire families from fitting, not merely examples.

### 5.7 Resources and execution order

For rough dense-model training estimates, use $C\approx6ND$, with parameter count $N$ and processed training tokens $D$. These are planning approximations, not measured FLOPs: attention, vocabulary projection, padding, implementation details, and hardware efficiency affect actual cost.

| Phase | Default allocation | Approximate training FLOPs |
| --- | --- | --- |
| Generated-world primary matrix | 100M × 1B tokens × 4 arms × 3 seeds | 7.2e18 |
| Natural-language primary matrix | 500M × 20B tokens × 4 arms × 3 seeds | 7.2e20 |
| Optional larger confirmation | 1.8B × 100B tokens × 2 arms × 3 seeds | 6.48e21 |

These totals exclude restoration runs, exposure-matched diagnostics, alternative schedules, SFT sweeps, and evaluations. Estimate those separately from the actual tokenized task sets. For example, one 500M-parameter SFT run processing 10M tokens is approximately 3e16 FLOPs by the same approximation; branching over many datasets, checkpoints, and rates multiplies this quickly.

Report two cost views for reused synthetic data: **marginal experiment cost**, which can treat the already released corpus as available, and **pipeline cost**, which includes generation, retrieval, judging, and regeneration where those costs can be measured or reliably sourced. Distinguish generator pretraining as an external inherited resource from generation inference. If historical costs are unavailable, say so rather than claiming a complete pipeline efficiency result. No GPU-hour estimate is justified until hardware and measured throughput are known.

Execute in this order:

1. Audit artifacts, source IDs, filtering alignment, and licensing; construct split manifests and run generator/evaluator checks.
2. Validate the controlled-world contrast and natural-language evaluation feasibility on one seed.
3. Freeze the primary matrix and target protocols, then complete the three-seed runs.
4. Run the bounded format, masking, optimization, and restoration controls that address the observed pattern.
5. Confirm only the strongest interpretable contrast at the larger scale. Add more seeds if variability leaves the conclusion unresolved.

Storage, evaluation, data preprocessing, and checkpoint retention are resource drivers even when their FLOPs are small. Retain checkpoint hashes, token/source counts, seeds, configuration files, generation versions, and per-example evaluation outputs.

### 5.8 Decision rules and informative outcomes

| Observation | Supported interpretation / next step |
| --- | --- |
| FI advantage vanishes after generic format adaptation | Much of the measured gap concerns elicitation or response formatting; avoid a broad plasticity claim. |
| Presentation matters at equal coverage and survives tuned SFT | Stronger evidence for a presentation-dependent transfer effect; test skill interventions and scale. |
| Restricted coverage hurts both presentations and restoration repairs it | Information support is a substantial explanation; investigate coverage-preserving instruction generation. |
| The preferred treatment reverses with SFT or pretraining schedule | Report an optimization interaction rather than a universal format ranking. |
| Same ordering, earlier acquisition throughout | Evidence for acceleration, not a changed skill-dependency structure. |
| Prerequisite exposure changes composite learning differently by presentation | Evidence toward a causal curriculum explanation, within the tested task construction. |
| No stable difference under controls | Bound the claim to the tested scales/support; assess whether the causal null and measurement framework are sufficiently informative. |

Advance beyond the pilot only when evaluations are usable, results are reproducible across seeds, and the leading explanation survives the relevant controls. Statistical significance without a useful effect size or a defensible comparison is not a promotion criterion. A favorable result in the generated world alone does not establish the same mechanism in web-trained LLMs.

## 6. What could the eventual paper claim?

A suitable working title is **“What Can an Instruction-Pretrained Model Learn Next? Separating Format, Coverage, and Curriculum.”** It is a question, not a commitment to a positive result.

| Tempting claim | Why it is insufficient | Stronger target |
| --- | --- | --- |
| Pretraining and SFT have the same loss | Established formulation; different distributions still matter | A controlled explanation of how those distributions affect later learning |
| Instruction pretraining improves further SFT | Already investigated by R02 | Transfer under known, withheld semantic support with content and format controls |
| Mixing earlier is better | Overlaps R11–R15; schedule-dependent | State when and why mixture/order helps an unseen future task |
| Millions of templates imply broad knowledge | Template count does not measure retained information | Separate surface diversity, operation diversity, and knowledge coverage |
| Better final scores mean greater plasticity | Starting performance, optimization, and forgetting are confounded | Full acquisition and retention curves under common and tuned adaptation |
| Skill ordering reveals a mechanism | Correlation and designed composition do not prove internal dependence | Intervene on prerequisite exposure and measure downstream consequences |

The intended contribution is a clear answer to a narrower question, supported by reusable paired data and an evaluation protocol. A new sampler or generation algorithm is optional and should follow the evidence rather than be invented to make the work appear methodological.

## 7. Documentation acceptance checklist

- [x] All seven supplied arXiv papers are included, along with the masking explainer.
- [x] Nineteen additional papers are included, for 26 primary-paper entries total.
- [x] Reading depth, evidence, hypotheses, and proposed experiments are distinguished.
- [x] FineInstructions’ released SFT artifacts and the earlier IPT transfer result are acknowledged.
- [x] Four distinct ideas are ranked, with one concrete first study and resource assumptions.
- [x] No novelty, knowledge-breadth, or adaptability claim is inferred solely from shared losses, template counts, or initial scores.
- [x] RL and long-context experiments remain outside the initial scope.
- [x] This deliverable adds documentation only; no training code, bulk datasets, model weights, or experiment results are included.
