# Analysis & Modeling Plan

This document sits between the EDA notebook and the modeling notebooks. Its job is
simple: explain, in plain language, why this project does what it does. Not just
"we ran logistic regression, then XGBoost, then SHAP" — but *why that order, why
those specific techniques, and what in the data made each choice necessary rather
than optional.* If you've read the EDA notebook and are about to start reading the
modeling notebooks, this is the bridge that should make every decision in them feel
obvious rather than arbitrary.

---

## Starting point: what the data actually looks like

The dataset has 32,581 loan applications, and about 22% of them ended in default.
That number alone shapes almost everything downstream. If you built a lazy model
that just guessed "this person will repay" for every single applicant, it would be
right 78% of the time — which sounds like a good score, but that model would be
useless, because it would never once catch an actual defaulter. This is the reason
accuracy is never used anywhere in this project. Instead, everything is judged on
how well a model *ranks* risky applicants above safe ones (AUC), how far apart it
manages to separate the two groups (KS), and — later, once a model exists — whether
its predicted probabilities can actually be trusted as real probabilities (Brier
score). These aren't arbitrary metric choices; they're a direct response to the
78/22 split.

A few smaller but still important things turned up in the raw data. There were 165
rows that were exact duplicates of other rows — same applicant, same loan, listed
twice. Left alone, a duplicate could easily end up once in the training data and
once in the test data, which would let the model "cheat" by having effectively seen
the test answer already. So those were dropped before anything else happened. There
were also a handful of rows — seven of them — where the applicant's age was listed
as something like 94, 123, or even 144 years old, and two rows where someone was
recorded as having worked longer than they'd been alive. These are just data-entry
mistakes, not some meaningful signal, so rather than throwing those rows away
entirely (and losing everything else useful about that applicant), the ages were
capped at a sane maximum and the impossible employment lengths were capped relative
to the applicant's age.

Two columns had missing values worth handling carefully: interest rate was missing
for about 9.6% of rows, and employment length was missing for about 2.7%. The
interesting part is *how* those gaps get filled in. It would be lazy to just fill
every missing interest rate with the overall average interest rate across the whole
dataset. But it turns out interest rate is almost entirely determined by loan grade
— they correlate at 0.93, which is about as strong as two different columns ever
correlate with each other in real data. So instead of a flat average, each missing
interest rate is filled in using the average interest rate *for that applicant's
specific grade*. A grade-A applicant's missing rate gets filled with the typical
grade-A rate, not some blended average that includes grade-G borrowers paying much
higher rates. That's a small detail, but it's the kind of small detail that adds up
to a model that's actually trustworthy rather than one that just happens to score
well.

One more thing worth mentioning: there's a column called `loan_percent_income`,
and it turns out to be almost exactly equal to `loan_amnt` divided by
`person_income` — the correlation between the two is 0.999, meaning it's not "a
similar idea," it's the literal same number already sitting in the dataset. This
matters because it explains why this project doesn't do much manual feature
engineering. In a lot of tutorials you'll see people proudly constructing new ratio
columns like "loan amount over income" — but here, that column already exists.
Building it again would just be adding a duplicate column dressed up as new work.

---

## The single biggest thing EDA found, and why it changes the whole plan

By far the most important discovery in EDA was how strongly `loan_grade` predicts
default on its own. Applicants with grade A default about 10% of the time. By grade
D that's already up to 59%. By grade G, it's 98%. That's not a gentle trend — it's
almost a light switch. Once you know someone's grade, you know most of what you
need to know about whether they'll default.

This is a legitimate feature — a bank assigns a grade to an applicant *before*
deciding whether to lend to them, using information they already have at that
point, so it's not "cheating" by looking into the future. But it does create a real
risk for a project like this one: if you're not careful, the whole exercise quietly
turns into "build a model that predicts the lender's own grade," which is a much
less interesting and much less honest thing to claim you built. So this finding
directly caused a deliberate decision, made before any serious modeling started:
train a second version of the model with `loan_grade` (and its close cousin,
interest rate) removed entirely, and see how much predictive power is left in just
the ordinary applicant details — income, age, how long they've been employed, why
they want the loan, and so on. That "ablation" model isn't an afterthought added
because the numbers looked suspicious later — it was planned from the moment this
pattern showed up in EDA, specifically to keep the final claims honest.

---

## Why logistic regression comes first, before anything fancier

Before touching a more powerful model, the project fits the simplest reasonable
one: logistic regression, with a setting that tells it to pay extra attention to
the minority (default) class since the data is imbalanced. There are two real
reasons for starting here rather than jumping straight to XGBoost.

First, it sets a floor. If a more complex model can't beat this simple one, that's
not a footnote — it's a sign something is broken, and it needs to be caught before
anyone gets excited about a fancier model's score.

Second, and less obvious: a linear model's coefficients are a cheap way to sanity
check the whole pipeline. Every coefficient should point in a direction that makes
real-world sense — higher interest rate should push risk up, a worse grade should
push risk up, and so on. If one of those came back backwards, that would be a strong
hint that something upstream is broken — a mislabeled column, a leaked feature,
something. A tree-based model like XGBoost doesn't hand you anything this easy to
eyeball, so doing this check with the simple model first, while it's cheap and
fast, is worth doing even though the simple model isn't the one that ends up used.

Given how strong `loan_grade` turned out to be, it wasn't a surprise that this
simple model scored *unusually well* for a first attempt — an AUC of 0.87, which is
higher than a typical "quick baseline" score in this kind of project. That's not
because logistic regression is secretly powerful; it's because the data itself is
unusually easy to separate, thanks to that one dominant feature.

---

## Why XGBoost, and why the imbalance-handling technique wasn't just picked and assumed

XGBoost was chosen as the main model because the features here plausibly interact
with each other in ways a straight line can't capture — grade combined with income
might matter differently than either one alone, for instance — and tree-based
models are naturally good at picking up on that kind of interaction without anyone
having to hand-engineer it.

For handling the imbalance, there were two realistic options: tell the model to
weight mistakes on defaulters more heavily during training (`scale_pos_weight`), or
generate synthetic examples of defaulters to balance the training data out before
fitting the model (SMOTE). The plan going in was not "use whichever one sounds more
sophisticated" — it was to actually train both and compare them honestly on the
same held-out data. That comparison matters, because it's easy to assume a
fancier-sounding technique like SMOTE must be better, when in practice it doesn't
always help and sometimes makes probabilities harder to trust. In this case, the
two came out very close — `scale_pos_weight` scored an AUC of 0.9459 versus SMOTE's
0.9442 — and since `scale_pos_weight` also doesn't require inventing any fake data,
it was the clear, well-justified choice rather than a coin flip.

The ablation model described earlier — the one without grade or interest rate —
was trained alongside this comparison, and it came in at an AUC of 0.909. That's a
real, meaningful drop from the full model's 0.946, but it's still a strong score,
which is the honest, two-sided finding: yes, the model leans heavily on grade, but
no, it isn't *only* grade — there's genuine signal in the ordinary applicant
details too, just less of it.

---

## Why WOE/IV analysis still happens even though there are only 11 features

Weight-of-Evidence and Information Value are classic credit-scoring techniques, and
in a lot of projects their main job is trimming down a huge pile of columns to a
manageable handful. That's not really needed here, since there are only 11
features to begin with. But the analysis still earns its place for two other
reasons.

One, it's simply the vocabulary of the field — a credit risk project that never
produces an IV table is missing something a reviewer familiar with the domain would
expect to see. Two, and more usefully, it acts as an independent second opinion on
the exact same question the ablation model was built to answer: which features
actually matter, and is anything suspiciously too strong to be trusted? The IV
numbers came back completely consistent with everything found so far —
`loan_percent_income`, `loan_grade`, `loan_int_rate`, and `person_income` all
landed in the "very strong, double-check this isn't leakage" range, and having
already checked it via the ablation model, that strength could be explained rather
than treated as a red flag. Meanwhile `person_age` and how long someone's had a
credit history came back nearly useless by this measure too, which matches what a
simple correlation check in EDA already suggested. Three different ways of asking
"what matters here" all gave the same answer, which is a much stronger basis for
trusting the result than any one of them alone.

---

## Why the hyperparameter tuning used Optuna, and why it was run the way it was

Tuning a model like XGBoost means picking values for seven or so different
settings — how many trees, how deep they go, how much of the data each tree sees,
and so on. Trying every possible combination of seven settings the traditional way
(grid search) explodes into an unreasonable number of combinations very quickly.
Optuna instead tries a batch of combinations, learns from how well each one did,
and uses that to make smarter guesses about where to look next — so it finds good
settings in far fewer attempts, and it can also notice early when a particular
attempt is clearly not going to work out and abandon it before wasting time
finishing it.

Since this machine has 10 CPU cores and the dataset is small enough that training
one model only takes a couple of seconds, it made sense to run several tuning
attempts side by side rather than one at a time — five attempts running at once,
each one using two of the ten cores, which uses the whole machine without any one
attempt starving the others of CPU time. That setup ran 80 full attempts in about
13 and a half seconds.

Given how much of the model's performance is already explained by a couple of very
strong features, it made sense not to expect tuning to change much — most of the
"work" here is really just "learn the grade signal correctly," and there's only so
much a model's internal settings can improve on that. That's exactly what
happened: the tuned model scored an AUC of 0.9484 versus 0.9459 for the untuned
version — a small, genuine improvement, reported as the small improvement it
actually is rather than dressed up as more than it was.

---

## Why calibration is treated as its own separate step

AUC and KS both measure whether a model *ranks* risky people above safe people
correctly. Neither one cares whether the actual number the model outputs — say,
"this person has a 30% chance of default" — is a number you can actually trust. A
model could output 0.001 and 0.002 for two applicants and get exactly the same AUC
as a model that output 0.10 and 0.90 for the same two people, as long as the
riskier one always scores higher. But if a bank wants to multiply a predicted
probability by a loan amount to estimate how much money it stands to lose, the
actual number matters enormously, not just the ranking.

This is a real concern here specifically because of the technique chosen earlier —
telling the model to weight defaulters more heavily during training is known to
distort the raw probabilities it outputs, pushing them toward extremes. So a
separate correction step, called calibration, was always part of the plan rather
than something bolted on after noticing a problem.

Calibration needs its own slice of data to work properly — not the training data,
which the model has already memorized and would make it look artificially well
calibrated already, and not the test data, which needs to stay completely untouched
so the final reported numbers are trustworthy. That's exactly why the data was
split into three parts (60% train, 20% validation, 20% test) instead of the more
common two. The validation slice, which had already been used once to help tune the
model, gets reused here to fit the calibration — which is normal practice, not
double-dipping, because it's still never used to produce the final score anyone
reports.

After calibration, the model's Brier score (a measure of how far off its predicted
probabilities are from reality) improved from 0.062 to 0.051 — about 17% better.
The ranking score (AUC) barely moved at all, drifting by 0.0002, which is exactly
what should happen: calibration is only supposed to fix the *scale* of the
predictions, not which applicants get ranked above which. The tiny drift that did
happen has a specific, boring explanation — the calibration technique used here can
occasionally give two different applicants the exact same adjusted score even
though their raw scores were slightly different, and when that happens, tiny
inconsistencies like this can appear in how ties get counted. It's expected
behavior for this specific technique, not a mistake, and it was checked with an
actual test in the code rather than just eyeballed and assumed fine.

---

## Why SHAP instead of the model's built-in feature importance

XGBoost can tell you, on its own, which features it used the most — but that
built-in number has two real weaknesses. It only tells you about the model as a
whole, never about one specific applicant, and it tends to unfairly favor columns
that have lots of different possible values just because they give the model more
opportunities to split on them.

SHAP fixes both problems. It can explain a single prediction — "here's exactly why
*this* applicant was scored as high risk, and by how much each piece of their
application contributed" — and it does so in a mathematically consistent way. That
distinction matters a lot for this specific use case: in real lending, when someone
is turned down for a loan, many places legally require the lender to explain why.
"The model said no" isn't an acceptable answer; "your income and your interest
rate were the two biggest factors" is. SHAP is what makes that second kind of
answer possible.

It also served as a third independent check on the whole "is this model just
learning the lender's grade" question that's been running through this entire
document. SHAP's ranking of the most important features — income, interest rate,
and the loan-to-income ratio — lines up with what the IV analysis found and what
the ablation model implied. Three separate methods, using three different kinds of
math, all pointing at the same conclusion is about as much confidence as this kind
of analysis can reasonably offer.

---

## Turning the model into an actual decision, not just a score

An AUC number doesn't tell anyone what to actually do. So the last analytical step
was to sort every test applicant by their calibrated risk score, split them into
ten roughly equal groups from safest to riskiest, and look at what the real default
rate was in each group. If the model is any good, the riskiest group should have a
dramatically higher real-world default rate than the safest group — and it did:
the riskiest group defaulted 100% of the time, and the safest group defaulted well
under 1% of the time.

That grouping also surfaced a genuine bug worth mentioning honestly rather than
quietly fixing and pretending it never happened. So many applicants in the riskiest
group had nearly identical, extremely high risk scores (grade-G-type applicants
clustered right up near "certain to default") that the standard way of splitting
data into ten equal groups actually only managed to produce nine distinct groups —
there weren't enough different score values left to make a tenth. The first version
of the policy analysis didn't account for this and ended up accidentally rejecting
nobody at all when it was supposed to reject the riskiest group. That got caught,
explained, and fixed properly rather than papered over.

Once fixed, the actual finding is a genuinely useful, concrete business result: if
a lender rejected just the single riskiest group of applicants out of the ten, the
default rate among everyone still approved would drop from about 22% down to about
14% — a 38% reduction in bad loans — while still approving roughly 90% of all
applicants. That's the kind of sentence that actually means something to a business
person, in a way that "the AUC is 0.95" never quite does on its own.

---

## What wasn't done, and why that's an honest gap rather than an oversight

One thing this project deliberately does not attempt is what's called
out-of-time validation — testing the model on data from a *later* time period than
it was trained on, to simulate how it would behave in the real world as time moves
forward and conditions change. That's genuinely important for a real deployment,
because lending conditions and the kinds of people applying for loans shift over
time in ways a random shuffle of the data can't capture. It isn't done here for a
simple reason: this dataset doesn't include any date or timestamp at all, not even
an imperfect stand-in for one. Rather than inventing a fake ordering just to be able
to claim this step was done, the honest choice is to state plainly that it wasn't
possible here, and that a real production version of this model would need it once
real, timestamped data became available.

---

## The whole chain, one sentence each

- The data is imbalanced, so every metric used measures ranking or calibration
  quality, never plain accuracy.
- A couple of columns had a small number of physically impossible values, fixed by
  capping them rather than discarding the rows.
- Missing interest rates are filled in using the typical rate for that applicant's
  grade, because grade and interest rate are almost the same information.
- No new ratio features were engineered, because the one obvious ratio
  (loan-to-income) already existed in the data.
- Loan grade turned out to predict default almost by itself, which is why a second
  model without grade was trained from the start, to make sure the project wasn't
  just re-learning the lender's own label.
- A simple logistic regression was trained first, both to set a floor to beat and
  to sanity-check the pipeline through its coefficients.
- XGBoost was chosen as the main model for its ability to pick up feature
  interactions, and the two competing imbalance-handling techniques were actually
  compared rather than one being assumed better.
- WOE/IV analysis was still done despite the small number of features, both because
  it's standard practice in this field and because it served as an independent
  second opinion on which features actually matter.
- Hyperparameter tuning used Optuna, run in parallel to make use of the available
  CPU cores, and produced a small, honestly-reported improvement.
- Calibration was treated as its own step, using a third slice of data, because the
  imbalance-handling technique chosen earlier is known to distort raw probabilities.
- SHAP was used over the model's built-in importance because it can explain
  individual decisions, which real lending decisions legally require, and because
  it offered a third independent check on the grade-dominance question.
- The final model's scores were translated into an actual lending policy with a
  real, quantified trade-off, and a genuine bug found along the way was fixed and
  explained rather than hidden.
- Out-of-time validation was skipped and openly acknowledged as a gap, because the
  data simply doesn't contain the information needed to do it honestly.
