# Dataset Feasibility Assessment for DDC Estimation

This document assesses 8 candidate datasets for DDC estimation suitability in econirl. Each dataset is scored against the 6 structural assumptions (A1 Markov through A6 Stationary Transitions) and evaluated for gap-filling value relative to existing examples.

## Summary Table

| Dataset | Domain | Gap Value | DDC Fit | Panel? | Download | Verdict |
|---------|--------|-----------|---------|--------|----------|---------|
| KKBOX Churn | Platform/Tech | HIGH | EXCELLENT | YES (21.5M txns, 2.4M users, median 5 renewals) | Kaggle | INTEGRATE |
| Freddie Mac Loans | Housing/Finance | HIGH | EXCELLENT | Kaggle=cross-section only; monthly panel needs freddiemac.com registration | Kaggle/Manual | DEFER (registration) |
| Expedia Hotel Search | Consumer Search | LOW | MODERATE | Within-session only | Kaggle | DEFER |
| NASA C-MAPSS | Maintenance | LOW | POOR | Run-to-failure only | Zenodo | DROP |
| MovieLens 25M | Platform | LOW | POOR | Sparse, no DDC structure | GroupLens | DROP |
| American Housing Survey | Housing | MODERATE | MODERATE | Single cross-section per file | Census | DEFER |
| Coveo E-Commerce | Consumer Search | LOW | MODERATE | Sample only (34 rows) | GitHub | DROP |
| ICU Sepsis (Chan) | Healthcare | LOW | N/A | Data not public | Paper PDF | DROP |

---

## Dataset 1: KKBOX Subscription Churn

**Source**: Kaggle (WSDM 2017 competition, qmdo97/kkboxdataset mirror)
**Downloaded**: 2.3GB raw data with transactions.csv (1.6GB), members_v3.csv (408MB), user_logs_v2.csv (1.4GB)

**Schema (raw transactions.csv)**:
- 21,547,746 rows, 9 columns
- msno (user hash), payment_method_id, payment_plan_days, plan_list_price, actual_amount_paid, is_auto_renew, transaction_date (YYYYMMDD), membership_expire_date, is_cancel
- 2,363,626 unique users
- Date range: 2015-01-01 to 2017-02-28 (26 months)
- Transactions per user: min=1, median=5, max=71
- 51.4% of users have 5 or more transactions (1.2M users)
- Cancel rate per transaction: 4.0%
- Auto-renew rate: 85.2%
- Dominant plan: 30-day subscription (88% of transactions)

**Members file**: 6.8M members with city, birth date, gender, registration channel, registration date

**DDC Framing**:
- Agent: subscriber (2.4M users, 1.2M with 5+ renewal observations)
- Time: monthly subscription renewal events (median 5 per user over 26 months)
- State: (tenure_bin, payment_method, plan_price_bin, auto_renew_status, city)
- Action: renew (0) or cancel (1)
- Structurally identical to Rust bus replacement (continue/stop decision each period)

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | PASS | Current subscription state (tenure, plan, auto-renew) sufficient |
| A2 Additive Separability | PASS | Plan cost, usage level, tenure are separable utility components |
| A3 IIA/Gumbel | PASS | Binary choice, IIA trivially satisfied |
| A4 Discrete Actions | PASS | Renew or cancel, naturally binary |
| A5 Time Homogeneity | PASS | 26-month window (Jan 2015 to Feb 2017), short enough for stability |
| A6 Stationary Transitions | PASS | Subscription state transitions are stable within observation window |

**Verdict**: INTEGRATE. Fills the completely empty Platform/Tech category. 2.4M users with median 5 monthly renewal transactions is an excellent panel. The renew/cancel framing is a textbook DDC optimal stopping problem. Transaction data provides explicit monthly decision points with prices, plan types, and cancel flags. This is the single strongest new dataset.

---

## Dataset 2: Freddie Mac Single-Family Loan Performance

**Source**: Kaggle mirror of Freddie Mac data (2.6GB, 2008-2018)
**Also available**: Direct from freddiemac.com after registration (full historical data, 55M+ loans)

**Schema**: Monthly loan-level performance data with origination characteristics and monthly payment status.

**DDC Framing**:
- Agent: mortgage borrower (loan_id)
- Time: monthly reporting periods
- State: (LTV bucket, delinquency status, interest rate spread, loan age)
- Action: stay current (0), prepay/refinance (1), default (2)
- Three-action optimal stopping problem with absorbing states (prepay and default are terminal)

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | PASS | Current delinquency + LTV captures relevant state |
| A2 Additive Separability | PASS | Monthly payment, equity position, interest rate are separable |
| A3 IIA/Gumbel | WARN | Prepay and default may have correlated unobservables |
| A4 Discrete Actions | PASS | Pay, prepay, default are naturally discrete |
| A5 Time Homogeneity | WARN | Financial crisis (2008-2012) vs recovery creates regime shifts |
| A6 Stationary Transitions | WARN | Housing market cycles affect LTV transitions |

**Key Strength**: Bajari, Chu, Nekipelov and Park (2013, NBER WP 18850) is a flagship DDC mortgage default paper. This provides a direct replication target. The monthly panel with millions of loans gives enormous statistical power.

**Key Concern**: Neither the preprocessed Kaggle version (148K loans, cross-section) nor the Kaggle PUDB mirror (2.6GB, origination data only) contain monthly payment performance records. The actual monthly performance panel data is only available directly from freddiemac.com/research/datasets/sf-loanlevel-dataset after free registration. This is the data Bajari et al. use.

**Verdict**: INTEGRATE but requires manual registration. Fills the completely empty Housing/Finance category. Mortgage default is one of the most studied DDC applications in economics. The Bajari et al. paper provides a clear replication target. Three-action choice adds complexity beyond binary Rust bus. You must register at freddiemac.com to access the monthly performance data.

---

## Dataset 3: Expedia Hotel Search (ICDM 2013)

**Source**: Kaggle mirror (406MB)
**Downloaded**: train.csv (2.2GB, ~5M rows), test.csv (1.4GB)

**Schema**: 54 columns including srch_id, date_time, prop_id, prop_starrating, prop_review_score, position, price_usd, random_bool, click_bool, booking_bool, plus 8 competitor price comparisons.

**Panel Structure**:
- 40K search sessions in 1M rows
- Each session shows ~30 hotels in ranked position (1 to 40)
- Click rate: 4.5%, booking rate: 2.8%
- 29.7% of impressions have randomized ranking (random_bool=1)

**DDC Framing**:
- Agent: search session (srch_id)
- Time: position in ranked results (sequential scrolling, 1 to 40)
- State: (position, price bucket, star rating, review score, competitor price advantage)
- Action: scroll past (0), click (1), book (2)
- This is a finite-horizon optimal stopping problem

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | PASS | Current position + hotel attributes sufficient |
| A2 Additive Separability | PASS | Price, quality, position are separable utility components |
| A3 IIA/Gumbel | WARN | Hotels at similar positions may be substitutes (nesting) |
| A4 Discrete Actions | PASS | Scroll, click, book are discrete |
| A5 Time Homogeneity | PASS | Single time period, parameters stable |
| A6 Stationary Transitions | PASS | Position increment is deterministic |

**Key Strength**: The random_bool column provides exogenous variation for structural identification. This is analogous to the randomized ranking in Ursu (2018, Marketing Science). The finite-horizon structure makes value function computation tractable.

**Key Weakness**: No persistent user ID for cross-session panel. Each session is independent. This limits the estimation to within-session dynamics only, which is a short horizon (5-40 positions). Also structurally similar to the existing Trivago search DDC example.

**Verdict**: DEFER. Good DDC structure with the randomization bonus, but overlaps with existing Trivago example. Consider integrating after higher-priority datasets (KKBOX, Freddie Mac) are done.

---

## Dataset 4: NASA C-MAPSS Turbofan Degradation

**Source**: Zenodo (12.4MB)
**Downloaded**: 4 sub-datasets (FD001-FD004)

**Schema**: unit_id, cycle, 3 operational settings, 21 sensor readings. Run-to-failure data.

**Panel Structure**:
- FD001: 100 engines, 128-362 cycles each (median 199)
- FD002: 260 engines, 128-378 cycles (median 199)
- FD003: 100 engines, 145-525 cycles (median 220)
- FD004: 249 engines, 128-543 cycles (median 234)

**DDC Framing**:
- Agent: engine unit_id
- Time: operational cycle
- State: degradation level (discretize from 21 sensors via PCA or health index)
- Action: continue operating (0), replace (1)

**Fatal Flaw**: This is RUN-TO-FAILURE data. Every engine runs until it fails. The replacement action is NEVER observed in the data. There are no decision-makers choosing when to replace. The data records degradation trajectories that all end in failure. You cannot estimate a DDC model from data where only one action is ever taken.

**DDC Suitability Scorecard**:

| Assumption | Score | Notes |
|-----------|-------|-------|
| A1 Markov | PASS | Sensor state captures degradation |
| A2 Additive Separability | PASS | Operating cost vs replacement cost |
| A3 IIA/Gumbel | PASS | Binary choice |
| A4 Discrete Actions | PASS | Continue vs replace |
| A5 Time Homogeneity | PASS | Controlled test conditions |
| A6 Stationary Transitions | PASS | Degradation dynamics are stable |
| **DATA** | **FAIL** | **Only one action observed (continue until failure)** |

**Verdict**: DROP. The data cannot support DDC estimation because the replacement decision is not observed. This is a prognostics dataset (predict remaining useful life), not a decision dataset. econirl already has Rust Bus, RDW, and Scania for replacement DDC, all of which actually observe both keep and replace actions.

---

## Dataset 5: MovieLens 25M

**Source**: GroupLens (250MB)
**Downloaded**: 25M ratings, 162K users, 59K movies

**Schema**: userId, movieId, rating (0.5-5.0), timestamp

**Panel Structure**:
- 162K users, median 71 ratings, max 32K ratings
- Date range: 1995-01-09 to 2019-11-21
- 20 genre categories

**DDC Framing Attempt**:
- Agent: userId
- Time: timestamp of rating
- State: user taste profile (cluster from rating history)
- Action: genre of next movie rated

**Fatal Flaws**:
1. No choice set. Users choose from a vast catalog. We do not observe what alternatives they considered.
2. Non-action is unobserved. We see ratings but not the decision not to watch or rate a movie.
3. No state transitions. Rating a comedy does not mechanically change your state in a way that affects your next genre choice.
4. Sparse and irregular. Median user has 71 ratings over years. Many months with no activity.
5. This is a preference revelation dataset, not a dynamic decision dataset.

**Verdict**: DROP. MovieLens has no natural DDC structure. Ratings do not constitute sequential decisions with state transitions and a choice set. This is a recommendation/collaborative filtering dataset, not a structural estimation dataset.

---

## Dataset 6: American Housing Survey 2023

**Source**: Census Bureau (123MB)
**Downloaded**: household.csv (55,669 HH x 1,180 columns), person.csv, mortgage.csv, project.csv

**Schema**: Extremely rich household-level data including tenure (own/rent), housing cost, household demographics, mobility indicators, and 1,180 total variables.

**Panel Structure**:
- 55,669 housing units in 2023
- CONTROL (unit ID) is unique per unit, no duplicates in single wave
- AHS resurveys the same housing units biennially (2021, 2023, 2025...)
- To build a panel, must download and link multiple survey years

**DDC Framing**:
- Agent: household at a housing unit
- Time: biennial survey waves
- State: (tenure type, housing cost bucket, household size, metro status)
- Action: stay (0), move out (1)

**Key Concern**: AHS tracks housing UNITS, not households. When a household moves out, the new occupant at that address is interviewed in the next wave. The moving household disappears from the panel. This means we observe staying but moving households are censored. This is a fundamental problem for DDC estimation of mobility decisions.

**Also**: Single-year download gives a cross-section only. Need 3 or more waves for a meaningful panel, each downloaded separately and linked by CONTROL.

**Verdict**: DEFER. The data is extraordinarily rich (1,180 variables) but requires multi-year linking and has the unit-tracking censoring problem. Lower priority than KKBOX and Freddie Mac, which have clean monthly panels. Could be revisited if a housing mobility DDC is specifically desired.

---

## Dataset 7: Coveo E-Commerce Search Sessions

**Source**: GitHub (SIGIR 2021 Data Challenge)
**Downloaded**: Sample only (34 browsing events, 10 search events in 3 sessions)

**Full Data**: 30M+ browsing events were available during the SIGIR 2021 competition. The full dataset must be requested from Coveo directly and is no longer freely downloadable.

**Verdict**: DROP. Only tiny sample data available. Full data requires contacting Coveo. Even with full data, this overlaps with the existing Trivago search DDC example.

---

## Dataset 8: ICU Sepsis (Chan et al. 2012)

**Source**: Paper PDF downloaded (647KB)
**Data**: Uses Kaiser Permanente Northern California (KPNC) proprietary hospital records

The paper describes an MDP for ICU discharge decisions with state defined by patient acuity scores and actions being discharge vs keep in ICU. The methodology is directly applicable to DDC estimation.

**Verdict**: DROP for data download purposes. The underlying data is proprietary hospital records, not publicly available. econirl already has the Komorowski MIMIC-III sepsis MDP (716 states, 25 actions). The Chan paper is a useful methodological reference but does not provide downloadable data.

---

## Recommendations

### Immediate Integration (Tier 1)

1. **KKBOX Subscription Churn**: Build `datasets/kkbox_churn.py` loader and `environments/kkbox_churn.py` environment. Binary renewal DDC with 2.4M users and median 5 monthly renewal transactions. Fills Platform/Tech gap. Raw transaction data has explicit monthly cancel/renew decisions with prices and plan types.

### Deferred -- Registration Required (Tier 2a)

2. **Freddie Mac Loan Performance**: The Kaggle PUDB mirror is cross-sectional origination data (one row per loan). The monthly performance panel needed for DDC estimation requires free registration at freddiemac.com/research/datasets/sf-loanlevel-dataset. Once registered, build `datasets/freddie_mac.py` with three-action mortgage DDC (pay/prepay/default). Fills Housing/Finance gap. Use Bajari et al. (2013) as replication target.

### Deferred -- Lower Priority (Tier 2b)

3. **Expedia Hotel Search**: Good within-session search DDC with randomized ranking (30% randomized) for structural identification. 40K sessions, ~30 hotels each. Overlaps with Trivago but adds randomization.

4. **American Housing Survey**: Rich housing data (55K HH, 1,180 variables) but needs multi-year linking (download 2019, 2021, 2023 separately) and tracks housing units not households (movers are censored).

### Dropped (Tier 3)

5. **NASA C-MAPSS**: Run-to-failure only, no replacement decisions observed.
6. **MovieLens 25M**: No DDC structure. Preference dataset, not decision dataset.
7. **Coveo**: Sample data only. Full data not freely available.
8. **ICU Sepsis (Chan)**: Proprietary data. Already have MIMIC-III sepsis MDP.
