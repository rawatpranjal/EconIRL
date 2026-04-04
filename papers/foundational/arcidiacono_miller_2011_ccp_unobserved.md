## CCP Estimation of Dynamic Discrete Choice Models with Unobserved Heterogeneity ∗

Peter Arcidiacono

Robert A. Miller

Duke University

Carnegie Mellon University

April 26, 2011

## Abstract

We adapt the Expectation-Maximization (EM) algorithm to incorporate unobserved heterogeneity into conditional choice probability (CCP) estimators of dynamic discrete choice problems. The unobserved heterogeneity can be time-invariant or follow a Markov chain. By developing a class of problems where difference in future value terms depend on few conditional choice probabilities, we extend the class of dynamic optimization problems where CCP estimators provide a computationally cheap alternative to full solution methods. Monte Carlo results confirm that our algorithms perform quite well, both in terms of computational time and in the precision of the parameter estimates.

Keywords: dynamic discrete choice, unobserved heterogeneity

∗ We thank Victor Aguirregabiria, Esteban Aucejo, Lanier Benkard, Jason Blevins, Paul Ellickson, George-Levi Gayle, Joe Hotz, Pedro Mira, three anonymous referees and the co-editor for their comments. We have benefited from seminars at UC Berkeley, Duke University, University College London, The Ohio State University, University of Pennsylvania, Stanford University, University of Texas, University of Virginia, Washington University, University of Wisconsin, IZA, Microeconometrics Conferences at the Cowles Foundation, the MOVE Conference at Universitat Autnoma de Barcelona and the NASM of the Econometric Society. Andrew Beauchamp, Jon James, and Josh Kinsler provided excellent research assistance. Financial support was provided for by NSF grants SES-0721059 and SES0721098.

## 1 Introduction

Standard methods for solving dynamic discrete choice models involve calculating the value function either through backwards recursion (finite-time) or through the use of a fixed point algorithm (infinite-time). 1 Conditional choice probability (CCP) estimators, originally proposed by Hotz and Miller (1993), provide an alternative to these computationally-intensive procedures by exploiting the mappings from the value functions to the probabilities of making particular decisions. CCP estimators are much easier to compute than full solution methods and have experienced a resurgence in the literature on estimating dynamic games. 2 The computational gains associated with CCP estimation give researchers considerable latitude to explore different functional forms for their models.

In order to implement CCP estimators, two things are necessary. First, the researcher must know how to formulate the value function-or the differenced value function across two choices-as a function of the conditional choice probabilities. These formulations depend upon the distribution of the structural errors and, except in special cases, forming choice paths far out into the future. Second, CCP estimators require calculating choice probabilities conditional on all state variables. Calculating the conditional choice probabilities can be difficult when some of the state variables are unobserved.

Our first contribution is to broaden the class of models where CCP estimation can be implemented without resorting to matrix inversion or simulation. We prove that the expected value of future utilities from optimal decision making can always be expressed as functions of the flow payoffs and conditional choice probabilities for any sequence of future choices, optimal or not. When two choice sequences with different initial decisions lead to the same distribution of states after a few periods, we say there is finite dependence, generalizing Altug and Miller (1998). In such cases the likelihood of a decision can be constructed from current payoffs and conditional choice probabilities that occur a few periods into the future.

Key to exploiting finite dependence, however, is knowing the mappings between the conditional choice probabilities and the difference in the payoffs between two choices. These mappings depend on

1 The full solution or nested fixed point approach for discrete dynamic models was developed in Miller (1984), Pakes (1986), Rust (1987) and Wolpin(1984), and further refined by Keane and Wolpin (1994, 1997).

2 Aguirregabiria and Mira (2010) have recently surveyed the literature on estimating dynamic models of discrete choice. For developments of CCP estimators that apply to dynamic games, see Aguirregabiria and Mira (2007), Bajari, Benkard, and Levin (2007), Jofre-Bonet and Pesendorfer (2003), Pakes, Ostrovsky, and Berry (2007), and Pesendorfer and Schmidt-Dengler (2008).

the distribution of the structural errors. We show how to obtain the mappings when the structural errors follow any generalized extreme value distribution, substantially broadening the class of error structures that are easily adapted to CCP estimation.

Our second contribution is developing CCP estimators that are capable of handling rich classes of unobserved heterogeneity where there are a finite number of unobserved states. 3 Accounting for unobserved heterogeneity, and therefore dynamic selection, is important to many economic problems and has been a standard feature of dynamic discrete choice models in labor economics. 4 Our estimators can readily be adapted to cases where the unobserved state variables are time-invariant, such as is standard in the dynamic discrete choice literature, as well as to cases where the unobserved states transition over time.

To operationalize our estimators, we modify the Expectations-Maximization (EM) algorithm, and in particular its application to sequential estimation as developed in Arcidiacono and Jones (2003), to include updates of the conditional choice probabilities. The EM algorithm iterates on two steps. In the expectations step, the conditional probability of each observation being in each unobserved state is calculated given the data and the structure of the model. In the maximization step, the unobserved states are treated as observed, with the conditional probabilities of being in each unobserved state used as weights. Because the unobserved states are treated as known in the second step of the EM algorithm, we show that there are natural ways of updating the CCP's in the presence of unobserved states. Since the EM algorithm requires solving the maximization step multiple times, it is important that the maximization step be fast. Hence, it is the coupling of CCP estimators-particularly those that exhibit finite dependence-with the EM algorithm which allows for large computational gains despite having to iterate.

We further show how to modify our algorithm to estimate the distribution of unobserved heterogeneity and the conditional choice probabilities in a first stage. The key insight is to use the empirical distribution of choices-rather than the structural choice probabilities themselves-when updating the conditional probability of being in each unobserved state. The estimated probabilities of being in particular unobserved states obtained from the first stage are then used as weights when

3 An alternative is to have unobserved continuous variables. Mroz (1999) shows that using finite mixtures when the true model has a persistent unobserved variable with continuous support yields similar estimates to the case when the unobserved variable is treated as continuous in estimation. For Bayesian approaches to this issue, see Imai, Jain, and Ching (2009) and Norets (2009).

4 For example, see Miller (1984), Keane and Wolpin (1997, 2000, 2001), Eckstein and Wolpin (1999), Arcidiacono (2005), Arcidiacono, Sloan, and Sieg (2007), and Kennan and Walker (2011).

estimating the second stage parameters, namely those parameters entering the dynamic discrete choice problem that are not part of the first stage estimation. We show how the first stage of this modified algorithm can be paired with non-likelihood based estimators proposed by Hotz et al (1994) and Bajari et al (2007) in the second stage. Our analysis complements their work by extending their applicability to unobserved time dependent heterogeneity.

We illustrate the small sample properties of our estimator using a set of Monte Carlo experiments designed to highlight the wide variety of problems that can be estimated with the algorithm. The first is a finite horizon version of the Rust bus engine problem with permanent unobserved heterogeneity. Here we compare computational times and the precision of the estimates with full information maximum likelihood. We further show cases where estimation is only feasible via conditional choice probabilities such as when the time horizon is unknown or when there are time-specific parameters and the data stop short of the full time horizon. The second Monte Carlo is a dynamic game of firm entry and exit. In this example, the unobserved heterogeneity affects the demand levels for particular markets which and, in turn, the value or entering or remaining in the market. The unobserved states are allowed to transition over time and the example explicitly incorporates dynamic selection. We estimate the model using the baseline algorithm as well as the two-stage method. For both sets of Monte Carlos, the estimators perform quite well both in terms of the precision of the estimates as well as computational time.

Our contributions build on some of the points made in the literature on estimating dynamic games. Bajari, Benkard and Levin (2007) build off the approach of Hotz et al. (1994), estimating reduced form policy functions in order to forward simulate the future component of the dynamic discrete choice problem. In principle, their method can be used for any distribution of the structural errors. In practice, this is difficult because the probabilities associated with particular choice paths vary with the correlation parameters. Aguirregabiria and Mira (2007) show how to incorporate permanent unobserved heterogeneity into stationary dynamic games. Their method requires inverting matrices multiple times where the matrices are dimensioned by the number of states. 5 Further, their estimator is restricted to the case where the unobserved heterogeneity only affects the payoff func-

5 Kasahara and Shimotsu (2008) propose methods to weaken the computational requirements of Aguirregabiria and Mira (2007), in part by developing a procedure of obtaining non-parametric estimates of the conditional choice probabilities in a first stage. Hu and Shum (2010b) take a similar approach while allowing the unobserved states to transition over time. Buchinsky et al (2005) use the tools of cluster analysis to incorporate permanent unobserved heterogeneity, seeking conditions on the model structure that allow them to identify the unobserved type of each agent as the number of time periods per observation grows.

tions. This limits the researcher's ability to account for dynamic selection by adopting a selection on observables approach to the transitions of the state variables.

The techniques developed in this paper are being used to estimate structural models in environmental economics, labor economics, industrial organization, and marketing. Bishop (2008) applies the reformulation of the value functions to the migration model of Kennan and Walker (2011) to accommodate state spaces that are computationally intractable using standard techniques. Joensen (2009) incorporates unobserved heterogeneity into a CCP estimator of educational attainment and work decisions. Beresteanu, Ellickson, and Misra (2010) combine our value function reformulation with simulations of the one-period ahead probabilities to estimate a large scale discrete game between retailers. Finally, Chung, Steenburgh, and Sudhir (2009), Beauchamp (2010), and Finger (2008) use our two-stage algorithm to obtain estimates of the unobserved heterogeneity parameters in a first stage, the latter two applying the estimator in a games environment.

The rest of the paper proceeds as follows. Section 2 uses Rust's bus engine problem (1987) as an example of how to apply CCP estimation with unobserved heterogeneity. Section 3 sets up the general framework for our analysis. Section 3 further shows that, for many cases, the differences in conditional value functions only depend upon a small number of conditional choice probabilities and extends the classes of error distributions that can easily be mapped into a CCP framework. Section 4 develops the estimators, while Section 5 derives the algorithms used to operationalize them. Section 6 shows how the parameters governing the unobserved heterogeneity can sometimes be estimated in a first stage. Section 7 reports a series of Monte Carlos conducted to illustrate both the small sample properties of the algorithms as well as the broad classes of models that can be estimated using these techniques. Section 8 concludes. All proofs are in the appendix.

## 2 Motivating Example

To motivate our approach, we first show how the tools developed in this paper apply to the bus engine example considered by Rust (1987) when unobserved heterogeneity is present. This example highlights several features of the paper. First, we show how to characterize the future value termor more precisely the difference in future value terms across the two choices- as a function of just the one period ahead probability of replacing the engine. Second, we show how to estimate the model when there is time-invariant unobserved heterogeneity. 6 In later sections, we extend the estimation to include more general forms of unobserved heterogeneity as well as showing generally

6 Kasahara and Shimotsu (2009) provide conditions under which this model is identified.

how to characterize differences in future value terms as functions of only the conditional choice probabilities from a few periods ahead.

## 2.1 Set up

In each period t ≤ ∞ Harold Zurcher decides whether to replace the existing engine of a bus by choosing d 1 t = 1, or keep it for at least one more period by choosing d 2 t = 1, where d 1 t + d 2 t = 1. The current period payoff for action j depends upon how much mileage the bus has accumulated since the last replacement, x t ∈ { 1 , 2 , . . . } , and the brand of the bus, s ∈ { 1 , . . . , S } . It is through s that we bring unobserved heterogeneity into the bus replacement problem; both x t and s are observed by Zurcher but the econometrician only observes x t .

Mileage advances one unit if Zurcher keeps the current engine and is set to zero if the engine is replaced. Thus x t +1 = x t +1 if d 2 t = 1 and x t +1 = 0 if d 1 t = 1. There is a choice-specific transitory shock, /epsilon1 jt , that also affects current period payoffs and is independently distributed Type 1 extreme value. The current period payoff for keeping the engine at time t is given by θ 1 x t + θ 2 s + /epsilon1 2 t , where θ ≡ { θ 1 , θ 2 } is a set of parameters to be estimated. Since decisions in discrete choice models are unaffected by increasing the payoff to all choices by the same amount, we normalize the current period payoff of the first choice to /epsilon1 1 t . This normalization implies θ 1 x t + θ 2 s + /epsilon1 2 t -/epsilon1 1 t measures, for a brand s bus in period t, the cost of maintaining an old bus engine for another period, net of expenditures incurred by purchasing, installing and maintaining a new bus engine.

Zurcher takes into account both the current period payoff, as well as how his decision today will affect the future, with the per-period discount factor given by β . He chooses d 1 t (and therefore d 2 t ) to sequentially maximize the expected discounted sum of payoffs:

Let V ( x t , s ) denote the ex-ante value function at the beginning of period t . 7 It is the discounted sum of current and future payoffs just before /epsilon1 t ≡ { /epsilon1 1 t , /epsilon1 2 t } is realized and before the decision at t is made, conditional on making optimal choices at t and every future period, when the bus is brand s , and the mileage is x t . We also define the conditional value function for choice j as the current period payoff of choice j net of /epsilon1 jt plus the expected future utility from Zurcher behaving optimally in the future:

<!-- formula-not-decoded -->

7 Since the optimal decision rule is stationary, subscripting by t is redundant.

<!-- formula-not-decoded -->

Let p 1 ( x, s ) denote the conditional choice probability (CCP) of replacing the engine given x and s . The parametric assumptions about the transitory cost shocks imply:

<!-- formula-not-decoded -->

## 2.2 CCP representation of the replacement problem

Rust (1987) showed the conditional value function for keeping the engine, defined in the second line of equation (2 . 2), can be expressed as:

<!-- formula-not-decoded -->

where γ is Euler's constant. Multiplying and dividing the expression inside the logarithm of equation (2 . 4) by exp [ v 1 ( x +1 , s )], yields:

<!-- formula-not-decoded -->

where the last line follows from equation (2.3). Equation (2.5) shows that the future value term in the replacement problem can be expressed as the conditional value of replacing at mileage x + 1 plus the probability of replacing the engine when the mileage is x +1. Applying the same logic to the conditional value function for engine replacement yields:

<!-- formula-not-decoded -->

Recall that replacing the engine resets the mileage to zero. Equation (2.2) then implies that

<!-- formula-not-decoded -->

for all x. Exploiting this property, we difference equations (2 . 5) and (2 . 6) to obtain: 8

<!-- formula-not-decoded -->

Substituting equation (2 . 7) into equation (2 . 3) implies that the probability of replacing the engine , p 1 ( x, s ) , can be expressed a function of the flow payoff of running the engine, θ 1 x + θ 2 s, the discount factor, β, and the one-period-ahead probabilities of replacing the engine, p 1 (0 , s ) , and p 1 ( x +1 , s ) .

/negationslash

8 Note the conditional value functions for period t + 1 do not cancel if the future value terms are written with respect to the second choice because v 2 ( x +1 , s ) = v 2 (0 , s ) .

## 2.3 Estimation

To estimate the model, we develop an algorithm that combines key insights from two literatures. The first is the literature on CCP estimation when there is no unobserved heterogeneity. In this literature, estimates of the conditional choice probabilities are obtained in a first stage and substituted into a second stage maximization. The second is the literature on the Expectation-Maximization (EM) algorithm, which provides a way of estimating dynamic discrete choice models when unobserved state variables are present. As we will show, the EM algorithm can be modified to accommodate CCP estimation.

Consider a sample of N buses over T time periods where all buses begin with zero mileage. The key insight of Hotz and Miller (1993) is that, when both x and s are observed variables, we can substitute a first stage estimate, ̂ p 1 ( x, s ), for p 1 ( x, s ) in (2.7), say ̂ p 1 ( x, s ). 9 Next, we substitute this expression into equation (2.3) to obtain the likelihood of replacing the engine given the first-stage conditional choice probabilities. Writing ̂ p 1 as the vector of conditional choice probabilities and d nt ≡ { d 1 nt , d 2 nt } , the likelihood contribution for bus n at time t is then given by:

To illustrate how CCP estimation might be amenable to the EM algorithm, we first demonstrate how to proceed in the infeasible case where s is unobserved but ̂ p 1 ( x, s ) is known. Let π s denote the population probability of being in state s . Integrating the unobserved state out of the likelihood function, the maximum likelihood (ML) estimator for this version of the problem is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Denote by d n ≡ ( d n 1 , . . . , d n T ) and x n ≡ ( x n 1 , . . . , x n T ) the full sequence of choices and mileages observed in the data for bus n . Conditioning on x n , ̂ p 1 , and θ, the probability of observing d n is the expression inside the logarithm in (2 . 9), while the joint probability of s and d n is the product of all the terms to the right of the summation over s . Given the ML estimates ( ̂ θ, ̂ π ) and using Bayes' rule, we can calculate q ns , the probability n is in unobserved state s as:

9 One candidate is a bin estimator where p 1 ( x, s ) is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By definition, ̂ π s then satisfies:

The EM algorithm is a computationally attractive alternative to directly maximizing (2.9). At the m th iteration, given values for θ ( m ) and π ( m ) , update q ( m +1) ns by substituting θ ( m ) and π ( m ) in for ̂ θ and ̂ π in equation (2.10). Next, update π ( m +1) by substituting q ( m +1) ns for ̂ q ns in equation (2 . 11). Finally, obtain θ ( m +1) from:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ ̂ We now show how to estimate the structural parameters when both s is unobserved and ̂ p ( x, s ) is unknown by building on the features of the estimators discussed above. We modify the EM algorithm, so that instead of just updating θ ( m ) , π ( m ) and q ( m ) ns at the m th iteration, we also update the conditional choice probabilities, p ( m ) 1 ( x, s ).

Note that at the maximization step q ( m +1) ns is taken as given, and that the maximization problem is equivalent to one where s n is observed and q ( m +1) ns are population weights. Note further that the maximum likelihood estimator at the maximization step is globally concave and thus very simple to estimate. Under standard regularity conditions ( θ ( m ) , π ( m ) ) converges to the ML estimator ( θ, π ) . 10

One way of updating p ( m ) 1 ( x nt , s n ) falls naturally out of the EM algorithm. To see this, first note that we can express p 1 ( x, s ) as:

<!-- formula-not-decoded -->

Applying the law of iterated expectations to both the numerator and the denominator, and using the fact that d nt is a component of d n , implies (2 . 13) can be written as:

<!-- formula-not-decoded -->

But the inner expectation in (2.14) is actually q ns as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It now follows that:

In the algorithm defined below we replace (2 . 16) with sample analogs in order to update the conditional choice probabilities.

10 See Dempster, Laird, and Rubin (1977) and Wu (1983) for details.

Our algorithm begins by setting initial values for θ (1) , π (1) , and p (1) 1 . Estimation then involves iterating on four steps where the m th iteration follows:

Step 1 From (2.10), compute q ( m +1) ns as:

<!-- formula-not-decoded -->

Step 2 Using q ( m +1) ns compute π ( m +1) s according to:

<!-- formula-not-decoded -->

Step 3 Using q ( m )+1 ns update p ( m +1) 1 ( x, s ) from:

<!-- formula-not-decoded -->

Step 4 Taking q ( m +1) ns and p ( m +1) 1 ( x nt , s n ) as given, obtain θ ( m +1) from:

<!-- formula-not-decoded -->

Note that Step 3 is a weighted average of decisions to replace conditional on x, where the weights are the conditional probabilities of being in unobserved state s . When s is observed, Step 3 collapses to a bin estimator that could be used in the first stage of CCP estimation.

An alternative to updating the CCP's using a weighted average of the data is based on the identity that the likelihood returns the probability of replacing the engine. Substituting equation (2.7) into equation (2.3) and evaluating it at the relevant values for bus n at time t yields:

<!-- formula-not-decoded -->

Thus at the m th iteration we could replace Step 3 of the algorithm with:

Step 3A Using θ ( m ) and p ( m ) 1 , that is the function p ( m ) 1 ( x, s ) , update p ( m +1) 1 ( x nt , s n ) using:

<!-- formula-not-decoded -->

Here, l ( d 1 nt = 1 | x nt , s, p ( m ) 1 , θ ( m ) ) is calculated using (2.8). The CCP updates are tied directly to the structure of the model. We illustrate the tradeoffs of each updating method in the ensuing sections.

## 3 Framework

This section lays out a general class of dynamic discrete choice models and derives a new representation of the conditional valuation functions that we draw upon in the subsequent sections on identification and estimation. In this section we also use the representation to develop the concept of finite dependence, and determine its functional form for disturbances distributed as generalized extreme value.

## 3.1 Model

In each period until T , for T ≤ ∞ , an individual chooses among J mutually exclusive actions. Let d jt equal one if action j ∈ { 1 , . . . , J } is taken at time t and zero otherwise. The current period payoff for action j at time t depends on the state z t ∈ { 1 , . . . , Z } . In the previous section z t ≡ ( x t , s ) where x t is observed but s is unobserved to the econometrician. We ignore the distinction in this section because it is not relevant for the agents in the model. If action j is taken at time t , the probability of z t +1 occurring in period t +1 is denoted by f jt ( z t +1 | z t ).

The individual's current period payoff from choosing j at time t is also affected by a choicespecific shock, /epsilon1 jt , which is revealed to the individual at the beginning of the period t . We assume the vector /epsilon1 t ≡ ( /epsilon1 1 t , . . . , /epsilon1 Jt ) has continuous support and is drawn from a probability distribution that is independently and identically distributed over time with density function g ( /epsilon1 t ). We model the individual's current period payoff for action j at time t by u jt ( z t ) + /epsilon1 jt .

The individual takes into account both the current period payoff as well as how his decision today will affect the future. Denoting the discount factor by β ∈ (0 , 1), the individual chooses the vector d t ≡ ( d 1 t , . . . , d Jt ) to sequentially maximize the discounted sum of payoffs:

where at each period t the expectation is taken over the future values of z t +1 , . . . , z T and /epsilon1 t +1 , . . . , /epsilon1 T . Expression (3.1) is maximized by a Markov decision rule which gives the optimal action conditional on t , z t , and /epsilon1 t . We denote the optimal decision rule at t as d o t ( z t , /epsilon1 t ), with j th element d o jt ( z t , /epsilon1 t ). The probability of choosing j at time t conditional on z t , p jt ( z t ), is found by taking d o jt ( z t , /epsilon1 t ) and integrating over /epsilon1 t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We then define p t ( z t ) ≡ ( p 1 t ( z t ) , . . . , p Jt ( z t )) as the vector of conditional choice probabilities.

Denote V t ( z t ), the (ex-ante) value function in period t, as the discounted sum of expected future payoffs just before /epsilon1 t is revealed and conditional on behaving according to the optimal decision rule:

<!-- formula-not-decoded -->

Given state variables z t and choice j in period t, the expected value function in period t + 1 , discounted one period into the future is:

<!-- formula-not-decoded -->

Under standard conditions, Bellman's principle applies and V t ( z t ) can be recursively expressed as:

<!-- formula-not-decoded -->

where the second line integrates out the disturbance vector /epsilon1 t . We then define the choice-specific conditional value function, v jt ( z t ) , as the flow payoff of action j without /epsilon1 jt plus the expected future utility conditional on following the optimal decision rule from period t +1 on: 11

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hotz and Miller (1993) established that differences in conditional value functions can be expressed as functions of the conditional choice probabilities, p jt ( z t ), and the per-period payoffs. Using their results, we show that we can express the value function, V t ( z t ), as a function of one conditional value function, v jt ( z t ), plus a function of the conditional choice probabilities, p t ( z t ). Lemma 1 Define /D4 ≡ ( /D4 1 , . . . , /D4 J ) where ∑ J j =1 /D4 j = 1 and /D4 j &gt; 0 for all j ′ . Then there exists a real-valued function ψ k ( /D4 ) for every k ∈ { 1 , ..., J } such that:

Substituting (3 . 5) into the right hand side of (3.4) we obtain:

<!-- formula-not-decoded -->

11 For ease of exposition we refer to v jt ( z t ) as the conditional value function in the remainder of the paper.

Equation (3.6) shows that the conditional value function can be expressed as the flow payoff of the choice plus a function of the one period ahead conditional choice probabilities and the one period ahead conditional value function for any choice. We could repeat this procedure ad infinitum, substituting in for v kt 1 ( z t +1 ) using (3.4) and then again with (3.5) for any choice k ′ , ultimately replacing the conditional valuation functions on the right side of (3.6) with a single arbitrary time sequence of current utility terms and conditional value correction terms as defined in (3.5).

To formalize this idea, consider a sequence of decisions from t to T . The first choice in the sequence is the initial choice j which sets d ∗ jt ( z t , j ) = 1. For periods τ ∈ { t +1 , . . . , T } , the choice sequence maps z τ and the initial choice j into d ∗ τ ( z τ , j ) ≡ { d ∗ 1 τ ( z τ , j ) , . . . , d ∗ Jτ ( z τ , j ) } . The choices in the sequence then must satisfy d ∗ kτ ( z τ , j ) ≥ 0 and ∑ J k =1 d ∗ kτ ( z τ , j ) = 1. Note that the choice sequence can depend upon new realizations of the state and may also involve mixing over choices.

Now consider the probability of being in state z τ +1 conditional on following the choices in the sequence. Denote this probability as κ ∗ τ ( z τ +1 | z t , j ) which is recursively defined by:

The future value term can now be expressed relative to the conditional value functions for the choices in the sequence. Theorem 1 shows that continuing to express the future value term relative to the value of the next choice in the sequence yields an alternative expression for v jt ( z t ).

<!-- formula-not-decoded -->

Theorem 1 For any state z t ∈ { 1 , . . . , Z } , choice j ∈ { 1 , . . . , J } and decision rule d ∗ τ ( z τ , j ) defined for periods τ ∈ { t, . . . , T } :

<!-- formula-not-decoded -->

Theorem 1 shows that future value terms for dynamic discrete choice models can be expressed as functions of flow payoffs and conditional choice probabilities for any sequences of choices out until T and the corresponding transition probabilities associated with the choice sequences. It provides the foundation for the identification results discussed in Arcidiacono and Miller (2011) and the estimators developed in Section 4. In this section we use the theorem for deriving conditions to construct estimators that do not depend utility flow terms u kτ ( z τ )+ ψ k [ p τ ( z τ )] beyond a few periods for each t . Then we show how the functions ψ k ( p ) are determined in the generalized extreme value case.

## 3.2 Finite Dependence

Equation (3 . 4) implies that j is preferred to k in period t if and only if v jt ( z t ) -v kt ( z t ) &gt; /epsilon1 jt -/epsilon1 kt . Consequently conditional valuation functions, such as v jt ( z t ) and v kt ( z t ) , only enter the likelihood function in their differenced form. Substituting (3 . 8) from Theorem 1 into expressions like v jt ( z t ) -v kt ( z t ) , reveals that all the terms in the sequence after a certain date, say ρ, would cancel out if the state variables had the same probability distribution at ρ, that is if κ ∗ ρ -1 ( z ρ | z t , j ) = κ ∗ ρ -1 ( z ρ | z t , k ) , and the same decisions are selected for all dates beyond ρ .

In the example from Section 2, z t ≡ ( x t , s ) , and the act of replacing an engine next period regardless of the choice made in the current period t, thus setting d ∗ 1 ,t +1 ( z t +1 , 1) = d ∗ 1 ,t +1 ( z t +1 , 2) = 1, restores the state variables to the same value ( x t +1 , s ) = (0 , s ) at period t +1. Thus any (common) sequence of choices that begins by replacing the engine next period, implies that when considering the difference v 2 t ( z t ) -v 1 t ( z t ) in their telescoped forms using (3 . 8) , all terms beyond the next period disappear.

Exploiting the power of Theorem 1 in this way can be developed within the general framework. Consider using (3.8) to express the conditional value functions for alternative initial choices j and j ′ . Differencing the two yields:

<!-- formula-not-decoded -->

We say a pair of choices exhibit ρ - period dependence if there exists a sequence from initial choice j and a corresponding sequence from initial choice j ′ such that for all z t + ρ +1 :

<!-- formula-not-decoded -->

The sequence of choices from j and j ′ then lead to the same state in expectation. When ρ -period dependence holds, the difference in future value terms for j and j ′ can be expressed as a function of the ρ -period ahead flow payoffs, conditional choice probabilities, and state transition probabilities. Once ρ -period dependence is achieved, the remaining choices in both sequences are set to be the same, implying that equation (3.9) can be written as:

<!-- formula-not-decoded -->

as the terms associated with time periods after t + ρ drop out. Conditional on knowing the relevant ψ k ( ) mappings, the CCP's, and the transitions on the state variables, the differenced conditional value function is now a linear function of flow payoffs from t to t + ρ . Further, only the ψ k ( /D4 ) mappings that are along the two choice paths are needed: the econometrician only needs to know ψ k ( /D4 ) if choice k is part of the two decision sequences. We next use some examples to illustrate how to exploit finite dependence in practice.

## 3.2.1 Example: Renewal Actions and Terminal Choices

We apply the results in the previous section to cases where the differences in future value terms across two choices only depend on one-period-ahead conditional choice probabilities and the flow payoff for a single choice. In particular, we consider renewal problems, such as Miller's (1984) job matching model or Rust's (1987) replacement model, where the individual can nullify the effects of a choice at time t on the state at time t +2 by taking a renewal action at time t +1. For example, if Zurcher replaces the engine at t +1, then the state at time t +2 does not depend upon whether the engine was replaced at time t or not. Let the renewal action be denoted as the first choice in the set { 1 , . . . , J } , implying that d 1 t = 1 if the renewal choice is take at time t . Formally, a renewal action at t +1 satisfies:

<!-- formula-not-decoded -->

for all ( z t +2 , j, j ′ ). The state at t +2 may then depend upon the state at t +1, but only through variables that were unaffected by the choice at t .

Since the renewal action at t + 1 leads to the same expected state regardless of the choice at t , we define d ∗ 1 ,t +1 ( z t +1 , j ) = 1 for all j . Equations (3 . 7) and (3 . 12) imply that κ ∗ t +1 ( z t +2 | z t , j ) = κ ∗ t +1 ( z t +2 | z t , j ′ ) for all j and j ′ . Expressing the future value terms relative to the value of the renewal choice, v jt ( z t ) -v j ′ t ( z t ) can be written as:

Hotz and Miller (1993) show another case where only one-period-ahead conditional choice probabilities and the flow payoff for a single choice are needed is when there is a terminal choice-a choice that, when made, implies no further choices. Let the terminal choice be denoted as the first

<!-- formula-not-decoded -->

choice. With no further choices being made, the future value term for the terminal choice can be collapsed into the current period payoff. v jt ( z t ) -v j ′ t ( z t ) then follows the same expression as (3.13) when neither j nor j ′ are the terminal choice.

## 3.2.2 Example: Labor Supply

To illustrate how finite dependence works when more than one period is required to eliminate the dependence, we develop the following stylized example. 12 Consider a model of labor supply and human capital. In each of T periods an individual chooses whether to work, d 2 t = 1 , or stay home d 1 t = 1. Individuals acquires human capital, z t , by working, with the payoff to working increasing in human capital. If the individual works in period t , z t +1 = z t +2 with probability 0.5 and z t +1 = z t +1 also with probability 0.5. For periods after t , the human capital gain from working is fixed at one additional unit. When the individual does not work, her human capital remains the same in the next period.

The difference in conditional value functions between working and staying home at period t , v 2 t ( z t ) -v 1 t ( z t ), can be expressed as functions of the two-period-ahead flow utilities and conditional probabilities of working and not working. To see this, consider v 1 t ( z t ) which sets the initial choice to not work ( d 1 t = 1). Now set the next two choices to work: d ∗ 2 ,t +1 ( z t +1 , 0) = 1 and d ∗ 2 ,t +2 ( z t +2 , 0) = 1. This sequence of choices (not work, work, work) results in the individual having two additional units of human capital at t +3. Given an initial choice to work ( d 2 t = 1) it is also possible to choose a sequence such that the individual will have two additional units of human capital at t +3, but now the sequence will depend upon the realization of the future states. In particular, if the decision to work at t results in an additional two units of human capital, then set the choice in period t +1 to not work, d 1 ,t +1 ( z t +2 , 1) = 1. However, if working at t results in only one additional unit, set the choice in period t +1 to work, d 2 ,t +1 ( z t +1 , 1) = 1. In either case, the third choice in the sequence is set to not work. We can write the future value terms relative to the choices in the sequences. The future value terms after t +3 then cancel out once we difference v 1 t ( z t ) from v 2 t ( z t ). 13

12 For an empirical application of finite dependence involving more than one period and multiple discrete choices see Bishop (2008).

13 There is another sequence that also results in a cancelation occurring after three periods. In this case, we set the sequence that begins with the choice to work ( d 2 t = 1) such that the next two choices are to not work regardless of the human capital realizations. In this case, at t +3 the individual will have two additional units of human capital with probability 0.5 and one additional unit with probability 0.5. To have the same distribution of human capital at t + 3 given an initial choice not to work ( d 1 t = 1) involves mixing. In particular, set the choice at t + 1 to not work with probability 0.5, implying that the probability of not working at t +1 is also 0.5. Setting the choice at t +2

3.3 Generalized Extreme Value Distributions To apply (3 . 8) in estimation, the functional form of ψ j ( ) for some j ∈ { 1 , . . . , J } must be determined. It is well known that ψ j ( /D4 ) = -ln( /D4 j ) when j is independently distributed as Type 1 Exreme Value. We show how to numerically calculate ψ j ( /D4 ) for any generalized extreme value (GEV) distribution, and then lay out a class of problems where the mapping ψ j ( /D4 ) has an analytic solution.

Suppose /epsilon1 t is drawn from the distribution function G ( /epsilon1 1 t , /epsilon1 2 t , . . . , /epsilon1 Jt ) where

<!-- formula-not-decoded -->

and G ( /epsilon1 1 t , /epsilon1 2 t , . . . , /epsilon1 Jt ) satisfies the properties outlined for the GEV distribution in McFadden (1978). 14 Letting H j ( Y 1 , . . . , Y J ) denote the derivative of H ( Y 1 , . . . , Y J ) with respect to Y j , we define φ j ( Y ) as a mapping into the unit interval where

<!-- formula-not-decoded -->

Note that, since H j ( Y 1 , . . . , Y J ) and H ( Y 1 , . . . , Y J ) are homogeneous of degree zero and one respectively, φ j ( Y ) is a probability as ∑ J j =1 φ j ( Y ) = 1. Indeed, McFadden (1978, page 80) establishes that substituting exp[ v jt ( z t )] in for Y j in equation (3.14) yields the conditional choice probability p jt ( z t ): p jt ( z t ) = e v jt ( z t ) H j ( e v 1 t ( z t ) , . . . , e v Jt ( z t ) )/ H ( e v 1 t ( z t ) , . . . , e v Jt ( z t ) ) (3.15) We now establish a relationship between φ j ( Y ) and ψ j ( /D4 ). Denoting the vector function φ ( Y ) ≡ { φ 2 ( Y ) , . . . , φ J ( Y ) } , lemma 2 shows that φ ( Y ) is invertible. Further, lemma 2 establishes that there is a closed form expression for ψ j ( /D4 ) when φ -1 ( /D4 ) is known. Lemma 2 When /epsilon1 t is drawn from a GEV distribution, the inverse function φ -1 ( /D4 ) exists and ψ j ( /D4 ) is given by: ψ j ( /D4 ) = ln H [ 1 , φ -1 2 ( /D4 ) , . . . , φ -1 J ( /D4 ) ] -ln φ -1 j ( /D4 ) + γ (3.16)

to work regardless of the level of human capital implies that an additional two units of human will result from the sequence with probability 0.5 with the probability of one additional unit resulting from the sequence also occurring with probability 0.5. Hence, the distribution of the states is the same given the two initial choices.

14 The properties are that H ( Y 1 , Y 2 , . . . , Y J ) is a nonnegative real valued function that is homogeneous of degree one, with lim H ( Y 1 , Y 2 , . . . , Y J ) →∞ as Y k →∞ for all j ∈ { 1 , . . . , J } , and for any distinct ( i 1 , i 2 , . . . , i r ) , the cross derivative ∂ H ( Y 1 , Y 2 , . . . , Y J ) /∂Y i 1 , Y i 2 , . . . , Y i r is nonnegative for r odd and nonpositive for r even.

It is straightforward to use lemma 2 in practice by evaluating φ -1 j ( ) at p t ( z t ) for a given z t . To see this, note that from (3.15) we can express the vector p t ( z t ) as:

<!-- formula-not-decoded -->

In some cases, ψ j ( ) (and therefore φ -1 j ( )) has an analytic solution. For example, consider a case where G ( /epsilon1 t ) factors into two independent distributions, one being a nested logit, and the other a GEV distribution. Let J denote the set of choices in the nest and let K denote the number of choices that are outside the nest. G ( /epsilon1 t ) can then be expressed as:

<!-- formula-not-decoded -->

Lemma 3 is particularly powerful when there is a renewal or terminal choice. Recall from Section 3.2.1 that the only ψ j ( ) mappings needed in these cases were for the renewal or terminal choices. The payoffs for these choices may naturally be viewed as having an independent error. For example, in the Rust case, bus engine maintenance actions are more likely to be correlated with each other than with engine replacement. Another example is firm decisions when exit is an option. Choosing how many stores or different levels of product quality are likely to have correlation patterns among where G 0 ( Y 1 , Y 2 , . . . , Y K ) satisfies the properties outlined for the GEV distribution as defined by McFadden (1978). The correlation of the errors within the nest is given by σ ∈ [0 , 1] and errors within the nest are uncorrelated with errors outside the nest. When σ = 1, the errors are uncorrelated within the nest, and when σ = 0 they are perfectly correlated. Lemma (3) then shows the closed form expression for ψ j ( /D4 ) for all j ∈ J . Lemma 3 If G ( /epsilon1 t ) can be expressed as in (3 . 18) , then ψ j ( /D4 ) = γ -σ ln( /D4 j ) -(1 -σ ) ln ( ∑ k ∈J /D4 k ) (3.19) Note that ψ j ( /D4 ) only depends on the conditional choice probabilities for choices that are in the nest: the expression is the same no matter how many choices are outside the nest or how those choices are correlated. Hence, ψ j ( /D4 ) will only depend on /D4 j ′ if /epsilon1 jt and /epsilon1 j ′ t are correlated. When σ = 1, /epsilon1 jt is independent of all other errors and ψ j ( /D4 ) only depends on /D4 j . /D4

the errors that are unrelated to the payoff from exiting. As long as the error associated with the renewal or terminal choice is independent of the other errors, any correlation pattern among the other errors will still result in ψ j ( /D4 ) = -ln( /D4 j ) when j is the renewal or terminal choice.

## 4 The Estimators and their Asymptotic Properties

In estimation, we parameterize the utility function, transition function, and the probability density function for /epsilon1 t by a finite dimensional vector θ ∈ Θ , where Θ denotes the parameter space for u jt ( x t , s t ) , f jt ( x t +1 | x t , s t ), g ( /epsilon1 t ) and β and is assumed to be convex and compact. 15 There are then two sets of parameters to be estimated: θ and π, where π includes the initial distribution of the unobserved heterogeneity, π ( s 1 | x 1 ) , and its transition probability matrix, π ( s t +1 | s t ). 16

The CCP estimators we propose are derived from two sets of conditions. First are conditions which ensure the estimates of θ and π maximize the likelihood function taking the estimates of the conditional choice probabilities as given. Second are conditions that ensure the estimated conditional choice probabilities are consistent with either the data or the underlying structural model.

## 4.1 The likelihood

Denote p jt ( x, s ) as a value for the probability an individual will choose j at t given observed states x and s . Let p indicate the ( J -1) × T × X × S vector of conditional choice probabilities with the elements given by p jt ( x, s ). 17 Denote by l jt ( x nt , s nt , θ, π, p ) the likelihood of observing individual n make choice j at time t, conditional on the state ( x nt , s nt ), the parameters θ and π, and the conditional choice probabilities p :

<!-- formula-not-decoded -->

When d jnt = 1, the expression in (4.1) simplifies to (2.8) in the motivating example. The corresponding likelihood of observing ( d nt , x nt +1 ) is then defined as:

<!-- formula-not-decoded -->

15 Identification of dynamic discrete choice models with unobserved heterogeneity is analyzed in Kasahara and Shimotsu (2009) and Hu and Shum (2010a).

16 We assume the number of unobserved states is known. Heckman and Singer (1984) provide conditions for identifying the number of unobserved states in dynamic models. In principle, one could estimate the model separately for different values of S , using the differences in the likelihoods to choose the number of unobserved states. See McLachlan and Peel (2000), Chapter 6.

17 In the stationary case the dimension is ( J -1) × X × S .

The joint likelihood of any given path of choices, d n ≡ ( d n 1 . . . , d n T ) , and observed states, x n ≡ ( x n 1 . . . , x n, T +1 ) , is derived by forming the product of (4 . 2) over the T sample periods, and then integrating it over the unobserved states, ( s n 1 , . . . , s n T ). Since the probability distribution for the initial unobserved state is π ( s 1 | x 1 ) , and the transition probability is π ( s t +1 | s t ), the likelihood of observing ( d n , x n ) conditional on x 1 for parameters ( θ, π, p ) is:

<!-- formula-not-decoded -->

The log likelihood of the sample is then given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 4.2 The estimators

Following our motivating example in Section 2, we define two estimators of ( θ, π, p ). At minimal risk of some ambiguity we label them both by ( ̂ θ, ̂ π, ̂ p ) . Both estimators of ( θ, π ) maximize the expression obtained by substituting an estimator for p into (4 . 4):

In one case ̂ p jt ( x, s ) satisfies the likelihood expression l jt ( x, s, θ, π, p ) evaluated at ( ̂ θ, ̂ π, ̂ p ) , meaning:

This constraint is motivated by the fact that the model itself generates the conditional choice probabilities. Thus our first estimator solves for ( θ, π, p ) in the system of equations defined by (4.6) along with satisfying (4.5). To deal with possibility of multiple solutions, we select the one attaining the highest likelihood.

/negationslash

<!-- formula-not-decoded -->

In the second case ̂ p jt ( x, s ) is calculated as a weighted average of d njt over the sample. No weight is given to d njt if x nt = x . When x nt = x , the weight corresponds to the probability of individual n being in unobserved state s at time t . To obtain the weight in this case, we define the joint likelihood of both s nt = s and the sequence ( d n , x n ) occurring, as:

<!-- formula-not-decoded -->

Similarly denote by L n ≡ L ( d n , x n | x n 1 ; θ, π, p ) the likelihood of observing ( d n , x n ). From Bayes' rule, the probability that s nt = s conditional on ( d n , x n ) is L nt ( s nt = s ) /L n . Let ̂ L nt ( s nt = s ) denote L nt ( s nt = s ) evaluated at ( ̂ θ, ̂ π, ̂ p ) , and similarly let ̂ L n denote L n evaluated at the parameter estimates. Our second estimator of p jt ( x, s ) satisfies:

Note that the numerator in the first line gives the average probability of ( d njt , x nt , s nt ) = (1 , x, s ) and the denominator gives the average probability of ( x nt , s nt ) = ( x, s ). Their ratio is then an estimate of the probability of a sampled person choosing action j at time t conditional on x and s , with the bracketed term in the second line giving the weight placed on d njt . The second estimator ( ̂ θ, ̂ π, ̂ p ) solves (4.5) along with the set of conditions given in (4.8); again in case of multiple solutions, the solution attaining the highest likelihood is selected.

<!-- formula-not-decoded -->

## 4.3 Large sample properties

Our representation of the conditional value functions implies that any set of conditional choice probabilities ˜ p defined for all ( j, t, x, s ) induces payoffs as a function of ( θ, π ) . Substituting ˜ p for p in (4 . 4) , and then maximizing the resulting expression with respect to ( θ, π ) , yields estimates of the structural parameters, which we denote by ( ˜ θ, ˜ π ) . If the payoff functional forms were correctly specified, ( ˜ θ, ˜ π ) would converge to the true parameters under standard regularity conditions for static random utility models. Imposing the condition that ˜ p = ̂ p merely ensures an internal consistency: the conditional valuation functions used in the functional form for utility in (4 . 4) are based on the same conditional choice probabilities that emerge if the individuals in the sample actually face primitives given by ( ̂ θ, ̂ π ) . The proof to the following theorem shows that if the model is identified, the true set of parameters satisfy this internal consistency condition. Intuitively this explains why both our estimators are consistent.

Theorem 2 If the model is identified, then ( ̂ θ, ̂ π, ̂ p ) is consistent in both cases.

The remaining large sample properties, √ N rate of convergence and asymptotic normality, can be established by appealing to well known results in the literature. The covariance matrices of the estimators are given in the appendix.

## 5 The Algorithm

In order to operationalize our estimators, we modify the EM algorithm. The EM algorithm iterates on two steps. In the expectations step, the conditional probabilities of being in each unobserved state are updated as well as the initial conditions and law of motion for the unobserved states. The maximization step proceeds as if the unobserved state is observed and uses the conditional probabilities of being in each unobserved state as weights.

We show how the m th iteration is updated to the ( m +1) th as well as how to initiate the algorithm. We lay out the expectations step, the maximization step, and then summarize the algorithm.

## 5.1 Expectations Step

For the sake of the exposition, we break down the expectations step into updating:

1. q ( m ) nst , the probability of n being in unobserved state s at time t ,
2. π ( m ) ( s 1 | x 1 ), the probability distribution over the initial unobserved states conditional on the initial observed states,
3. π ( m ) ( s ′ | s ) , the transition probabilities of the unobserved states,
4. p ( m ) ( x, s ) , the conditional choice probabilities.

## 5.1.1 Updating q ( m ) nst

The first step of the m th iteration is calculating the conditional probability of being in each unobserved state in each time period given the values of the structural parameters and conditional choice probabilities from the m th iteration, { θ ( m ) , π ( m ) , p ( m ) } . The likelihood of the data on n given the parameters at the m th iteration is found by evaluating (4.3) at { θ ( m ) , π ( m ) , p ( m ) } :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we denote by L ( m ) n ( s nt = s ) the joint likelihood of the data and unobserved state s occurring at time t, given the parameter evaluation at iteration m . Evaluating (4.7) at { θ ( m ) , π ( m ) , p ( m ) } yields:

At iteration m +1, the probability of n being in unobserved state s at time t , q ( m +1) nst , then follows from Bayes rule:

<!-- formula-not-decoded -->

## 5.1.2 Updating π ( m ) ( s | x )

Setting t = 1 in (5 . 3) yields the conditional probability of the n th individual being in unobserved state s in the first time period. When the state variables are exogenous at t = 1, we can update the probabilities for the initial states by averaging the conditional probabilities obtained from the previous iteration over the sample population:

<!-- formula-not-decoded -->

To allow for situations where the distribution of the unobserved states in the first period depends on the values of the observed state variables, we form averages over q ( m +1) nst for each value of x . Generalizing (5 . 4), we set:

<!-- formula-not-decoded -->

## 5.1.3 Updating π ( m ) ( s ′ | s )

Updating the probabilities of transitioning among unobserved states requires calculating the probability of n being in unobserved state s ′ at time t conditional on the data and also on being in unobserved state s at time t -1, q ns ′ t | s . The joint probability of n being in states s and s ′ at time t -1 and t can then be expressed as the product of q nst -1 and q ns ′ t | s . The updating formula for the transitions on the unobserved states is then based on the identities:

<!-- formula-not-decoded -->

where the n subscript on an expectations operator indicates that the integration is taken over the population. The second line then follows from the law of iterated expectations.

Substituting the relevant sample analogs at the m th iteration for q ns ′ t | s and q nst -1 into (5.6)

then yields our update of π ( m ) ( s ′ | s ). First we compute q ( m +1) ns ′ t | s according to:

<!-- formula-not-decoded -->

Then, using the sample analog of (5 . 6) yields:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 5.1.4 Updating p ( m ) jt ( x, s )

Following our motivating example, we propose two methods of updating the CCP's. One way is to use the current estimates of the model parameters coupled with the current conditional choice probabilities. Generalizing equation (2 . 22) , the value of p ( m +1) jt ( x, s ) at the m + 1 iteration is computed according to:

An alternative to updating CCP's with the model, is to use the data and the conditional probabilities of being in each of the unobserved states, q nst . Substituting s nt for s n , and q nst for q ns , we can rewrite equations (2 . 13) through (2 . 16) for any choice j at time t to show that at the model's true parameters:

<!-- formula-not-decoded -->

This formulation suggests a second way of updating p ( m ) jt ( x, s ) is to use the weighted empirical likelihood:

## 5.2 Maximization Step

<!-- formula-not-decoded -->

The primary benefit of the EM algorithm is that its maximization step is much simpler than the optimization problems defined in equations (4 . 4) through (4 . 8). Rather than maximizing a logarithm of weighted summed likelihoods-which requires integrating out over all the unobserved states-the maximization step treats the unobserved states as observed and weights each observation by q ( m ) nst .

Thus θ ( m +1) is updated using:

<!-- formula-not-decoded -->

## 5.3 Summary

We have now defined all the pieces necessary to implement the algorithm. It is triggered by setting initial values for the structural parameters, θ (1) , the initial distribution of the unobserved states plus their probability transitions, π (1) , and the conditional choice probabilities p (1) . Natural candidates for these initial values come from estimating a model without any unobserved heterogeneity and perturbing the estimates. Each iteration in the algorithm has four steps. Given ( θ ( m ) , π ( m ) , p ( m ) ) the ( m +1) th iteration follows:

Step 1 Compute q ( m +1) nst , the conditional probabilities of being in each unobserved state, using (5.3).

- Step 2 Compute π ( m +1) ( s 1 | x 1 ), the set of initial population probabilities of the unobserved states, and π ( m +1) ( s ′ | s ), the transition parameters for the unobserved states, using (5.5) and (5.7).

Step 3 Compute p ( m +1) , the conditional choice probabilities, using either (5.8) or (5.9).

- Step 4 Compute the structural parameters, θ ( m +1) , by solving (5.10).

By construction, the converged values satisfy the first order conditions of the estimators we defined in the previous section.

## 6 A Two Stage Estimator

The estimation approach described above uses information from both the observed state transitions as well as the underlying model of the decision process. When the CCP's are identified from the reduced form alone, we show it is possible to estimate the parameters governing the unobserved heterogeneity and the conditional choice probabilities in a first stage. The structural parameters of the dynamic discrete choice process are then estimated in a second stage.

One advantage of estimating the CCP's in a first stage is that, as we show in Arcidiacono and Miller (2011), the unrestricted flow utilities can then be computed directly; imposing further model based structure can be tested in a straightforward way. Another advantage is that these first stage estimates can be paired with any one of several CCP estimators already developed for models where

there is no unobserved heterogeneity. For example, at the end of this section we show how to pair our first stage estimates with simulation-based estimators in the second stage.

We begin by partitioning the structural parameters into those that affect the observed state transitions, θ 1 , and those that do not, θ 2 , where θ ≡ { θ 1 , θ 2 } . The transitions on the observed state variables are then given by f jt ( x t +1 | x t , s t , θ 1 ). We show how to estimate θ 1 , the initial probability distribution of the unobservables, π ( s 1 | x 1 ), their transitions, π ( s ′ | s ) , and unrestricted estimates of the CCP's, p jt ( x t , s t ) , in a first stage. Also estimated in the first stage are the conditional probabilities of being in an unobserved state in a particular time period, q nst . In the second stage, we use the first stage estimates of θ 1 , π ( s ′ | s ), q nst , and p jt ( x t , s t ), to estimate the remaining parameters, θ 2 , which includes those parameters governing u jt ( x t , s t ), G ( /epsilon1 t ) , and the discount factor β .

In the second stage, we use the first stage estimates of θ 1 , π ( s ′ | s ), q nst , and p jt ( x t , s t ), to estimate the remaining parameters, θ 2 , which includes those parameters governing u jt ( x t , s t ), G ( /epsilon1 t ) , and the discount factor β . The two stage estimator presents an attractive option when there are additional outcomes-in this case x t +1 - that are affected by the unobserved state.

## 6.1 First stage

The function l jt ( x nt , s, θ, π, p ) imposes the structure of the economic model on the probability a particular choice is made. In the first stage we do not impose this structure and replace l jt ( x nt , s, θ, π, p ) in equation (4 . 2) with p jt ( x nt , s nt ):

<!-- formula-not-decoded -->

The updates for q ( m ) nst , π ( m ) ( s 1 | x 1 ), and π ( m ) ( s ′ | s ) are then defined by the same formulas used in the baseline algorithms, namely (5 . 3), (5 . 5) and (5 . 7). The only difference is that, when calculating the likelihood of ( d nt , x nt +1 ) given x nt , s nt and the m th estimate of the parameters and conditional choice probabilities, we evaluate (6.1) at ( θ ( m ) 1 , p ( m ) ) which uses the empirical likelihood for the choices rather than the structural likelihood implied by the model.

At the m th iteration, the maximization step then recovers θ ( m +1) 1 and p ( m +1) from:

<!-- formula-not-decoded -->

This optimization problem is additively separable in θ 1 and p implying the parameters can be estimated in stages. p ( m +1) jt ( x, s ) has the closed form solution given by equation (5 . 9) , the empirical

update for the CCP's in the baseline algorithm. To prove this claim, note that for all j and j ′ in the choice set, the first order conditions can be expressed as:

<!-- formula-not-decoded -->

Multiplying both sides by p ( m +1) jt p ( m +1) j ′ t and summing over all j ′ ∈ { 1 , . . . , J } , gives the result.

The first stage is then initiated by setting θ (1) , p (1) , π (1) ( s 1 | x 1 ) and π (1) ( s ′ | s ). At the m th iteration, q ( m +1) nst , π ( m +1) ( s 1 | x 1 ), and π ( m +1) ( s ′ | s ) are all updated as in the baseline algorithm but using the empirical likelihood for the choices. Then, p ( m +1) is updated using (5 . 9) and θ ( m +1) 1 is solved from (6 . 2). Since this is a standard EM algorithm, the log likelihood increases at each iteration and will converge to a local maximum.

## 6.2 Second stage

The first stage estimates for p are used in the second stage to obtain estimates for θ 2 , the parameters defining u jt ( x t , s t ), G ( /epsilon1 t ) , and the discount factor β . A likelihood function could be formed to estimate these remaining parameters, but simulation methods, such as those proposed by Hotz, Miller, Sanders, and Smith (1994) and Bajari, Benkard and Levin (2007) are also available.

Consider, for example, a modified version of the estimator of Hotz et al (1994). 18 From Equation (3 . 5), the difference between any two conditional value functions is:

<!-- formula-not-decoded -->

For each unobserved state we stack the ( J -1) mappings from the conditional choice probabilities into the differences in conditional value functions for each individual n in each period t :

<!-- formula-not-decoded -->

18 See Finger (2008) for an application of our two-stage estimator, where the second stage uses the estimator developed by Bajari, Benkard, and Levin (2007).

Second stage estimation is based on forward simulating the differences in conditional value functions arrayed in (6 . 3) to obtain their differences in terms of weighted sums of future utilities u kτ ( x nτ , s nτ )+ ψ [ p t ( x nτ , s nτ )] for τ ≥ t. In the papers we cited above these differences are simulated for all τ ≤ T or in the infinite horizon case, until these differences are negligible due to the combination of discounting and the simulated paths of future choices. A method of moments estimator is then formed by squaring a weighted sum over the sample population and minimizing it with respect to the structural parameters.

The simulation approaches in Hotz, Miller, Sanders, and Smith (1994) and Bajari, Benkard, and Levin (2007) involve simulating both the path of the state variables as well as the decisions significantly out into the future. With finite dependence, an alternative is to use the decision rules such that finite dependence holds and use simulation only for transitions of the state variables. 19 Hence, if the problem exhibits ρ -period dependence, only simulations over ρ periods are needed. In this case, forward simulation is particularly powerful as the paths are drawn from where the individual is currently at. For example, Bishop (2008) uses the techniques in this paper for a state space that has 1 . 12 E +184 elements with finite dependence possible after three periods. By forward simulating, she is able to evaluate the future value term at a small subset of likely future states.

## 7 Small Sample Performance

To evaluate the small sample properties and the computational speed of our estimators, we now conduct two Monte Carlo studies. In each design, we illustrate the nature and extent of the problem that our estimators solve, by showing that unobserved heterogeneity would produce biased estimates if left unaccounted. Taken together, the two exercises cover finite and infinite horizon models, single agent problems and dynamic games, and cases where the unobserved state is fixed and when it varies over time.

## 7.1 Renewal and Finite Horizon

Our first Monte Carlo revisits the renewal problem described in Section 2, where the unobserved state is fixed over time. We describe the experimental design for this study, and then report our Monte Carlo results on the computational gains and the efficiency loss of our estimators relative to FIML. We also consider how well our estimator performs in nonstationary settings when the sample

19 This has the added benefit of weakening the assumptions regarding the time horizon as well as how the state variables transition far out into the future.

period falls short of the individual's time horizon.

## 7.1.1 The bus engine problem revisited

We adapt the model discussed in Section 2 to a finite horizon setting, again normalizing the dependence of flow utility to zero when the engine is replaced. The payoff of keeping the current engine depends on the state s where s ∈ { 1 , 2 } and accumulated mileage, x 1 t . Maintenance costs increase linearly with accumulated mileage up to 25 and then flatten out. Tracking accumulated mileage beyond 25 is therefore redundant. The flow payoff of keeping the current engine is then specified as:

<!-- formula-not-decoded -->

Mileage accumulates in increments of 0.125. Accumulated mileage depends on the decision to replace the engine, the previous mileage, and a permanent route characteristic of the bus denoted by x 2 . We assume that x 2 is a multiple of 0.01 and drawn from a discrete equiprobability distribution between 0.25 and 1.25. Higher values of x 2 are then associated with shorter trips or less frequent use. The probability of x t +1 conditional on x t ≡ { x 1 t , x 2 } and d jnt = 1, f j ( x t +1 | x t ), is specified as:

implying that the mileage transitions follow a discrete analog of an exponential distribution. Since accumulated mileages above 25 are equivalent to mileages at 25 in the payoff function, we collapse all mileage transitions above 25 to 25, implying that f 1 (25 | x t ) = exp[ -x 2 (25)] and f 2 (25 | x t ) = exp[ -x 2 (25 -x 1 t )].

<!-- formula-not-decoded -->

Relative to the simpler version we presented in Section 2, there are then three changes to the conditional value function. First, the conditional value function is now subscripted to reflect the finite horizon. Second, we have an intercept term on the flow payoff of running the engine. Finally, the mileage transitions are now stochastic. These modifications imply that the difference in conditional value function between running and replacing the engine given in (2 . 7) now become:

<!-- formula-not-decoded -->

where the sum over x t +1 goes from x t to 25 in increments of 0.125. Since x 2 does not affect flow payoffs but does affect future utility through the mileage transitions, we can estimate β .

We simulate data for a decision-maker who lives for 30 periods and makes decisions on 1000 buses in each period. The data are generated by deriving the value functions at each state using backwards recursion. We then start each bus engine at zero miles and simulate the choices. The data are generated using two unobserved states with the initial of probabilities of each unobserved state set to 0.5. The econometrician is assumed to see only the last 20 periods, implying an initial conditions problem when s is unobserved. Summarizing the dimensions of this problem, there are two choices, twenty periods of data, two unobserved states, two hundred and one possible mileages and one hundred and one observed permanent characteristics. The number of states is therefore 20 × 2 × 201 × 101 = 812 , 040. Additional details regarding the data generating process and the estimation methods is in the appendix.

## 7.1.2 CCP versus FIML

The first column of Table 1 shows the parameters of the model. The next two columns show estimates from 50 simulations using both full information maximum likelihood and conditional choice probabilities respectively when the type of the bus is observed. The conditional choice probabilities are estimated using a logit that is a flexible function of the state variables. 20 Both methods produced estimates centered around the true values. There is some loss of precision using the CCP estimator compared to FIML, but the standard deviations of the CCP estimates are all less than fifty percent higher than the FIML standard deviations and in most cases much less. These results are comparable to those found by Hotz et al (1994) and the two-step estimator of Aguirregabiria and Mira (2002) in their Monte Carlo studies of a similar problem. 21

Weighed against this loss of precision are the computational gains associated with CCP estimation. There are two reasons explaining why it is almost 1700 times faster than FIML. The full solution method solves the dynamic programming problem at each candidate value for the parameter estimates, whereas this estimator pairs a smoothed bin estimator of the CCP's (to handle sparse cells as explained in the appendix) with a logit that estimates the structural parameters. Second, the number of CCP's used to compute the CCP estimator is roughly proportional to the number

20 The state space is too large to use a bin estimator. Alternatives to the flexible logit include nonparametric kernels and basis functions. We use flexible logits because of their computational convenience. See the appendix for details of the terms included in the logit.

21 The main difference between our specification and theirs is that we exploit the renewal property. This approach circumvents simulation into the indefinite future taken by Hotz et al (1994), and avoids inverting matrices with dimension equal to the number of elements in the state space (just over 40,000 in our case) followed by Aguirregabiria and Mira (2002).

of data points, because of the finite dependence property. In a typical empirical application, and also here, this number is dwarfed by the size of the state space which is the relevant benchmark for solving the dynamic optimization problems and FIML estimation.

Column 4 shows the results for CCP methods when bus type is unobserved but the heterogeneity in bus type is ignored in estimation. Averaging over the two unobserved states, the expected benefit of running a new engine is 2.5. The estimate of θ 0 when unobserved heterogeneity is ignored is lower than this due to dynamic selection. Since buses with s = 1 are less likely to be replaced, they are disproportionately represented at higher accumulated miles. As a result, the parameter on accumulated mileage, θ 1 , is also biased downward. These results suggest that neglecting to account for unobserved heterogeneity can induce serious bias, confirming for this structural dynamic framework early research on unobserved heterogeneity in duration models by Heckman (1981).

In Columns 5 and 6 we estimate the model treating s as unobserved. We used the second CCP estimator, which updates the CCP's using the estimated relative frequencies of being in the unobserved state, updated via a reduced form logit explained in the appendix. To handle the initial conditions problem that emerges from only observing engine replacements after Zurcher has been operating his fleet for ten periods, the initial probability of being in an unobserved state (at period eleven) is estimated as a flexible function of the initial observed state variables. 22 Again, both FIML and CCP methods yielded estimates centered around the truth. There is surprisingly little precision loss from excluding the brand variable in the data and modeling s as unobserved. Our results also show the loss from using CCP estimator is smaller than FIML; all standard deviations are less than twenty five percent higher than the standard deviations of the corresponding FIML estimates. That the difference in precision shrinks occurs because some of the structure of the model is imposed in the CCP estimation through the probabilities of being in particular unobserved states.

Regardless of which method is used, treating bus brand as an unobserved state variable, rather than observed, increases computing time. The increased computational time in the CCP estimator is fully explained by the iterative nature of the EM algorithm, because each iteration essentially involves estimating the model as if the brand is observed. 23 Similarly, though not quite as transparently, FIML does not evaluate the likelihood of each bus history given the actual brand (as in Column 2), but the likelihood for both brands. This explains why estimation time for FIML es-

22 This approach is also used by Keane and Wolpin (1997,2000,2001), Eckstein and Wolpin (1999), and Arcidiacono, Sieg, and Sloan (2007).

23 We did not explore the possibility of speeding up the EM algorithm using techniques developed by Jamshidian and Jennrich (1997) and others.

sentially doubles (because there are two brands), increasing by more than two hours, whereas CCP increases by a factor of eighty, or by six and a half minutes.

## 7.1.3 Nonstationary problems with short panels

To further explore the properties of our estimators we made the exercises more complex, in the process precluding FIML estimation. As discussed Arcidiacono and Miller (2011), dynamic discrete choice models are partially identified even when there is no data on the latter periods of the optimization problem. In particular, suppose individuals have information about their future, never captured in the data, that affects their current choices. In life cycle contexts this might include details about inheritances, part time work opportunities following retirement, and prognoses of future health and survival. In this model, Zurcher might know much more about the time horizon and how engine replacement costs will evolve in the future. Both factors affect the optimal policy; he will tend to replace bus engines when they are cheaper, extending the life of some and shortening the life of others to take advantage of sales, and towards the end of the horizon he will become increasingly reluctant to replace engines at all.

We now assume that the cost of running old engines and replacing them varies over time, and substitute a time dependent parameter θ 0 t for θ 0 in equation (7 . 1) . To emphasize the fact that the time shifters affect the costs for each bus the same way, we subscript bus-specific variables by n . Equation (7 . 1) becomes:

<!-- formula-not-decoded -->

We assume Zurcher knows both the value of T and the sequence { θ 01 , . . . , θ 0 T } . In contrast, the econometrician does not observe the values of θ 0 t , and knows only that the data ends before T.

Implementing FIML amounts to estimating the value of the integer T, the sequence { θ 01 , . . . , θ 0 T } , as well as the parameters of interest { θ 2 , θ 3 , β } . If T comes from the (unbounded) set of positive integers, then the model is not identified for panels of fixed length. In practice, strong parametric assumptions must be placed on the length of the horizon, T, and how aggregate effects here modeled by θ 0 t , evolve over time after the sample ends. 24 In contrast, implementing the second of our CCP estimators (that exploits the finite dependence property and updates using the estimated relative

24 Similarly the first CCP estimator is not feasible. To update p ( m +1) 1 t ( x nt , s n ) for period t with l 1 ( x nt , s n , θ ( m ) , π ( m ) ) , p ( m ) ) using (5 . 8) we require p ( m ) 1 ,t +1 ( x n,t +1 , s n ). (See for example Equation (2 . 8) in Section 2.). But an input p ( m -1) 1 ,t +2 ( x n,t +2 , s n ) is required in l 1 ( x n,t +1 , s n , θ ( m -1) , π ( m -1) ) , p ( m -1) ) to update p ( m ) 1 ,t +1 ( x n,t +1 , s n ) for period t +1, and so on. The upshot is that the full θ 0 t sequence and also T is estimated, as in FIML.

frequencies of the unobserved variable), does not require any assumptions about the process driving θ 0 t or the horizon length. This is because the conditional choice probabilities from the last period of the data convey all the relevant information regarding the time horizon and the future values of θ 0 t , enabling estimation of the flow payoffs up to one period before the sample ends. 25

We again set T to 30, but now assume the econometrician only observes periods 11 through 20, increasing the number of buses from 1000 to 2000 to keep the size of the data sets comparable to the previous Monte Carlos. 26 Results with both s observed and unobserved are given in the last two columns of Table 1. When s is observed, adding time-varying intercepts has virtually no effect on the precision of the other parameters, or on the computational time. When s is unobserved, computation time almost doubles relative to CCP estimation with no time-varying effects (from 6.59 minutes to 11.31 minutes), and the standard deviations of the parameters increase by up to fifty percent. However, the estimates are still centered around the truth and reasonably precise, demonstrating that structural estimation in these more demanding environments can be quite informative.

## 7.2 Dynamic Games

The second Monte Carlo experiment applies our estimators to infinite horizon dynamic games with private information. To motivate the exercise, we first show how our framework can be adapted to stationary infinite horizon games with incomplete information. We then apply our estimators to an entry and exit game. 27 Here the unobserved states affect demand and evolve over time according to a Markov chain. Finally, we report on the performance of the baseline estimators as well as the alternative two-stage estimator developed in Section 6.

## 7.2.1 Adapting our framework to dynamic games

We assume that there are I firms in each of many markets, and that the systematic part of payoffs to the i th firm in a market not only depends on its own choice in period t, denoted by d ( i ) t ≡ ( d ( i ) 1 t , . . . , d ( i ) Jt ) , the state variables z t , but also the choices of the other firms in that market, which

25 More generally, in problems exhibiting finite dependence of ρ, the second estimator provides estimates up to ρ periods before the sample ends.

26 The data in period 20 are only used to create the conditional choice probabilities used in the future value term for period 19. This is because the econometrician requires one-period-ahead CCP's to create differences in the conditional value functions for the current period.

27 The empirical literature on dynamic games with exit begins with Gowrisankaran and Town (1997). Two-step estimation of dynamic games with exit includes Beauchamp (2010), Beresteanu et al. (2010), Collard-Wexler (2008), Dunne et al. (2009), Gowrisankaran et al. (2009), and Ryan (2009), .

Table 1: Monte Carlo for Optimal Stopping Problem †

|                     |            |            |            |          |              | Time Effects   | Time Effects   | Time Effects   |
|---------------------|------------|------------|------------|----------|--------------|----------------|----------------|----------------|
|                     | s observed | s observed | s observed | s        | s unobserved | s unobserved   | s observed     | s unobserved   |
|                     | DGP        | FIML       | CCP        | CCP      | FIML         | CCP            | CCP            | CCP            |
|                     | (1)        | (2)        | (3)        | (4)      | (5)          | (6)            | (7)            | (8)            |
| θ 0 (Intercept)     | 2          | 2.0100     | 1.9911     | 2.4330   | 2.0186       | 2.0280         |                |                |
|                     |            | (0.0405)   | (0.0399)   | (0.0363) | (0.1185)     | (0.1374)       |                |                |
| θ 1 (Mileage)       | -0.15      | -0.1488    | -0.1441    | -0.1339  | -0.1504      | -0.1484        | -0.1440        | -0.1514        |
|                     |            | (0.0074)   | (0.0098)   | (0.0102) | (0.0091)     | (0.0111)       | (0.0121)       | (0.0136)       |
| θ 2 (Unobs. State)  | 1          | 0.9945     | 0.9726     |          | 1.0073       | 0.9953         | 0.9683         | 1.0067         |
|                     |            | (0.0611)   | (0.0668)   |          | (0.0919)     | (0.0985)       | (0.0636)       | (0.1417)       |
| β (Discount Factor) | 0.9        | 0.9102     | 0.9099     | 0.9115   | 0.9004       | 0.8979         | 0.9172         | 0.8870         |
|                     |            | (0.0411)   | (0.0554)   | (0.0591) | (0.0473)     | (0.0585)       | (0.0639)       | (0.0752)       |
| Time (Minutes)      |            | 130.29     | 0.078      | 0.033    | 275.01       | 6.59           | 0.079          | 11.31          |
|                     |            | (19.73)    | (0.0041)   | (0.0020) | (15.23)      | (2.52)         | (0.0047)       | (5.71)         |

† Mean and standard deviations for fifty simulations. For columns (1)-(6), the observed data consist of 1000 buses for 20 periods. For columns (7)-(8), the intercept ( θ 0 ) is allowed to vary over time and the data consist of 2000 buses for 10 periods. See text and appendix for additional details.

we now denote by d ( -i ) t ≡ ( d (1) t , . . . , d ( i -1) t , d ( i +1) t , . . . , d ( I ) t ) . Consistent with the games literature, we assume that the environment is stationary. Denote by U ( i ) j ( z t , d ( -i ) t ) + /epsilon1 ( i ) jt the current utility of firm i in period t, where /epsilon1 ( i ) jt is an identically and independently distributed random variable that is private information to the firm. Although the firms all face the same observed state variables, these state variables will affect the firms in different ways. For example, a characteristic of firm i will affect the payoff for firm i differently than a characteristic of firm i ′ . Hence, the payoff function is superscripted by i .

Firms make simultaneous choices in each period. We denote P ( d ( -i ) t | z t ) as the probability firm i 's competitors choose d ( -i ) t conditional on the state variables z t . Since /epsilon1 ( i ) t is independently distributed across all the firms, P ( d ( -i ) t | z t ) has the product representation:

/negationslash

<!-- formula-not-decoded -->

We impose rational expectations on the firm's beliefs about the choices of its competitors and assume firms are playing stationary Markov-perfect equilibrium strategies. Hence, the beliefs of the firm match the probabilities given in equation (7.4). Taking the expectation of U ( i ) j ( z t , d ( -i ) t ) over d ( -i ) t , we define the systematic component of the current utility of firm i as a function of the firm's state variables as

<!-- formula-not-decoded -->

The values of the state variables at period t +1 are determined by the period t choices by both the firm and its competitors as well as the period t state variables. Denote F j ( z t +1 ∣ ∣ ∣ z t , d ( -i ) t ) as the probability of z t +1 occurring given action j by firm i in period t, when its state variables are z t and the other firms in its markets choose d ( -i ) t . The probability of transitioning from z t to z t +1 given action j by firm i in then given by:

<!-- formula-not-decoded -->

The expressions for the conditional value function for firm i is then no different than what was described in Section 3 subject to the condition that we are now in a stationary environment. For

example, equation (3.6) is modified in the stationary games environment to: 28

<!-- formula-not-decoded -->

One caveat is that multiple equilibria may be an issue. As illustrated in Pesendorfer and SchmidtDengler (2010), iterating to obtain the equilibrium may make it impossible to solve for particular equilibria. One of the benefits of our two-stage estimator applied to games is that no equilibrium is iteratively solved for in the first stage. Instead, conditional choice probabilities are taken from the data themselves with the data on another outcome used to pin down the distribution of the unobserved states. With consistent estimates of the conditional choice probabilities from the first stage, we can then estimate the structural parameters of the dynamic decisions taking these as given-no updating of the CCP's is required.

## 7.2.2 An entry and exit game

Our second Monte Carlo illustrates the small sample properties of our algorithms in a games environment. We analyze a game of entry and exit. In this game d ( i ) t ≡ ( d ( i ) 1 t , d ( i ) 2 t ) where d ( i ) 1 t = 1 means i exits the industry in period t, and d ( i ) 2 t = 1 means the firm is active, either as an entrant (when d ( i ) 2 ,t -1 = 0), or as a continuing incumbent (when d ( i ) 2 ,t -1 = 1). When a firm exits, it is replaced by a new potential entrant.

Following the notational convention in the rest of the paper, we partition the state variables of the firm, z t , into those the econometrician observes, x t , and the unobserved state variables, s t . The observed state has two components. The first is a permanent market characteristic, denoted by x 1 , and is common across firms in the market. Each market faces an equal probability of drawing any of the possible values of x 1 where x 1 ∈ { 1 , 2 , . . . , 10 } . The second observed characteristic, x 2 t , is whether or not each firm is an incumbent, x 2 t ≡ { d (1) 2 t -1 , . . . , d ( I ) 2 t -1 } . Firms who are not incumbents must pay a start up cost making it less likely that these firms will choose to stay in the market. The observed state variables are then x t ≡ { x 1 , x 2 t } .

/negationslash

The unobserved state variable s t ∈ { 1 , . . . , 5 } , which we interpret as a demand shock, follows a first order Markov chain. We assume that the probability of the unobserved state remaining unchanged in successive periods is fixed π ∈ (0 , 1). If the state does change, any other state is equally likely to occur implying that the probability of s t +1 conditional on s t when s t = s t +1 is (1 -π ) / 4.

28 The results described earlier for the single agent case on finite dependence, the structure of the errors, and estimating conditional choice probabilities in the presence of unobserved states apply to games as well.

The flow payoff of firm i being active, net of private information /epsilon1 ( i ) 2 t , is modeled as:

/negationslash

We normalize U ( i ) 1 ( x ( i ) t , s ( i ) t , d ( -i ) t ) = 0. U ( i ) 2 ( x ( i ) t , s ( i ) t , d ( -i ) t ) is then used to form u ( i ) 2 ( x t , s t ) by way of (7.5).

<!-- formula-not-decoded -->

We assume that the firm's private information, /epsilon1 ( i ) jt , is distributed Type 1 extreme value. Since exiting is a terminal choice with the exit payoff normalized to zero, the Type 1 extreme value assumption and (7.7) imply that the conditional value function for being active is:

<!-- formula-not-decoded -->

The future value term is then expressed as a function solely of the one-period-ahead probabilities of exiting and the transition probabilities of the state variables.

We also generated price data on each market, denoted by y t to capture the idea that unobserved demand shocks typically affect other outcomes apart from the observed decisions. Prices are a function of the permanent market characteristic, x 1 , the number of firms active in the market, and a shock denoted by η t . We assume the shock follows a standard normal distribution and is independently distributed across markets and periods. The shock is revealed to each market after the entry and exit decisions are made. The price equation is then specified as:

<!-- formula-not-decoded -->

Note that we cannot obtain consistent estimates of the price equation by regressing prices on the x and the entry/exit decisions. This is because the unobserved state s t would be part of the residual and is correlated with the entry/exit decisions. As we show in the results section, ignoring this selection issue results in the estimates of the effects of competition on price being upward-biased.

The number of firms in each market is set to six and we simulated data for 3,000 markets. The discount factor is set at β = 0 . 9 . Starting at an initial date with six potential entrants in the market, we ran the simulations forward for twenty periods. To show that our algorithms can easily be adapted to cases where there is an initial conditions problem, we used only the last ten periods to estimate the model. Initial probabilities of being in each unobserved state are again estimated as a flexible function of the state variables in the first observed period. 29 Note that multiple equilibria

29 Further details of the reduced form for the initial conditions parameters can be found in the appendix.

may be possible here. We did not run into this issue in the data creation or in estimation and we assume that one equilibrium is being played in the data. A key difference between this Monte Carlo and the renewal Monte Carlo is that the conditional choice probabilities have an additional effect on both current utility and the transitions on the state variables due to the effect of the choices of the firm's competitors on profits.

## 7.2.3 Results

The first column of Table 2 shows the parameters of the data generating process. The next two columns show what happens when s t is observed and when it is ignored. When s t is observed, all parameters are centered around the truth with the average estimation time being eight seconds. Column 3 shows that ignoring s t results in misleading estimates of the effects of competition on prices and profits. The parameters in both the profit function and in the price equation on the number of competitors ( θ 3 and α 3 ) are biased upward, significantly underestimating the effect of competition. 30

Column 4 reports our results for the estimator when the conditional choice probabilities are updated using the model. All estimates are centered around the truth with the average computational time being a little over 21 minutes. Column 5 updates the conditional choice probabilities with a reduced form logit of the type analyzed in the renewal problem. The standard deviations of the estimated profit parameters (the θ 's) increase slightly relative to the case when the CCP's are updated with the model. Computation time also increases by a little over twenty-five percent.

Column 6 presents results using the two-stage method. Here, the distribution for the unobserved heterogeneity and the CCP's are estimated in a first stage along with the parameters governing the price process. The only parameters estimated in the second stage are those governing the current flow payoffs. The standard deviations of the coefficient estimates are similar to the case when the CCP's are updated with the model. Computation time, however, is faster, averaging a little over fifteen minutes.

Finally, we consider the case when no price data is available. When data are only available on the discrete choices, is it still possible to estimate a specification where the unobserved states are allowed to transition over time? Column 7 shows that in some cases the answer is yes. Estimating the model without the price data and updating the CCP's using the model again produces estimates

30 Estimating the price process with market fixed effects did not change this result. In this case, α 3 was estimated to be around -0.2, again underestimating the effects of competition.

that are centered around the truth with the standard deviations of the estimates similar to that of the two-stage method. The one parameter that is less precisely estimated is the persistence of the unobserved state. Computation time is also fast at a little less than seventeen minutes. These results would suggest that the additional information on prices is not particularly helpful. Note, however, that there are six entry/exit decisions for every one price observation. In unreported results we reduced the maximum number of firms in a market to two and in this case including price data substantially improved the precision of the estimates.

Although the Monte Carlo results show that rich forms of unobserved heterogeneity can be accounted for using our methods, their are both computational and identification limitations to accommodating large numbers of unobserved state variables. With regard to identification, the number of unobserved states that can be accommodated is limited by both variation in the observed state variables as well as the length of the panel. There are also computational restrictions as integrating out large numbers of unobserved state variables substantially complicates forming the likelihood of the data which is a necessary input into forming the conditional probabilities of being in particular unobserved states. Even though this is less of an issue here than in full solution methods where the unobserved states would have to be integrated out within the likelihood maximization routine, increasing the number of unobserved states nonetheless will increase computational time. The increase in computational time is somewhat mitigated through our assumption that the unobserved states follow a Markov process. 31 Namely, consider forming the joint likelihood that an individual made choice d t and was in unobserved state s t . The way in which events after t and before t affect this likelihood operate fully through the cumulative probability of being in each of the unobserved states at t -1 and t +1 given d t and s t ; the paths into the states at t -1 and t +1 do not matter except through this cumulative probability.

## 8 Conclusion

CCPmethods can reduce the computational time of estimating dynamic discrete choice models. This paper extends the class of models that are easily adapted to the CCP framework, by broadening the set of dynamic discrete choice problems where few conditional choice probabilities are needed, as well as showing how to incorporate unobserved heterogeneity into CCP estimation.

We establish that future utility terms can always be expressed as function of conditional choice probabilities and the flow payoffs for any choice sequence. When two choice sequences with different

31 The computational time becomes even less of a concern when the unobserved states are permanent.

Table 2: Monte Carlo for Entry/Exit Game †

|            |                                 | DGP (1)   | s t observed (2)   | Ignore s t (3)   | CCP-Model (4)   | CCP-Data (5)   | 2-stage (6)   | No Prices (7)   |
|------------|---------------------------------|-----------|--------------------|------------------|-----------------|----------------|---------------|-----------------|
|            | θ 0 (Intercept)                 | 0         | 0.0207             | -0.8627          | 0.0073          | 0.0126         | -0.0251       | -0.0086         |
|            |                                 |           | (0.0779)           | (0.0511)         | (0.0812)        | (0.0997)       | (0.1013)      | (0.1083)        |
|            | θ 1 (Obs. State)                | 0.05      | -0.0505            | -0.0118          | -0.0500         | -0.0502        | -0.0487       | -0.0495         |
| Profit     |                                 |           | (0.0028)           | (0.0014)         | (0.0029)        | (0.0041)       | (0.0039)      | (0.0038)        |
| Parameters | θ 2 (Unobs. State)              | 0.25      | 0.2529             |                  | 0.2502          | 0.2503         | 0.2456        | 0.2477          |
|            |                                 |           | (0.0080)           |                  | (0.0123)        | (0.0148)       | (0.0148)      | (0.0158)        |
|            | θ 3 (No. of Competitors)        | -0.2      | -0.2061            | 0.1081           | -0.2019         | -0.2029        | -0.1926       | -0.1971         |
|            |                                 |           | (0.0207)           | (0.0115)         | (0.0218)        | (0.0278)       | (0.0270)      | (0.0294)        |
|            | θ 4 (Entry Cost)                | -1.5      | -1.4992            | -1.5715          | -1.5014         | -1.4992        | -1.4995       | -1.5007         |
|            |                                 |           | (0.0131)           | (0.0133)         | (0.0116)        | (0.0133)       | (0.0133)      | (0.0139)        |
|            | α 0 (Intercept)                 | 7         | 6.9973             | 6.6571           | 6.9991          | 6.9952         | 6.9946        |                 |
|            |                                 |           | (0.0296)           | (0.0281)         | (0.0369)        | (0.0333)       | (0.0335)      |                 |
|            | α 1 (Obs. State)                | -0.1      | -0.0998            | -0.0754          | -0.0995         | -0.0996        | -0.0996       |                 |
| Price      |                                 |           | (0.0023)           | (0.0025)         | (0.0028)        | (0.0028)       | (0.0028)      |                 |
| Parameters | α 2 (Unobs. State)              | 0.3       | 0.2996             |                  | 0.2982          | 0.2993         | 0.2987        |                 |
|            |                                 |           | (0.0045)           |                  | (0.0119)        | (0.0117)       | (0.0116)      |                 |
|            | α 3 (No. of Competitors)        | -0.4      | -0.3995            | -0.2211          | -0.3994         | -0.3989        | -0.3984       |                 |
|            |                                 |           | (0.0061)           | (0.0051)         | (0.0087)        | (0.0088)       | (0.0089)      |                 |
|            | π (Persistence of Unobs. State) | 0.7       |                    |                  | 0.7002          | 0.7030         | 0.7032        | 0.7007          |
|            |                                 |           |                    |                  | (0.0122)        | (0.0146)       | (0.0146)      | (0.0184)        |
|            | Time (Minutes)                  |           | 0.1354             | 0.1078           | 21.54           | 27.30          | 15.37         | 16.92           |
|            |                                 |           | (0.0047)           | (0.0010)         | (1.5278)        | (1.9160)       | (0.8003)      | (1.6467)        |

† Mean and standard deviations for 100 simulations. Observed data consist of 3000 markets for 10 periods with 6 firms in each market. In column (7), the CCP's are updated with the model. See text and appendix for additional details.

initial choices lead to the same distribution of states after a few periods, then estimation requires only conditional choice probabilities for a few periods ahead.

We further show how to accommodate unobserved heterogeneity via finite mixture distributions into CCP estimation. The computational simplicity of the estimator extends to unobserved state variables that follow a Markov chain. Our baseline algorithm iterates between updating the conditional probabilities of being in a particular unobserved state, updating the CCP's for any given state (observed and unobserved), and maximizing a likelihood function where the future values terms are in large part functions of the CCP's.

When the transition on the unobserved states and the CCP's are identified without imposing the structure of the underlying model, it is possible to estimate the parameters governing the unobserved heterogeneity in a first stage. We update the CCP's using the unrestricted distribution of discrete choices weighted by the estimated probabilities of being in particular unobserved states. This approach provides a first stage estimator for blending unobserved heterogeneity into non-likelihood based approaches such as Hotz et al. (1994) and Bajari, Benkard, and Levin (2007) in a second stage to recover the remaining structural parameters.

Our estimators are √ N consistent and asymptotically normal. We undertake two Monte Carlo studies, modeling a dynamic optimization problem and a dynamic game, to investigate small sample performance. These studies indicate that substantial computational savings can result from using our estimators with little loss of precision.

## References

- [1] Altug, S., and R. A. Miller (1998): 'The Effect of Work Experience on Female Wages and Labour Supply,' Review of Economic Studies , 62, 45-85.
- [2] Aguirregabiria, V., and P. Mira (2002):'Swapping the Nested Fixed Point Algorithm: A Class of Estimators for Discrete Markov Decision Models,' Econometrica , 70, 1519-1543.
- [3] Aguirregabiria, V., and P. Mira (2007): ' Sequential Estimation of Dynamic Discrete Games,' Econometrica, 75, 1-54.
- [4] Aguirregabiria, V., and P. Mira (2010): 'Dynamic Discrete Choice Structural Models: A Survey,' Journal of Econometrics , 156(1), 38-67.

- [5] Arcidiacono, P. (2005): 'Affirmative Action in Higher Education: How do Admission and Financial Aid Rules Affect Future Earnings?' Econometrica , 73, 1477-1524.
- [6] Arcidiacono, P., and J. B. Jones (2003): 'Finite Mixture Distributions, Sequential Likelihood, and the EM Algorithm,' Econometrica , 71, 933-946.
- [7] Arcidiacono, P., and R.A. Miller (2011):'Identification of Dynamic Discrete Choice Models with Short Panels', working paper.
- [8] Arcidiacono, P., H. Seig, and F. Sloan (2007):'Living Rationally Under the Volcano? An Empirical Analysis of Heavy Drinking and Smoking.' International Economic Review , 48(1).
- [9] Bajari, P., L. Benkard, and J. Levin (2007): ' Estimating Dynamic Models of Imperfect Competition,' Econometrica, 75, 1331-1371.
- [10] Beauchamp, A. (2010): 'Abortion Supplier Dynamics', working paper.
- [11] Beresteanu A., P.B. Ellickson, and S. Misra (2010): 'The Dynamics of Retail Oligopoly,' working paper.
- [12] Bishop, K.C. (2008):'A Dynamic Model of Location Choice and Hedonic Valuation,' working paper.
- [13] Bresnahan, T. F., S. Stern, and M. Trajtenberg (1997): 'Market Segmentation and the Sources of Rents from Innovation: Personal Computers in the Late 1980s,' RAND Journal of Economics , 28, 17-44.
- [14] Buchinsky, M., J. Hahn, and V. J. Hotz (2005): 'Cluster Analysis: A Tool for Preliminary Structural Analysis,' working paper.
- [15] Chung, D., T. Steenburgh, and K. Sudhir (2009):'Do Bonuses Enhance Sales Productivity? A Dynamic Structural Analysis of Bonus-Based Compensation Plans,' working paper.
- [16] Collard-Wexler, A. (2008): 'Demand Fluctuations and Plant Turnover in the Ready-Mix Concrete Industry,' working paper.
- [17] Dempster, A. P., N. M. Laird, and D. B. Rubin (1977): 'Maximum Likelihood from Incomplete Data via the EM Algorithm,' Journal of Royal Statistical Society Series B, 39,1-38

- [18] Dunne, T., Klimek S., Roberts, M. and Y. Xu (2009): 'Entry, Exit and the Determinants of Market Structure,' NBER Working Paper #15313.
- [19] Eckstein, Z., and K. Wolpin (1999):'Why Youths Drop out of High School: The Impact of Preferences, Opportunities, and Abilities,' Econometrica , 67(6), 1295-1339.
- [20] Finger, S.R. (2008): 'Research and Development Competition in the Chemicals Industry,' working paper.
- [21] Gowrisankaran, G., C. Lucarelli, P. Schmidt-Dengler, and R. Town (2009):'Government policy and the dynamics of market structure: Evidence from Critical Access Hospitals,' working paper.
- [22] Gowrisankaran, G. and R. Town (1997):'Dynamic equilibrium in the hospital industry,' Journal of Economics, Management, and Strategy , 6, 45-74.
- [23] Gayle, G., and L. Golan (2007): 'Estimating a Dynamic Adverse Selection Model: Labor Force Experience and the Changing Gender Earnings Gap 1968-1993,' working paper.
- [24] Gayle, G., and R. A. Miller (2006): 'Life Cycle Fertility and Human Capital Accumulation,' working paper.
- [25] Hansen, L. P. (1982): 'Large Sample Properties of Generalized Methods of Moments Estimators,' Econometrica, 50, 1029-1054.
- [26] Hamilton, J. D. (1990): 'Analysis of Time Series Subject to Changes in Regime,' Journal of Econometrics , 45, 39-70.
- [27] Heckman, J.J. (1981):'The Incidental Parameters Problem and the Problem of Initial Conditions in Estimating a Discrete Time-Discrete Data Stochastic Process,' in Structural Analysis of Discrete Data with Econometric Applications , ed. C.F. Manski and D. McFadden. Cambridge, MA: MIT Press, 179-195.
- [28] Heckman, J.J., and S. Navarro (2007):'Dynamic Discrete Choice and Dynamic Treatment Effects,' Journal of Econometrics , 136, 341-396.
- [29] Heckman, J.J. and B. Singer (1984):'A Method for Minimizing the Impact of Distributional Assumptions in Econometric Models for Duration Data,' Econometrica , 52(2), 271-320.

- [30] Hotz, V. J., and R. A. Miller (1993): ' Conditional Choice Probabilities and Estimation of Dynamic Models,' Review of Economic Studies , 61, 265-289.
- [31] Hotz, V. J., R. A. Miller, S. Sanders and J. Smith (1994): 'A Simulation Estimator for Dynamic Models of Discrete Choice,' Review of Economic Studies, 60, 265-289.
- [32] Hu, Y. and M. Shum (2010a):'Nonparametric identification of dynamic models with unobserved state variables,' working paper.
- [33] Hu, Y. and M. Shum (2010b):'A Simple Estimator for Dynamic Models with Serially Correlated Unobservables,' working paper.
- [34] Imai, S., N. Jain, and A. Ching (2009): 'Bayesian Estimation of Dynamic Discrete Choice Models,' Econometrica , 77(6), 1865-1899.
- [35] Jamshidian, Mortaza, and Robert Jennrich (1997): 'Acceleration of the EM Algorithm by using Quasi-Newton Methods,' Journal of the Royal Statistical Society , B 59, pp. 569-587.
- [36] Joensen, J.S. (2009): 'Academic and Labor Market Success: The Impact of Student Employment, Abilities, and Preferences,' working paper.
- [37] Joefre-Bonet, M. and M. Pesendorfer (2003):'Estimation of a Dynamic Auction Game,' Econometrica , 71(5), pp. 1443-1489.
- [38] Kasahara, H. and K. Shimotsu (2008): 'Pseudo-likelihood Estimation and Bootstrap Inference for Structural Discrete Markov Decision Models,' Journal of Econometrics , 146(1) 92-106.
- [39] Kasahara, H., and K. Shimotsu (2009): 'Nonparametric Identification and Estimation of Finite Mixture Models of Dynamic Discrete Choices,' Econometrica , 77(1), 135-175.
- [40] Keane, M., and K. Wolpin (1994): 'The Solution and Estimation of Discrete Choice Dynamic Programming Models by Simulation and Interpolation: Monte Carlo Evidence' The Review of Economics and Statistics , 76, 648-672.
- [41] Keane, M., and K. Wolpin (1997): 'The Career Decisions of Young Men,' Journal of Political Economy, 105, 473-522.
- [42] Keane, M., and K. Wolpin (2000):'Eliminating Race Differences in School Attainment and Labor Market Success,' Journal of Labor Economics .

- [43] Keane, M., and K. Wolpin (2001):'The Effect of Parental Transfers on Borrowing Constraints,' International Economic Review , 42(4), 1051-1103.
- [44] Kennan, J. and J.R. Walker (2011):'The Effect of Expected Income on Individual Migration Decisions,' Econometrica , 79(1), 211-251.
- [45] McFadden, D. (1978): 'Modelling the Choice of Residential Location,' Spatial Interaction and Planning Models, editors F. Snickars and J. Weibull, North Holland Amsterdam, 75-96.
- [46] McLachlan, G.J. and D. Peel (2000): Finite Mixture Models , Wiley series in probability and statistics, New York.
- [47] Miller, R. A. (1984): 'Job Matching and Occupational Choice,' Journal of Political Economy, 92, 1086-1020.
- [48] Miller, R.A. and S. Sanders (1997):'Human capital development and welfare participation,' Carnegie Rochester Conference Series on Public Policy , Vol. 46, pp. 1-45.
- [49] Mroz, T. (1999):'Discrete Factor Approximations for Use in Simultaneous Equation Models: Estimating the Impact of a Dummy Endogenous Variable on a Continuous Outcome,' Journal of Econometrics , 129, 994-1001.
- [50] Newey, W. K., and D. McFadden (1994): 'Large Sample Estimation and Hypothesis Testing,' Chapter 36 in Handbook of Econometrics, 4, editors R. Engle and D. McFadden, Elsevier Science, Amsterdam, 2111-2245.
- [51] Norets, A. (2009):'Inference in Dynamic Discrete Choice Models with Serially Correlated Unobserved State Variables', Econometrica , 77(5), 1665-1682.
- [52] Pakes, A. (1986): 'Patents as Options: Some Estimates of the Value of Holding European Patent Stocks,' Econometrica, 54, 755-784.
- [53] Pakes, A., M. Ostrovsky, and S. Berry (2007): ' Simple Estimators for the Parameters of Discrete Dynamic Games (with Entry/Exit Examples),' RAND Journal of Economics , 38(2), 373-399.
- [54] Pesendorfer, M., and P. Schmidt-Dengler (2008): ' Asymptotic Least Square Estimators for Dynamic Games,' Review of Economic Studies , 75, 901-908.

- [55] Pesendorfer, M. and P. Schmidt-Dengler (2010): 'Sequential Estimation of Dynamic Discrete Games: A Comment', Econometrica , Vol. 78, No. 2, 833-842.
- [56] Rust, J. (1987): 'Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher,' Econometrica , 55, 999-1033.
- [57] Rust, J. (1994):'Structural Estimation of Markov Decision Processes,' in Handbook of Econometrics, Volume 4 , ed. by R.E. Engle and D. McFadden. Amsterdam: Elsevier-North Holland, Chapter 14.
- [58] Ryan, S. (2009): 'The Costs of Environmental Regulation in a Concentrated Industry,' working paper.
- [59] Su, C. and K.L. Judd (2009): 'Constrained Optimization Approaches to Estimation of Structural Models,' working paper.
- [60] Wolpin, K. (1984): 'An Estimable Dynamic Stochastic Model of Fertility and Child Mortality,' Journal of Political Economy, 92, 852-874.
- [61] Wu, C. F. (1983): 'On the Convergence Properties of the EM Algorithm,' The Annals of Statistics, 11, 95-103.

## A Proofs

Proof of Lemma 1. Equation (3 . 3) implies V t ( z t ) can be written as:

<!-- formula-not-decoded -->

Subtracting v kt ( z t ) from both sides yields:

<!-- formula-not-decoded -->

From Proposition 1 of Hotz and Miller (1993, page 501), there exists a mapping ψ (1) k ( ) for each j ∈ { 1 , ..., J } such that:

which implies:

They also prove ( A. 3) implies there exists a mapping ψ (2) j ( ) for each j ∈ { 1 , ..., J } such that:

/D4

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Substituting ( A. 4) and ( A. 5) into ( A. 2) completes the proof:

<!-- formula-not-decoded -->

Proof of Theorem 1. The proof is by backward induction. We first establish that it holds when the time horizon is T and where the decision is made at T ′ and when T ′ + 1 = T . We then show that if it holds for a generic T ′ where 1 &lt; T ′ &lt; T then it also holds at T ′ -1, completing the proof. Noting that v kT ( z T ) ≡ u k ( z T ) for all k ∈ { 1 , . . . , J } and z T ∈ { 1 , . . . , Z } , including those in the decision rule d ∗ kT ′ ( z T ′ , j ), and noting that when T ′ +1 = T , equation (3.6) can be expressed as:

<!-- formula-not-decoded -->

establishing that the theorem holds for t = T -1.

Setting T ′ such that 1 &lt; T ′ &lt; T and assuming (3.8) holds implies:

<!-- formula-not-decoded -->

Moving back to T ′ -1, equation (3.6) implies:

<!-- formula-not-decoded -->

Substituting for v kT ′ ( z T ′ ) in (A.9) with (A.8) completes the proof.

<!-- formula-not-decoded -->

Now consider the infinite horizon problem. For t &lt; T ′ , we can express v jt ( z t ) as:

<!-- formula-not-decoded -->

We can bound | V T ′ +1 ( z T ′ +1 ) | by V , which implies:

since

<!-- formula-not-decoded -->

It now follows from ( A. 11) that for all T ′ : ∣ ∣ ∣ ∣ ∣ v jt ( z t ) -u jt ( z t ) -T ′ ∑ τ = t +1 J ∑ k =1 Z ∑ z τ =1 β τ -t [ u kτ ( z τ ) + ψ k [ p τ ( z τ )]] d ∗ kτ ( z τ , j ) κ ∗ τ -1 ( z τ | z t , j ) ∣ ∣ ∣ ∣ ∣ ≤ β T ′ -t +1 V Since β &lt; 1 , the term β T ′ -t +1 V → 0 as T ′ →∞ , proving the theorem. Proof of Lemma 2. Define /DA j ≡ ln Y j and let G j ( ε ) ≡ ∂G ( ε ) /ε j . Let H ≡ H ( e /DA 1 , e /DA 2 , . . . , e /DA J ) . Since H ( Y 1 , Y 2 , . . . , Y J ) is homogeneous of degree one, and therefore the partial derivative H j ( Y 1 , Y 2 , . . . , Y J ) is homogenous of degree zero: G j ( /DA j + ε j -/DA 1 , . . . , /DA j + ε j -/DA J ) = H j ( e /DA 1 , . . . , e /DA J ) exp [ -H e -/DA j -ε j ] e -/DA -ε j From Theorem 1 of McFadden (1978, page 80), integrating over G j ( ε t ) yields the conditional choice probability: /D4 j = ∫ G j ( /DA j + ε j -/DA 1 , . . . , ε j , . . . , /DA j + ε j -/DA J ) dε j (A.12) = e /DA j -/DA 1 H j [ 1 , e /DA 2 -/DA 1 , . . . , e /DA J -/DA 1 ] / H [ 1 , e /DA 2 -/DA 1 , . . . , e /DA J -/DA 1 ] ≡ φ j ( 1 , e /DA 2 -/DA 1 , . . . , e /DA J -/DA 1 ) By Proposition 1 of Hotz and Miller (1993) we can invert the vector function: /D4 /DA /DA /DA /DA

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 3. From (3 . 18):

H ( Y 1 , . . . , Y J ) = H 0 ( Y 1 , . . . , Y K ) + (∑ j ∈J Y 1 /σ j ) σ (A.16) For all j ∈ J , the formula for φ j ( Y ) for the nested logit components are: φ j ( Y ) = Y 1 /σ j ( ∑ j ∈J Y 1 /σ j ) σ -1 H ( Y 1 , . . . , Y J ) Let φ -1 ( /D4 ) ≡ ( φ -1 2 ( /D4 ) , . . . , φ -1 J ( /D4 )) denote the inverse of φ ( Y ) ≡ ( φ 1 ( Y ) , . . . , φ J -1 ( Y )). Then from ( A. 12):

<!-- formula-not-decoded -->

Summing over k ∈ J and taking the quotient yields:

/D4

<!-- formula-not-decoded -->

Proof of Theorem 2. Proof of Part 1. For convenience we consolidate the structural parameters into the vector λ ≡ ( θ, π ) . Denote the true parameters and conditional choice probabilities by λ 0 and p 0 respectively. Let l ( λ, /D4 ) denote the corresponding vector of likelihoods associated each choice probability, implying p 0 = l ( λ 0 , p 0 ). For each N define Λ N as the set of parameters solving (4 . 5) at p = p, where ( ̂ θ, π, p ) simultaneously satisfies (4 . 6):

which implies by direct verification that: φ -1 j ( ) = A σ j (A.18) where A is unknown but greater than zero. Substituting in for φ -1 j ( ) in (A.17) with ( A. 18) , we obtain for each choice j ∈ J : /D4 j = A 1 /σ /D4 j ( ∑ k ∈J A 1 /σ /D4 k ) σ -1 H ( 1 , φ -1 2 ( /D4 ) , . . . , φ -1 J ( /D4 )) = A /D4 j ( ∑ j ∈J /D4 j ) σ -1 H ( 1 , φ -1 2 ( /D4 ) , . . . , φ -1 J ( /D4 )) which implies: H ( 1 , φ -1 2 ( /D4 ) , . . . , φ -1 J ( /D4 )) = A (∑ k ∈J /D4 k ) ( σ -1) (A.19) We can now substitute in (A.19) and (A.18) into the expression for ψ j ( /D4 ) given in (A.15), completing the proof. ψ j ( /D4 ) = ln [ A (∑ j ∈J /D4 j ) ( σ -1) ] -ln [ A /D4 σ j ] + γ = γ -σ ln( /D4 j ) -(1 -σ ) ln (∑ k ∈J /D4 k )

<!-- formula-not-decoded -->

Also define the set of parameters that maximize the corresponding expected log likelihood subject to the same constraint as:

<!-- formula-not-decoded -->

By definition λ 0 ∈ Λ 1 because ( λ 0 , p 0 ) solves:

<!-- formula-not-decoded -->

/negationslash

Appealing to the continuity of L ( d n , x n | x n 1 ; λ, p N ) and p ( λ N ) , the weak uniform law of large numbers implies there exists a sequence ̂ λ N ∈ Λ N converging to λ 0 . Now consider sequences ˜ λ N ∈ Λ N that converge to other elements in Λ 1 , say λ 1 = λ 0 . The assumption of identification implies that for all λ 1 = λ 0 :

<!-- formula-not-decoded -->

By continuity and the law of large numbers:

<!-- formula-not-decoded -->

This proves that choosing the element which maximizes the criterion function, ̂ λ N , from the set of fixed points, Λ N , selects a weakly consistent estimator for λ 0 .

Proof of Part 2. For each t define the joint distribution of ( x, s ) , induced by the parameter vector ( λ, p ) and the data, as:

<!-- formula-not-decoded -->

By the law of large numbers, for each x, the X × S -1 dimensional random variable P N t ( x, s, ̂ λ, ̂ p ) converges in probability to:

Similarly the joint distribution of ( j, x, s ) is defined at t as:

<!-- formula-not-decoded -->

which converges in probability to:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let P N ( λ, p ) denote the T × ( J -1) × X × S dimensional vector formed from components P N t ( j, x, s, λ, p ) / P N t ( x, s, λ, p ), and let P ( λ, p ) denote the vector of corresponding limit points P t ( j, x, s, λ, p ) / P t ( x, s, λ, p ) . Then the parameters solving the fixed point characterized by (4 . 5) and (4 . 8) are elements of the set defined by:

<!-- formula-not-decoded -->

/negationslash

and similar to Part 1, elements in Λ ′ N converge weakly to elements in the set:

<!-- formula-not-decoded -->

Noting that ( λ 0 , p 0 ) ∈ Λ ′ 1 , the arguments in Part 1 can be repeated to complete the proof that the fixed point solution in Λ ′ N achieving the highest value of (4 . 4) is consistent.

## A.1 Asymptotic Covariance Matrix

The asymptotic covariance matrix of our estimators are derived from Taylor expansions of two sets of equations, the first order conditions of (4 . 5) for λ, and a set of equations that solve the conditional choice probability nuisance parameter vector p. The first order conditions of (4 . 5) can be written as:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since ( ̂ λ, ̂ p ) is consistent and L λn ( λ, p ) is continously differentiable, we can expand ( A. 20) around ( λ 0 , p 0 ) to obtain:

where

<!-- formula-not-decoded -->

The first estimator sets ̂ p to solve (4 . 6) for each ( j, t, x, s ) . Stacking l jt ( x, s ; λ, p ) for each choice ( j, t ) (time indexed in the nonstationary case), and each value ( x, s ) of state variables to form l ( λ, p ) , the ( J -1) × T × X × S vector function of the CCP parameters ( λ, p ) , our estimator satisfies the ( J -1) TXS additional parametric restrictions l ( ̂ λ, ̂ p ) = ̂ p. From the identity we expand the second equation to the first order and rearrange to obtain:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using ( A. 22) we substitute out √ N ( p -p 0 ) in ( A. 21) , which yields:

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Appealing to the central limit theorem, and the fact that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

the asymptotic covariance matrix for √ N ( ̂ λ -λ 0 ) is thus:

In the second estimator, the condition that l ( ̂ λ, ̂ p ) = ̂ p is replaced by the ( J -1) TXS equalities in (4 . 8) . Define:

<!-- formula-not-decoded -->

where L n ≡ L ( d n , x n | x n 1 ; λ, p ) , and L n ( s nt = s ) is given by (4 . 7) evaluated at ( λ, p ) . For each sample observation n , stack Q njtxs ( λ, p ) to form the ( J -1) TXS dimensional vector Q n ( λ, p ) . In vector form (4 . 8) can then be expressed as:

<!-- formula-not-decoded -->

We form the vector h n ( λ, p ), the expected outer product of h n ( λ, p ), and its square derivative matrix:

<!-- formula-not-decoded -->

From Hansen (1982, Theorem 3.1) or Newey and McFadden (1994, Theorem 6.1), it now follows that √ N ( ̂ λ -λ 0 ) is asymptotically normally distributed with mean zero and covariance matrix given by the top left square block of Γ -1 ΩΓ -1 ′ with dimension of λ .

## B Additional Information on the Monte Carlo Exercises

All simulations were conducted in Matlab version 7.5 on the Duke economics department 64-bit batch cluster. The code was not parallelized. The cluster and the operating system of Matlab ensures one processor is dedicated to each Matlab job. All non-linear optimization was done using Matlab's

canned optimization routine, fminunc with the default values used to determine convergence. No derivatives were used in the maximization routines for the structural parameters. Convergence for the EM algorithm was determined by comparing log likelihood values 25 iterations apart. The algorithm was stopped when this difference was less than 10 -7 for two successive iterations.

## B.1 Optimal Stopping

This subsection provides further computational details about the optimal stopping problem. We discuss the data generating process, as well as updating the conditional choice probabilities and the parameters governing the initial conditions.

## B.1.1 Data Creation

For the true parameter values and the transition matrix for mileages implied by (7.2) and reported in the first column of Table 1, we obtain the value functions by backwards recursion for every possible mileage, observed permanent characteristic, unobserved state, and time. We draw permanent observed and unobserved characteristics from discrete uniform distributions with support 101 and 2 respectively, and start each bus at zero mileage. Given the parameters of the utility function, the value function, and the permanent observed and unobserved states, we calculate the probability of a replacement occurring in the first period. We then draw from a standard uniform distribution. If the draw is less than the probability of replacement, the decision in the first period is to replace. Otherwise we keep the engine. Conditional on the replacement decision, we draw a mileage transition using (7 . 2). Continuing this way, decisions and mileage transitions are simulated for thirty periods.

## B.1.2 The likelihood

Conditional on the permanent observed state, the mileage and the unobserved state s , the likelihood of a particular decision at time t takes a logit form. The likelihoods for the FIML and CCP cases are respectively given by:

<!-- formula-not-decoded -->

When s is unobserved, the log likelihood for a particular bus history is found by first taking the product of the likelihoods over time conditional on type, and then summing across the types inside the logarithm. Thus in the FIML case, the likelihood is: 32

<!-- formula-not-decoded -->

In the CCP case, L t ( d t | x t , s ; θ ) is replaced by L t ( d t | x t , s, p ; θ ).

## B.1.3 Conditional choice probability estimates

We approximate (5 . 9) , the second estimator for the conditional choice probabilities with a flexible logit, where the dependent variable is d 1 t . There are five cases:

1. To obtain the estimates reported in Column 4 of Table 1 (when s is ignored), we estimate the CCP's using W 1 t ≡ ( 1 , x 1 t , x 2 1 t , x 2 , x 2 2 , x 1 t x 2 ) as regressors in a logit. 33
2. For the parameters reported in Column 3, W 1 t is fully interacted with W 2 t ≡ ( 1 , s, t, st, t 2 , st 2 ) , that is 36 parameters to estimate in the logit generating the CCP's. Since s is observed, this flexible logit is estimated once.
3. When s is unobserved, the flexible logit described in the previous case is estimated at each iteration of the EM algorithm; at the m th iteration the conditional probabilities of being in each observed state, q ( m ) s , are used to weight the flexible logit.
4. For the last two columns, where there are aggregate effects, we fully interact the first set of variables with the s , but not t and t 2 . Instead, we include time dummies, but given the moderate sample size, we did not interact them with the other variables. Hence the logits were estimated with 21 regressors: 12 combinations of x 1 t , x 2 , and s , as well as 9 time dummies.

## B.1.4 Initial conditions

Initial probabilities are specified as a flexible function of the first period observables, denoted by W 0 . Included in W 0 are the mileage at the first observed time period for the bus, x 11 , as well as the

32 Since we are taking products of potentially small probabilities, numerical issues could arise. These can be solved by scaling up the L t ( d t | x t , s ; θ, p ) terms by a constant factor. However, in neither of our Monte Carlos was this an issue.

33 For a sufficiently large but finite sample we can saturate the finite set of regressors with a flexible logit that yields numerically identical estimates as the weighted bin estimator presented in the text.

permanent observed characteristic, x 2 . The prior probability of being in unobserved state 2 during the first observed period in the data, t = 1, given the data for n is given by:

<!-- formula-not-decoded -->

At iteration m , we calculate the likelihood for each data point conditional on the unobserved state. Under FIML:

The iterate δ ( m +1) solves:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the CCP case, we replace L t ( d t | x t , s ; θ ( m ) ) with L t ( d t | x t , s, p ( m ) ; θ ( m ) ) , and replace L ( d n , s n = s | x n ; θ ( m ) ) with L ( d n , s n = s | x n , p ( m ) ; θ ( m ) ) . 34

## B.1.5 Creation of time-varying intercepts

In the case where the replacement costs varied over time (Column 8 of Table 1), we create the data by drawing values for the intercept from a normal distribution with standard deviation of 0.5. The value of θ 0 t +1 , is set to 0 . 7 θ 0 t plus the value drawn at t +1 from the normal distribution.

## B.2 Entry/Exit

We now turn to the details of the Monte Carlo for the dynamic game. Again we describe the data creation as well as the variables used in both the conditional choice probabilities and in the reduced form controls for the initial conditions problem.

## B.2.1 Data creation

The first step in creating the data is to obtain the probability of entering for every state. Equation (7.8) gives the flow payoff for being in the market conditional on the choices of the other firms. Note that the expected flow payoff of entering depends on the probabilities of other firms entering. Given initial guesses for the probability of exiting in each state, we form all the possible combinations of the entry decisions of the other firms using (7.4). We then substitute (7.4) and (7.8) into (7.5) to form the expected flow payoff of staying in or entering the market in every state. Since the transitions

34 The saturation argument we mentioned in the previous footnote applies here too.

on the state variables conditional on the entry/exit decisions are known, we have all the pieces to form (7.9). Given (7.9), the Type 1 extreme value assumption implies the probability of exiting is 1 / (1+exp( v ( i ) 2 ( x t , s t )). We can then update the entry exit probabilities used to form (7.4). We then iterate on (7.4), (7.5), (7.9), and the logit probability of exiting until a fixed point is reached. 35

The observed permanent market characteristics and the initial unobserved states were drawn from a discrete uniform distribution. We then began each market with no incumbents and simulated the model forward. We then removed the first ten periods of data from the sample.

## B.2.2 The likelihood

We now derive the likelihood at time t for market n of the observed decisions and price process given the data and the parameters. Note that x nt +1 , which includes the permanent market characteristic as well as the incumbency status of each of the firms, is a deterministic function of x nt and y nt . The likelihood contribution for the i th firm at time t conditional on unobserved state s t is:

Denote E ( y t ) = α 0 + α 1 x 1 + α 2 s t + α 3 ∑ I i =1 d ( i ) 2 t . Denoting n as the market, the likelihood of the data in market n at time t conditional on s t is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where φ ( · ) is the density function of the standard normal distribution and σ is the standard deviation of η t .

We can then substitute (B.1) into (4.3) to obtain the likelihood of the data for a particular market:

<!-- formula-not-decoded -->

To make clear the number of calculations required to form the expression in (B.2) for a particular market, we specify (B.2) using matrix notation. Denote A nt as a 1 × S vector with components given by the likelihood at t = 1 conditional on a particular unobserved states times the initial probability of being in the unobserved state:

<!-- formula-not-decoded -->

35 Multiple equilibria may be a possibility. This issue did not cause any problems for this set of Monte Carlos.

If T = 1, summing over the elements of A n 1 would give L ( d n , y n , x n | x n 1 ; θ, α, π, p ). For t &gt; 1, we form an S × S matrix where the ( i, j ) element gives the probability of moving from s t -1 = i to s t = j times the likelihood contribution at t conditional on being in unobserved state j :

<!-- formula-not-decoded -->

Taking A n 1 times A n 2 gives a 1 × S vector of the joint likelihood of the data and being in each of the unobserved states. We define A n as the product of A nt over T :

<!-- formula-not-decoded -->

A n is then a row vector with S elements with each element giving the joint likelihood of the data and being in a particular unobserved state at T . To form A n , an S × S matrix is multiplied by an 1 × S matrix T times. Let the s th element be denoted by A n ( s ). The likelihood for the n th market is then given by:

<!-- formula-not-decoded -->

## B.2.3 Obtaining conditional choice probabilities

Four sets of CCP's are used in this Monte Carlo:

1. When s t is ignored (Column 3 of Table 2), we specify the conditional probability of exiting at t + 1 as a flexible function of the observed variables, W 1 ti (for the i th firm in a given market at time t ). The variables included in W 1 ti are combinations of the permanent market characteristic, x 1 , whether the firm is active in period t , d ( i ) 2 t , and the number of firms in the market at t :

We then estimate a logit on the probability of exiting using the variables in W 1 ti as controls.

<!-- formula-not-decoded -->

2. When s t is observed (Column 2), we add the variables in W 2 ti to the logit, where:

<!-- formula-not-decoded -->

implying 10 parameters govern the CCP's.

3. When the conditional choice probabilities are updated with the data (Column 5), and when using the two-stage method (Column 6), we use the variables in both W 1 t and W 2 t . In both these cases, the m th iteration uses the conditional probabilities of being in each unobserved state, q ( m ) nst , as weights in the logit estimation.
4. Finally, when the CCP's are updated with the model (Columns 4 and 7), we update the probability of exiting using the logit formula for the likelihood:

<!-- formula-not-decoded -->

## B.2.4 Initial conditions

There is an initial conditions problem in the stationary equilibrium, because the distribution of s 1 depends on the the distribution of the observed states. We estimate this distribution jointly with the other parameters of the model. Since the unobserved state applies at the market level of aggregation, the relevant endogenous variable is the lagged number of firms in the initial period. We regress the lagged number of firms in the initial period on a flexible function of the characteristics of the market, in this exercise, a constant, x 1 , and x 2 1 . Denote the residual from this regression as ζ . We then approximate the initial probability of being in unobserved state s for the n th market using a multinomial logit form:

<!-- formula-not-decoded -->

With δ 1 set to zero, there are 8 parameters to be estimated. We estimate π ( s | x n 1 ) at each iteration using a similar procedure to Section B.1.4, now allowing for the fact that the unobserved states follow a Markov transition. Despite this additional complication, the algorithm is the same: calculate the likelihood given each initial unobserved state and take it as a given when maximizing to update δ .