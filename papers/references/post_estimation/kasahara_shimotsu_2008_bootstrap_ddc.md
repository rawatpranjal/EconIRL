## Pseudo-likelihood Estimation and Bootstrap Inference for Structural Discrete Markov Decision Models ∗

Hiroyuki Kasahara Department of Economics University of Western Ontario hkasahar@uwo.ca

Katsumi Shimotsu Department of Economics Queen's University shimotsu@econ.queensu.ca

June 11, 2008

## Abstract

This paper analyzes the higher-order properties of the estimators based on the nested pseudo-likelihood (NPL) algorithm and the practical implementation of such estimators for parametric discrete Markov decision models. We derive the rate at which the NPL algorithm converges to the MLE and provide a theoretical explanation for the simulation results in Hotz et al. (1994) and Aguirregabiria and Mira (2002), in which iterating the NPL algorithm improves the accuracy of the estimator. We then propose a new NPL algorithm that can achieve quadratic convergence without fully solving the fixed point problem in every iteration and apply our estimation procedure to a finite mixture model. We also develop one-step NPL bootstrap procedures for discrete Markov decision models. The Monte Carlo simulation evidence based on a machine replacement model of Rust (1987) shows that the proposed one-step bootstrap test statistics and confidence intervals improve upon the first order asymptotics even with a relatively small number of iterations.

Keywords: Edgeworth expansion, finite mixture, k -step bootstrap, maximum pseudolikelihood estimators, nested fixed point algorithm, Newton-Raphson method, policy iteration.

JEL Classification Numbers: C12, C13, C14, C15, C44, C63.

## 1 Introduction

Understanding the dynamic response of individuals and firms is imperative for properly assessing various policy proposals. As numerous empirical studies have demonstrated, the estimation

∗ The authors are grateful to the co-editor and two anonymous referees whose comments greatly improved the paper. The authors thank Victor Aguirregabiria, Chris Bennett, Christopher Ferrall, Silvia Gon¸ calves, Lance Lochner, James MacKinnon, John Rust, Steven Stern, and seminar participants at Canadian Econometric Study Group, Indiana University, University of Maryland, University of Virginia, University of Western Ontario, Queen's University, and York University for helpful comments. The authors thank the SSHRC for financial support. This work was made possible by the facilities of the Shared Hierarchical Academic Research Computing Network (SHARCNET:www.sharcnet.ca).

of dynamic structural models enhances our understanding of individual and firm behavior, especially when expectations play a major role in decision making. 1

The literature on estimating parametric discrete Markov decision models was pioneered by Gotz and McCall (1980), Wolpin (1984), Miller (1984), Pakes (1986), and Rust (1987, 1988). Such models are typically estimated by the nested fixed point algorithm (NFXP) which requires repeatedly solving the fixed point problem during optimization and can be very costly when the dimensionality of state space is large. Hotz and Miller (1993) developed a simpler estimator called the conditional choice probabilities estimator (CCP) based on the policy iteration mapping-denoted by Ψ( P, θ )-which maps an arbitrary choice probability P and the model parameter θ to another choice probability. The true choice probability is characterized as a fixed point of the mapping, i.e., P θ = Ψ( P θ , θ ). The CCP estimator minimizes the discrepancy between the observed choice probabilities and Ψ( ˆ P 0 , θ ), where ˆ P 0 is an initial estimate. The CCP estimator requires only one policy iteration to evaluate the objective function, leading to a very significant computational gain over the NFXP.

The CCP estimator has been used in empirical applications by Hotz and Miller (1993), Miller and Sanders (1997), Altug and Miller (1998), Slade (1998), Aguirregabiria (1999), and Rota (2004). A number of recent papers in empirical industrial organization build on the idea of Hotz and Miller (1993) to develop CCP-type estimators for models with multiple agents (cf., Bajari, Benkard, and Levin, 2007; Pakes, Ostrovsky, Berry, 2007; Pesendorfer and Schmidt-Dengler, 2006; Bajari and Hong, 2006; Aguirregabiria and Mira, 2007).

Aguirregabiria and Mira (2002) [henceforth, AM] extended the CCP estimator and proposed an estimator based on the nested pseudo-likelihood (NPL) algorithm. Upon obtaining ˆ θ from the CCP estimator, one can update the conditional choice probabilities estimate as ˆ P 1 = Ψ( ˆ P 0 , ˆ θ ), which provides a more accurate estimator of P θ than ˆ P 0 . Next, one can obtain another estimator of θ , ˆ θ 1 , by using Ψ( ˆ P 1 , θ ) instead of Ψ( ˆ P 0 , θ ). Iterating this procedure generates a sequence of estimators, including the CCP estimator as the initial estimator and the Maximum Likelihood Estimator (MLE) as its limit. Somewhat surprisingly, AM showed that the estimator based on the NPL algorithm for any number of iterations has the same limiting distribution as the MLE.

The NPL algorithm provides a menu of first-order equivalent estimators that empirical re-

1 Contributions include Berkovec and Stern (1991), Keane and Wolpin (1997), Rust and Phelan (1997), Gilleskie (1998), Eckstein and Wolpin (1999), and Imai and Keane (2004).

searchers can choose from, but little is known about their higher-order properties. Since the choice among these estimators involves a trade-off between efficiency and computational burden, understanding their higher-order properties is necessary for making an appropriate choice in a given situation. In fact, the simulations by AM reveal that iterating the policy iteration mapping improves the accuracy of the parameter estimates, often by a substantial magnitude, suggesting that higher-order properties may be of practical importance.

We present simulation results showing that tests based on first-order asymptotics can be unreliable. While bootstrap tests are known to provide a better inferential tool than first-order asymptotic approximations, few studies have analyzed a bootstrap-based inference method for discrete Markov decision models. The main obstacle is the computational burden, since the bootstrap requires repeated parameter estimation under different simulated samples while it is not unusual for estimating one set of the parameters to take more than a day.

The contributions of this paper are four-fold. First, we analyze the higher-order properties of the estimator based on the NPL algorithm and derive the stochastic differences [cf., Robinson (1988)] between the MLE and the sequence of the estimators generated from the NPL algorithm. We show the rate at which the sequence of these estimators approaches the MLE and provide a theoretical explanation for the simulation results in AM, in which iterating the NPL algorithm improves the accuracy of the estimator.

Second, we propose two new estimation procedures based on the NPL algorithm. First, we develop a nested modified pseudo-likelihood (NMPL) algorithm that uses a pseudo-likelihood defined in terms of two policy iterations as opposed to one policy iteration in the NPL. We show the convergence rate of the NMPL algorithm is faster than quadratic while that of the NPL is less than quadratic. Second, we propose a version of the NPL and NMPL algorithms, called the one-step NPL and NMPL algorithms, that use only one Newton-Raphson (NR) step to update the parameter θ during each iteration. By using only one NR step rather than fully solving the pseudo-likelihood problem for every iteration, we can reduce the computational cost significantly. In the context of MLE, it is known that iterating NR steps from a one-step estimator shrinks its stochastic distance from the MLE. See, among others, Pfanzagl (1974), Janssen, Jureckova, and Veraverbeke (1985), Robinson (1988). Our one-step estimators may be viewed as a version of these iterative estimators. In our context, however, it is essential that an estimate of the nuisance parameter P be sequentially updated between NR steps.

Third, we show that our estimation procedure is applicable to a model with unobserved heterogeneity. Note that, to date, the Hotz-Miller type approach has not yet been extended to handle unobserved heterogeneity because of the difficulty in obtaining initial nonparametric consistent estimators. 2 We apply our estimation procedure to a finite mixture model, which is a popular approach when preferences are likely to be different, by utilizing the recent identification results of Kasahara and Shimotsu (2007) for finite mixture models of dynamic discrete choices.

Fourth, we develop a bootstrap procedure for parametric discrete Markov decision models, applying the framework by Davidson and MacKinnon (1999a) and Andrews (2002b, 2005). The key insight is that an estimate from the original sample is within N -1 / 2 distance of the bootstrap estimate and hence provides a good starting value for the NPL/NMPL algorithms. As a result, bootstrap estimators obtained by taking a small number of NPL/NMPL iterations from the original estimate achieve higher-order improvements. Since their computational burden is substantially less than that of the NFXP, our proposed bootstrap is feasible for many discrete Markov decision models where the standard bootstrap procedure is too costly to implement.

Using a machine replacement model of Rust (1987), we examine the finite sample performance of our proposed estimators and bootstrap procedures. The bootstrap CIs perform better than the asymptotic CIs, and the one-step bootstrap CIs with a few iterations often achieve performance similar to the bootstrap CIs based on the MLE.

The remainder of the paper is organized as follows. Section 2 introduces the model. In Section 3, we analyze the NPL algorithm and propose a modification. Section 4 describes our one-step estimation algorithm. Section 5 applies our estimation procedure to a model with unobserved heterogeneity. Section 6 analyzes the higher-order improvements from applying parametric bootstrapping to the one-step NPL algorithm, and Section 7 reports some simulation results. Proofs and technical results are collected in Appendices A and B.

## 2 The Econometric Model

This section introduces the class of discrete Markov decision models considered in this paper. An agent maximizes the expected discounted sum of utilities, E [ ∑ ∞ j =0 β j U ( s t + j , a t + j ) | a t , s t ], where

2 An exception is Aguirregabiria and Mira (2007) who proposed and applied the NPL algorithm to a model with unobserved heterogeneity in the context of the models of dynamic discrete games. Imai et al. (2006) develop an MCMC estimation algorithm that accommodates unobserved heterogeneity.

s t is the vector of states and a t is an action to be chosen from the discrete and finite set A = { 1 , 2 , . . . , J } . The transition probabilities are given by p ( s t +1 | s t , a t ). The Bellman equation for this dynamic optimization problem is written as W ( s t ) = max a ∈ A { U ( s t , a ) + β ∫ W ( s t +1 ) p ( ds t +1 | s t , a ) } . From the viewpoint of an econometrician, the state vector can be partitioned as s t = ( x t , /epsilon1 t ), where x t is observable and /epsilon1 t is unobservable. We make the following assumptions.

- Assumption 1 (Additive Separability): The unobservable state variable /epsilon1 t is additively separable in the utility function so that U ( s t , a t ) = u ( x t , a t ) + /epsilon1 t ( a t ), where /epsilon1 t ( a t ) is the a -th element of the unobservable state vector /epsilon1 t = { /epsilon1 t ( a ) : a ∈ A } .
- Assumption 2 (Conditional Independence): The transition probability of the state variables can be written as p ( s t +1 | s t , a t ) = g ( /epsilon1 t +1 | x t +1 ) f ( x t +1 | x t , a t ), where g ( /epsilon1 | x ) has finite first moments and is twice differentiable in /epsilon1 uniformly in x ∈ X ; the support of /epsilon1 ( a ) is the real line for all a .
- Assumption 3: The observable state variable x t has compact support X ⊂ R d .

Assumptions 1 and 2 are first introduced by Rust (1987) and widely used in the literature. Assumption 2 implies that any statistical dependence between /epsilon1 t and /epsilon1 t +1 is transmitted through x t +1 . This enables one to integrate out /epsilon1 t +1 from W ( s t +1 ) and define the integrated value function, V ( x ), as a fixed point on a smaller state space (the support of x ).

In practice, it is often assumed that /epsilon1 t has Type I extreme value distribution which yields closed-form expressions for the choice probabilities as well as for the expectations of future utility. 3 We may allow /epsilon1 t to obey a normal for a binary case as in Aguirregabiria and Mira (2007) while the nested logit specification can be used for a multivariate case to allow correlations across alternatives (cf., Arcidiacono, 2004; Kasahara and Lapham, 2007). 4

Define the integrated value function V ( x ) = ∫ W ( x, /epsilon1 ) g ( d/epsilon1 | x ), and let B V be the space of V ≡ { V ( x ) : x ∈ X } . The Bellman equation can be rewritten in terms of this integrated value

3 Examples include Rust (1987), Das (1992), Kennet (1994), Ahn (1995), Rothwell and Rust (1997), Rust and Phelan (1997), Gilleskie (1998), G¨ on¨ ul (1999), Heyma (2004), and Kennan and Walker (2006).

4 It is also possible to consider a member of the generalized extreme value (GEV) distribution that allows for errors to be correlated across multiple nests while still having closed-form expressions as in Arcidiacono (2005). Another possibility is to consider multinomial probit which allows flexible correlation structure but is computationally demanding due to the need for evaluating multivariate integrals numerically. An approximation method of Keane and Wolpin (1997) by backward recursion has been used for estimating the dynamic models with multivariate normally distributed errors.

function as:

<!-- formula-not-decoded -->

Let Γ( · ) be the Bellman operator defined by the right-hand side of the above Bellman equation. The Bellman equation is compactly written as V = Γ( V ).

Let P ( a | x ) denote the conditional choice probabilities of the action a given the observable state x , and let B P be the space of { P ( a | x ) : x ∈ X } . Given the value function V , P ( a | x ) is expressed as P ( a | x ) = ∫ I { a = arg max j ∈ A [ v ( x, j ) + /epsilon1 ( j )] } g ( d/epsilon1 | x ), where v ( x, a ) = u ( x, a ) + β ∫ X V ( x ′ ) f ( dx ′ | x, a ) is the choice-specific value function and I ( · ) is an indicator function. The right-hand side of this equation can be viewed as a mapping from one Banach (B-) space B V to another B-space B P . Define the mapping Λ( V ) : B V → B P as [Λ( V )]( a | x ) ≡ ∫ I { a = arg max j ∈ A [ v ( x, j ) + /epsilon1 ( j )] } g ( d/epsilon1 | x ).

We now derive the mapping from choice probabilities to value functions based on Hotz and Miller (1993). First, the Bellman equation (1) can be rewritten as

<!-- formula-not-decoded -->

where E [ /epsilon1 ( a ) | x, a ; ˜ v x , P ( a | x )] = [ P ( a | x )] -1 ∫ /epsilon1 ( a ) I { ˜ v ( x, a ) + /epsilon1 ( a ) ≥ ˜ v ( x, j ) + /epsilon1 ( j ) , j ∈ A } g ( d/epsilon1 | x ) with ˜ v ( x, a ) = v ( x, a ) -v ( x, 1) and ˜ v x ≡ { ˜ v ( x, a ) : a &gt; 1 } .

Define P x ≡ { P ( a | x ) : a &gt; 1 } . For each x , there exists a mapping from the utility differences ˜ v x to the conditional choice probabilities P x . Denote this mapping as P x = Q x (˜ v x ). Hotz and Miller (1993) showed that this mapping is invertible so that the utility differences can be expressed in terms of the conditional choice probabilities: ˜ v x = Q -1 x ( P x ). Invertibility allows us to express the conditional expectations of /epsilon1 ( a ) in terms of the choice probabilities P x as e x ( a, P x ) ≡ E [ /epsilon1 ( a ) | x, a ; Q -1 x ( P x ) , P ( a | x )].

By substituting these functions into (2), we obtain

<!-- formula-not-decoded -->

where u P ( x ) = ∑ a ∈ A P ( a | x )[ u ( x, a )+ e x ( a, P x )] and E P V ( x ) = ∑ a ∈ A P ( a | x ) ∫ X V ( x ′ ) f ( dx ′ | x, a ). Here, u P is the expected utility function implied by the conditional choice probability P x whereas

E P is the conditional expectation operator for the stochastic process { x t , a t } induced by the conditional choice probability P ( a t | x t ) and the transition density f ( x t +1 | x t , a t ).

Define P ≡ { P x : x ∈ X } . The value function implied by the conditional choice probability P is a unique solution to the linear operator equation (3): V = ( I -βE P ) -1 u P . The right-hand side of this equation can be viewed as a mapping from the choice probability space B P to the value function space B V . Define this mapping as ϕ ( P ) ≡ ( I -βE P ) -1 u P . Then we may define a policy iteration operator Ψ as a composite operator of ϕ ( · ) and Λ( · ): P = Ψ( P ) ≡ Λ( ϕ ( P )). Given the fixed point of this policy iteration operator, P , the fixed point of the Bellman equation (1) can be expressed as V = ϕ ( P ).

Before proceeding, we collect some definitions. Because P and V are infinite dimensional when x t is continuously distributed, the derivatives of Ψ, Λ, and ϕ need to be defined as Fr´ echet (F-) derivatives. For a map g : X → Y , where X and Y are B-spaces, g is F-differentiable at x iff there exists a linear and continuous map T such that g ( x + h ) -g ( x ) = Th + o ( || h || ) as h → 0 for all h in some neighborhood of zero, where || · || is an appropriate norm (e.g. sup norm, Euclidean norm if g ∈ R M ). If it exists, this T is called the F-derivative of g at x , and we let Dg ( x ) denote the F-derivative of g . Note that Dg ( x ) is an operator. When X is a Euclidean space, the F-derivative coincides with the standard derivative dg ( x ) /dx . Concepts such as the chain rule, product rule, higher-order and partial derivatives, and Taylor expansion are defined analogously to the corresponding concepts defined for the functions in Euclidean spaces. For further details the reader is referred to Zeidler (1986). Ichimura and Lee (2006) use the F-derivatives to characterize the asymptotic distribution of semiparametric M-estimators. Let D j g ( x, y ) denote the j th order F-derivative of g ( x, y ), and let D x g ( x, y ) denote the partial F-derivative of g ( x, y ) with respect to x . If x is a finite dimensional parameter, D x g ( x, y ) is equal to the standard partial derivative ∂g ( x, y ) /∂x.

One of the important properties of the policy iteration operator Ψ is that the derivative of Ψ in P is zero at the fixed point. AM proves this property in the case where the support of x t is finite. The following proposition establishes that this zero-Jacobian property also holds even when the support of x t is not finite and V does not belong to a Euclidean space.

Proposition 1 Suppose Assumptions 1 - 3 hold. Then ϕ ( · ) is F-differentiable at the fixed point P . If Ψ( · ) is F-differentiable at P , then Dϕ ( · ) = D Ψ( · ) = 0 (zero operator) if evaluated at the

fixed point P . In other words, Dϕ ( P ) ξ = D Ψ( P ) ξ = 0 for any ξ ∈ B P .

Proposition 1 implies that, in conjunction with the information matrix equality, the asymptotic orthogonality between the MLE of P and α . Because of this asymptotic orthogonality, the estimation error in P has only a second-order effect on estimation of α (see Proposition 2).

## 3 Maximum Likelihood Estimator and its Variants

We consider a parametric model by assuming that the utility function and the transition probabilities are unknown up to an L θ × 1 parameter vector θ ≡ ( θ u , θ g , θ f ), where θ u , θ g , and θ f are the parameter vectors in the utility function u , the density of unobservable state variables g , and the conditional transition probability function f , respectively. Consequently, the policy iteration operator Ψ is parameterized as Ψ( P, θ ) = Λ( ϕ ( P, θ ) , θ ). This corresponds to AM's notation Ψ θ ( P ) .

Let P θ denote the fixed point of the policy iteration operator so that P θ = Ψ( P θ , θ ). Let { w i : i = 1 , 2 , . . . , N } be a random sample of w = ( a, x ′ , x ) from the population, where x i is drawn from the stationary distribution implied by P θ and f θ f , a i is drawn conditional on x i from P θ ( ·| x i ), and x ′ i is drawn from f θ f ( ·| x i , a i ). Under Assumption 2, the log-likelihood function can be decomposed into conditional choice probability and transition probability terms as:

<!-- formula-not-decoded -->

Here, we consider the conditional likelihood, but it is also possible to incorporate the likelihood contribution from the initial observations using the implied stationary distribution (see Section 5). Since θ f can be estimated consistently without having to solve the Markov decision model, we focus on the estimation of α ≡ ( θ u , θ g ) given initial consistent estimates of θ f from the likelihood l N, 2 ( θ f ). Thus, Ψ( P, θ ) = Ψ( P, α, θ f ), and we use both Ψ( P, θ ) and Ψ( P, α, θ f ) henceforth.

The maximum likelihood estimator solves the following constrained maximization problem:

<!-- formula-not-decoded -->

Rust (1987) develops the celebrated Nested Fixed Point algorithm (NFXP) by formulating the

parameter restriction in terms of Bellman's equation. The NFXP repeatedly solves the fixed point problem at each parameter value to maximize the likelihood with respect to α . Let ˆ α denote the solution to the maximization problem (5), and let ˆ P denote the associated conditional choice probability estimate characterized by the fixed point: ˆ P = Ψ( ˆ P, ˆ α, ˆ θ f ).

## 3.1 Nested Pseudo-likelihood (NPL) Algorithm

Assuming an initial consistent estimator ˆ P 0 is available, the nested pseudo-likelihood (NPL) algorithm developed by AM is recursively defined as follows.

Step 1: Given ˆ P PL j -1 , update α by ˆ α PL j = arg max α N -1 ∑ N i =1 ln Ψ( ˆ P PL j -1 , α, ˆ θ f )( a i | x i ).

- Step 2: Update P using the obtained estimate ˆ α PL j by ˆ P PL j = Ψ( ˆ P PL j -1 , ˆ α PL j , ˆ θ f ) .

Iterate Steps 1-2 until j = k .

Let P 0 be the true set of conditional choice probabilities, and let f 0 be the true conditional transition probability of x . Let Θ α and Θ f be the set of possible values of α and θ f , and define Θ = Θ α × Θ f . Following AM, consider the following regularity conditions:

/negationslash

- Assumption 4. (a) Θ α and Θ f are compact. (b) Ψ( P, α, θ f ) is three times continuously Fdifferentiable. (c) Ψ( P, α, θ f )( a | x ) &gt; 0 for any ( a, x ) ∈ A × X and any { P, α, θ f } ∈ B P × Θ α × Θ f . (d) w i = { a i , x ′ i , x i } , for i = 1 , 2 , . . . , N , are independently and identically distributed, and dF ( x ) &gt; 0 for any x in the support of x i , where F ( x ) is the distribution function of x i . (e) There is a unique θ 0 f ∈ int(Θ f ) such that, for any ( a, x, x ′ ) ∈ A × X × X , f θ 0 f ( x ′ | x, a ) = f 0 ( x ′ | x, a ). (f) There is a unique α 0 ∈ int(Θ α ) such that, for any ( a, x ) ∈ A × X , P θ 0 ( a | x ) = P 0 ( a | x ). For any α = α 0 , Pr θ 0 ( { ( a, x ) : Ψ( P 0 , α, θ 0 f )( a | x ) = P 0 ( a | x ) } ) &gt; 0. (g) E θ 0 sup ( P,α,θ f ) || D s Ψ( P, α, θ f )( a | x ) || 2 &lt; ∞ for s = 1 , . . . , 4. (h) ˆ θ f -θ 0 f = O p ( N -1 / 2 ), ˆ P PL 0 -P 0 = o p (1), and the MLE, ˆ α , satisfies √ N (ˆ α -α 0 ) → d N (0 , Ω). (i) With respect to continuously distributed elements of x , P ( a | x ) has a uniformly bounded and Lipschitz continuous derivative up to order d , where d is the number of such elements.

/negationslash

Assumptions 4(a)-4(f) are similar to the regularity conditions 4(a)-(f) in AM. Although we do not pursue this here, it may be possible to state Assumption (b) in terms of the smoothness

conditions on the transition densities and the random utility distribution. The supremum in 4(g) may be taken in a neighborhood of ( P 0 , α 0 , θ 0 f ).

Following Robinson (1988), for matrix/mapping and (nonnegative) scalar sequences of random variables { X N , N ≥ 1 } and { Y N , N ≥ 1 } , respectively, we write X N = O p ( Y N )( o p ( Y N )) if || X N || ≤ CY N for some (all) C &gt; 0 with probability arbitrarily close to one for sufficiently large N .

Our first main result shows that the sequence of the estimators generated from the NPL algorithm converges to the MLE, ˆ α , at a superlinear, but less than quadratic, convergence rate.

Proposition 2 Suppose Assumptions 1-4 hold. Then, for k = 1 , 2 , . . .

<!-- formula-not-decoded -->

This proposition provides a theoretical explanation for the result of the AM's Monte Carlo experiment. Their experiment illustrates that the finite sample properties of the estimators generated from the NPL algorithm improve monotonically with k and that the estimators with k = 2 or 3 substantially outperform the estimator with k = 1.

Note that ˆ P PL 0 -P 0 = O p ( N -b ) with b &gt; 1 / 4 suffices for √ N (ˆ α PL k -α 0 ) → d N (0 , Ω) for all k ≥ 1 . This weakens assumption (g) of Proposition 4 of AM and also implies that the NPL algorithm is valid even if x i has an infinite support and a kernel-based estimator is used to estimate P 0 . The result suggests that the NPL algorithm may work even with relatively imprecise initial estimates of the conditional choice probabilities.

If ˆ P PL 0 -P 0 = O p ( N -b ) with b ∈ (1 / 4 , 1 / 2] , repeated substitution gives ˆ α PL k -ˆ α = O p ( N -( k -1) / 2 -2 b ) and ˆ P PL k -ˆ P = O p ( N -( k -1) / 2 -2 b ). In particular, if the support of x i is finite and we can obtain ˆ P PL 0 such that ˆ P PL 0 -P 0 = O p ( N -1 / 2 ), then the convergence rate becomes N -( k +1) / 2 .

## 3.2 Nested Modified Pseudo-likelihood (NMPL) Algorithm

We now introduce the nested modified pseudo-likelihood (NMPL) algorithm that achieves a faster rate of convergence than the NPL algorithm:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2: Update P using the obtained estimate ˆ α MPL j by ˆ P MPL j = Ψ( ˆ P MPL j -1 , ˆ α MPL j , ˆ θ f ) .

Iterate Steps 1-2 until j = k .

<!-- formula-not-decoded -->

/negationslash

/negationslash

The following proposition shows the estimator of α generated from the NMPL algorithm converges at a rate faster than quadratic while the estimator of P converges at a quadratic rate.

Proposition 3 Suppose Assumptions 1-5 hold. Then, for k = 1 , 2 , . . .

<!-- formula-not-decoded -->

If ˆ P MPL 0 -P 0 = O p ( N -b ) with b ∈ (0 , 1 / 2] , then the convergence rate is given by ˆ α MPL k -ˆ α = O p ( N -1 / 2 -b 2 k + N -3 b 2 k -1 ) and ˆ P MPL k -ˆ P = O p ( N -b 2 k ). In particular, if ˆ P MPL 0 -P 0 = O p ( N -1 / 4 ), then we have ˆ α MPL k -ˆ α = O p ( N -1 / 2 -2 k -2 ). Note that ˆ P MPL 0 -P 0 = O p ( N -b ) with b &gt; 1 / 6 suffices for √ N (ˆ α MPL k -α 0 ) → d N (0 , Ω) for all k ≥ 1. Therefore, the NMPL algorithm requires a weaker condition on the initial estimate of P 0 than the NPL algorithm.

It is important to emphasize that the NMPL algorithm requires more policy iterations than the NPL for computing each ˆ α j . In fact, the NPL algorithm requires only a single policy iteration for each pseudo-maximum likelihood estimation, whereas the NMPL algorithm requires evaluating the policy iteration mappings as many times as the number of trial values of the structural parameter vector θ . Furthermore, when θ is multiplicatively separable in the utility function, the pseudo-likelihood function of the NPL is globally concave, but that is not necessarily the case with the NMPL. As a result, the overall computational cost for achieving a given rate of convergence is likely to be higher with the NMPL than with the NPL.

Our proposed modified algorithm introduces the additional updating only in step 1. If we introduce the additional updating in step 2 but not in step 1, then the convergence rate would be the same as that of the NPL algorithm stated in Proposition 2. On the other hand, if we introduce the additional updating in both step 1 and step 2 with ˆ P MPL j = Ψ 2 ( ˆ P MPL j -1 , ˆ α MPL j , ˆ θ f ),

then the convergence rate of P will be improved and become faster than a quadratic rate as follows: ˆ P MPL k -ˆ P = O p ( || ˆ α MPL k -ˆ α || ) = O p ( N -1 / 2 || ˆ P MPL k -1 -ˆ P || 2 + || ˆ P MPL k -1 -ˆ P || 3 ).

## 4 One-step NMPL Algorithm

We propose one-step NMPL algorithms which update the parameter α using one Newton step without fully solving the optimization problem. This reduces the computational cost of the corresponding estimators especially when the dimension of α is high. Let L N ( P, α, θ f ) denote the objective function for the NMPL algorithm as L N ( P, α, θ f ) = N -1 ∑ N i =1 ln Ψ 2 ( P, α, θ f )( a i | x i ).

Suppose that an initial consistent estimator of α is available. 5 The one-step NMPL algorithm, with its estimator denoted by (˜ α MPL k , ˜ P MPL k ), is defined recursively as:

Step 1: Given ( ˜ P MPL j -1 , ˜ α MPL j -1 , ˆ θ f ), update α by

<!-- formula-not-decoded -->

where Q N,j -1 = Q N ( ˜ P MPL j -1 , ˜ α MPL j -1 , ˆ θ f ).

Step 2: Update P using ˜ α MPL j by ˜ P MPL j = Ψ( ˜ P MPL j -1 , ˜ α MPL j , ˆ θ f ).

Iterate Steps 1-2 until j = k .

The matrix Q N,j -1 determines whether the one-step NMPL algorithm uses the NR, default NR, line-search NR, or Gauss-Newton (GN) steps. The NR choice of Q N,j -1 is Q NR N,j -1 = ( ∂ 2 /∂α∂α ′ ) L N ( ˜ P MPL j -1 , ˜ α MPL j -1 , ˆ θ f ). The default NR choice of Q N,j -1 , denoted Q D N,j -1 equals Q NR N,j -1 if ˜ α MPL j defined in (6) satisfies L N ( ˜ P MPL j -1 , ˜ α MPL j , ˆ θ f ) ≥ L N ( ˜ P MPL j -1 , ˜ α MPL j -1 , ˆ θ f ), but equals some other matrix otherwise. Typically, (1 /ε ) I dim( α ) for some small ε &gt; 0 is used. The linesearch NR choice, Q LS N,j -1 , computes ˜ α MPL,λ j for λ ∈ (0 , 1] using (1 /λ ) Q NR N,j -1 and chooses the one that maximizes the objective function. The GN choice, denoted Q GN N,j -1 , uses a matrix that approximates the NR matrix Q NR N,j -1 . A popular choice is the outer-product-of-the-gradient (OPG)

5 The initial rootN consistent estimate, ˜ α MPL 0 , can be obtained from applying the original NPL algorithm with k = 1 or using Hotz and Miller estimator. Furthermore, when we apply the one-step NMPL algorithm to the bootstrap-based inference, we may use the estimate from the original sample as an initial rootN consistent estimate for the bootstrap sample.

estimator Q OPG N,j -1 = -N -1 ∑ N i =1 ( ∂/∂α ) ln Ψ 2 ( ˜ P MPL j -1 , ˜ α MPL j -1 , ˆ θ f )( a i | x i )( ∂/∂α ′ ) ln Ψ 2 ( ˜ P MPL j -1 , ˜ α MPL j -1 , ˆ θ f )( a i | x i ), because this does not require the calculation of the second derivative of the objective function.

The following proposition establishes that the one-step NMPL algorithm achieves a similar rate of convergence to the original NMPL algorithm: it achieves the quadratic rate of convergence when the NR, default NR, or line-search NR is used. This is because taking one NR step brings the resultant estimator sufficiently close to the estimator from the NMPL algorithm.

Proposition 4 Suppose the assumptions of Proposition 3 hold and the initial estimates (˜ α MPL 0 , ˜ P MPL 0 ) are consistent. Then, for k = 1 , 2 , . . . ,

<!-- formula-not-decoded -->

It is also possible to consider the one-step NPL algorithm in which (˜ α PL k , ˜ P PL k ) is defined analogously using N -1 ∑ N i =1 ln Ψ( P, α, θ )( a i | x i ) as L N ( P, α, θ ). In this case, under the assumptions of Proposition 2, the convergence rates starting from the initial consistent estimates (˜ α PL 0 , ˜ P PL 0 ) are given by ˜ α PL k -ˆ α = O p ( || ˜ α PL k -1 -ˆ α || 2 + N -1 / 2 || ˜ P PL k -1 -ˆ P || + || ˜ P PL k -1 -ˆ P || 2 ) [+ O p ( N -1 / 2 || ˆ α -˜ α PL k -1 || ) for OPG] and ˜ P PL k -ˆ P = O p ( || ˜ α PL k -ˆ α || ). Its proof is presented in a supplementary appendix available from the authors upon request. In case of the one-step NPL, however, computational saving relative to the original NPL algorithm may be limited, since the maximization of the pseudo-likelihood function of the NPL algorithm requires only one policy iteration.

## 5 Unobserved Heterogeneity

In the model of Section 2, individuals are homogeneous in terms of the parameter θ representing their preferences and transition probabilities. In many empirical applications, however, preferences and transition probabilities are likely to be different across individuals. An approach often used in practice is to treat such heterogeneity as unobserved by econometricians and to allow for a finite mixture of types (cf., Keane and Wolpin, 1997). This section discusses an extension of our estimation method to a finite mixture model.

Suppose there are M types of individuals, where type m is characterized by a type-specific

parameter θ m = ( α m ′ , θ m ′ f ) ′ and the probability of being type m in the population is π m ( m = 1 , . . . , M ). 6 It is assumed that the number of types, M , is known and π m ∈ (0 , 1). As often done in practice, we reparametrize the type probabilities as π m ( γ ) = exp( γ m ) / (1 + ∑ M -1 m =1 exp( γ i )) for m = 1 , . . . , M -1 and π M ( γ ) = 1 -∑ M -1 m =1 π m ( γ ), where γ = ( γ 1 , . . . , γ M -1 ) ′ .

Let ζ = ( γ ′ , θ 1 ′ , . . . , θ M ′ ) ′ be the parameter to be estimated, and let Θ ζ denote the set of possible values of ζ . Let {{ a it , x it , x i,t +1 } T t =1 } N i =1 be a panel data such that w i = { a it , x it , x i,t +1 } T t =1 is randomly drawn across i 's from the population. The initial state x i 1 is assumed to be randomly drawn from a type-specific stationary distribution implied by the conditional choice probability and the transition probability. We consider the asymptotics when T is fixed and N →∞ .

Conditional on being type m , the likelihood of observing w i is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where P θ m is the fixed point of Ψ( · , θ m ) while λ ( x ; P θ m , f θ f m ) is the stationary distribution of x for type m defined as the fixed point of the mapping defined by (8). Since solving (8) given ( P θ m , f θ f m ) is often less computationally intensive than computing P θ m , we assume the full solution of (8) is available given ( P θ m , f θ f m ).

The MLE of ζ is defined as

<!-- formula-not-decoded -->

Let P m be the conditional choice probability for type m . Stack P m 's as P = ( P 1 , . . . , P M ), and let P 0 denote its true value. Define Ψ ( P , ζ ) = (Ψ( P 1 , θ 1 ) , . . . , Ψ( P M , θ M )) and Ψ 2 ( P , ζ ) = (Ψ 2 ( P 1 , θ 1 ) , . . . , Ψ 2 ( P M , θ M )). The pseudo-likelihood function for the NPL algorithm is

<!-- formula-not-decoded -->

6 If the transition probabilities are common across types so that θ m f = θ f for m = 1 , . . . , M , then we may use the 2-stage procedure analogous to that of Section 3.

and L PL ( w i ; P m , θ m ) = λ ( x i 1 ; Ψ( P m , θ m ) , f θ f m ) ∏ T t =1 f θ f m ( x i,t +1 | x it , a it )Ψ( P m , θ m )( a it | x it ), where λ is given by the fixed point of the mapping defined by (8). The pseudo-likelihood function for the NMPL algorithm is defined by L MPL N ( P , ζ ) = N -1 ∑ N i =1 l MPL ( w i ; P , ζ ), where l MPL ( w i ; P , ζ ) = l PL ( w i ; Ψ ( P , ζ ) , ζ ), i.e., we replace P m in the pseudo-likelihood function for the NPL algorithm L PL N ( P , ζ ) with Ψ( P m , θ m ). Let L MPL ( w i ; P m , θ m ) = L PL ( w i ; Ψ( P m , θ m ) , θ m ).

Let { π 0 ,m } M m =1 be the true set of type probabilities, and let { P 0 ,m , f 0 ,m } M m =1 be the true sets of type-specific conditional choice probabilities and transition probabilities. Let P 0 ( w ) denote the true set of probabilities for w defined as P 0 ( w ) ≡ ∑ M m =1 π 0 ,m λ ( x 1 ; P 0 ,m , f 0 ,m ) × ∏ T t =1 f 0 ,m ( x t +1 | x t , a t ) P 0 ,m ( a t | x t ). Let ˆ P PL 0 and ˆ P MPL 0 be initial consistent estimators of P . Consider the following regularity conditions that correspond to Assumptions 4 and 5.

/negationslash

Assumption 4UH. (a) Θ ζ is compact. (b) λ ( x ; P, f ) is three times continuously F-differentiable. (c) λ ( x ; P, f θ f ) &gt; 0 for any x ∈ X and any { P, θ f } ∈ B P × Θ f . (d) w i = { ( a it , x it , x i,t +1 ) : t = 1 , . . . , T } for i = 1 , . . . , N , are independently and identically distributed, and dF ( x ) &gt; 0 for any x ∈ X , where F ( x ) is the distribution function of x i . (e) For any { P m , θ m f } ∈ B P × Θ f , there exists a unique solution to the fixed point problem of (8). (f) There is a unique ζ 0 ∈ int(Θ ζ ) such that, for any w = { ( a t , x t , x t +1 ) : t = 1 , . . . , T } , ∑ M m =1 π m ( γ 0 ) L ( w ; θ 0 ,m ) = P 0 ( w ). For any ζ = ζ 0 , Pr ζ 0 ( { w : ∑ M m =1 π m ( γ ) L s ( w ; P 0 ,m , θ m ) = P 0 ( w ) } ) &gt; 0 for s ∈ { PL,MPL } . (g) E ζ 0 sup ( P,f ) || D s λ ( x ; P, f ) || 2 &lt; ∞ for s = 0 , . . . , 4. (h) ˆ P PL 0 -P 0 = o p (1), ˆ P MPL 0 -P 0 = o p (1), and the MLE ˆ ζ satisfies √ N ( ˆ ζ -ζ 0 ) → d N (0 , Ω ζ ). (i) With respect to continuously distributed elements of x , λ ( x ; P, f ) has a uniformly bounded and Lipschitz continuous derivative up to order d , where d is the number of such elements.

/negationslash

Assumption 4UH(h) requires the availability of initial nonparametric consistent estimators for the type-specific conditional choice probabilities. Kasahara and Shimotsu (2007) derive sufficient conditions for nonparametric identification of a finite mixture model (7)-(8), showing that the type-specific choice probabilities can be nonparametrically identified when T ≥ 6 provided that the variation of x 's leads to sufficiently large variations in the choice probabilities across types. Kasahara and Shimotsu (2006) develop a series logit estimator of finite mixture models which provides an initial consistent estimate for the type-specific choice probabilities.

The following Lemma establishes the key property of the pseudo-likelihood functions of the NPL and NMPL algorithm in the context of a finite mixture model. Define P ζ = ( P θ 1 , . . . , P θ M ).

Lemma 1 Suppose Assumptions 1-3 hold and Ψ( · ) and λ ( · ; · , · ) are F-differentiable. Then D P l PL ( w i ; P ζ , ζ ) = D P l MPL ( w i ; P ζ , ζ ) = 0 . Suppose, in addition, Assumption 4(a)-(c), 4(e)(g) and 4UH hold. Then D P ζ L PL N ( P ˆ ζ , ˆ ζ ) = O p ( N -1 / 2 ) and D P ζ L MPL N ( P ˆ ζ , ˆ ζ ) = 0 .

Thus, at the fixed point, the parameter of interest ζ and the nuisance parameter P are asymptotically orthogonal for the NPL algorithm and are orthogonal in any sample size for the NMPL algorithm. Given this result, we may develop the NPL and NMPL algorithms for a finite mixture model which have similar convergence properties to those in Section 3.

Given the initial consistent estimators, ˆ P PL 0 and ˆ P MPL 0 , the NPL and NMPL algorithms are defined as follows. Let s ∈ { PL,MPL } .

Step 1: Given ˆ P s j -1 , ˆ ζ s j is computed by

<!-- formula-not-decoded -->

Step 2: For m = 1 , . . . , M , update ˆ P s,m j -1 using the obtained estimate ˆ θ s,m j as ˆ P s,m j = Ψ( ˆ P s,m j -1 , ˆ θ s,m j ).

Iterate Steps 1-2 until j = k .

The following proposition corresponds to Propositions 2 and 3 and establishes the convergence rates of the NPL and NMPL algorithms for a finite mixture model. Define ˆ P = P ˆ ζ , the MLE of P .

Proposition 5 Suppose Assumptions 1-3, 4(a)-(c), 4(e)-(g), 5, and 4UH hold. Then, for k = 1 , 2 , . . .

<!-- formula-not-decoded -->

Kasahara and Shimotsu (2006) show the convergence rate of the series logit estimator which can be used as initial estimators, ˆ P PL 0 and ˆ P MPL 0 . For instance, when x is scalar and the choice probabilities are four times continuously differentiable with respect to x , Lemma 1 in Kasahara and Shimotsu (2006) implies that the series estimator of type-specific choice probabilities

converges at the rate of N -1 / 5 . In this case, using the series estimator as an initial estimator, we may establish from Proposition 5 that √ N ( ˆ ζ PL k -ζ 0 ) → d N (0 , Ω ζ ) for all k ≥ 2 while √ N ( ˆ ζ MPL k -ζ 0 ) → d N (0 , Ω ζ ) for all k ≥ 1; hence, to obtain an estimator that is asymptotically equivalent to the MLE, the NPL algorithm requires at least two iterations while one iteration suffices for the NMPL algorithm.

The one-step NPL and NMPL algorithms are analogously defined to the NPL and NMPL algorithms except that they update the parameter ζ using one Newton step without fully solving the pseudo-maximization problem (10). Specifically, the one-step NMPL algorithm updates its estimate as ˜ ζ MPL j = ˜ ζ MPL j -1 -Q MPL N ( ˜ P MPL j -1 , ˜ ζ MPL j -1 ) -1 ( ∂/∂ζ ) L MPL N ( ˜ P MPL j -1 , ˜ ζ MPL j -1 ). Then, ˜ P MPL j -1 is updated as ˜ P s,m j = Ψ( ˜ P s,m j -1 , ˜ θ s,m j ) for m = 1 , . . . , M . This process is iterated for j = 1 , . . . , k . The NR choice of Q MPL N is Q MPL N ( P , ζ ) = ( ∂ 2 /∂ζ∂ζ ′ ) L MPL N ( P , ζ ) whereas the OPG estimator is Q MPL N ( P , ζ ) = -N -1 ∑ N i =1 ( ∂/∂ζ ) l MPL ( w i ; P , ζ )( ∂/∂ζ ′ ) l MPL ( w i ; P , ζ ). The one-step NPL algorithm is defined analogously.

The following proposition corresponds to Proposition 4 and shows that the one-step NPL/NMPL algorithm achieves a similar rate of convergence as the original NPL/NMPL algorithm for a finite mixture model. The proof is omitted because it follows the proof of Proposition 4.

Proposition 6 Suppose the assumptions of Proposition 5 hold and the initial estimates ( ˜ ζ PL 0 , ˜ P PL 0 ) and ( ˜ ζ MPL 0 , ˜ P MPL 0 ) are consistent. Then, for k = 1 , 2 , . . . ,

<!-- formula-not-decoded -->

The asymptotic covariance matrix of ˆ ζ is given by Σ( ζ 0 ) = D ( ζ 0 ) -1 V ( ζ 0 )( D ( ζ 0 ) -1 ) ′ , where D ( ζ ) = -E ( ∂ 2 /∂ζ∂ζ ′ ) l ( w ; ζ ) and V ( ζ ) = E ( ∂/∂ζ ) l ( w ; ζ )( ∂/∂ζ ′ ) l ( w ; ζ ). As in Section 3.3, we may estimate the asymptotic covariance matrix either using the averages of the derivatives of l ( w i ; ˆ ζ ) or the derivatives of the summands of the pseudo-likelihood function.

## 6 Parametric Bootstrap and Higher-order Improvements

In this section, building upon Andrews (2005), we analyze the higher-order improvements from applying parametric bootstrapping to the parametric discrete Markov decision models.

## 6.1 Assumptions

We first introduce technical conditions that are used in establishing the higher-order improvements from applying parametric bootstrapping. They are essentially the same as Assumptions 4.1-4.3 in Andrews (2005). Let c be a non-negative constant such that 2 c is an integer. Let g ( w i , θ ) = (( ∂/∂θ ′ ) ln P θ ( a | x ) , ( ∂/∂θ ′ f ) ln f θ f ( x ′ | x, a )) ′ , and let h ( w i , θ ) ∈ R L h denote the vector containing the unique components of g ( w i , θ ) and g ( w i , θ ) g ( w i , θ ) ′ and their partial derivatives with respect to θ through order d = max { 2 c +2 , 3 } . Let λ min ( A ) denote the smallest eigenvalue of the matrix A . Let d ( θ, B ) denote the distance between the point θ and the set B.

We assume the true parameter θ 0 lies in a subset Θ 0 of Θ and establish asymptotic refinements that hold uniformly for θ 0 ∈ Θ 0 . For some δ &gt; 0, let Θ 1 = { θ ∈ Θ : d ( θ, Θ 0 ) &lt; δ/ 2 } and Θ 2 = { θ ∈ Θ : d ( θ, Θ 0 ) &lt; δ } be slightly larger sets than Θ 0 . For the reason why these sets need to be considered, see Andrews (2005).

- Assumption 6. (a) Θ 1 is an open set. (b) Given any ε &gt; 0, there exists η &gt; 0 such that || θ -θ 0 || &gt; ε implies that E θ 0 ln P θ 0 ( a i | x i ) -E θ 0 ln P θ ( a i | x i ) &gt; η and E θ 0 ln f θ 0 f ( x ′ i | x i , a i ) -E θ 0 ln f θ f ( x ′ i | x i , a i ) &gt; η for all θ ∈ Θ and θ 0 ∈ Θ 1 . (c) sup θ 0 ∈ Θ 1 E θ 0 sup θ ∈ Θ || g ( w i , θ ) || q 0 &lt; ∞ , sup θ 0 ∈ Θ 1 E θ 0 sup θ ∈ Θ {| ln P θ ( a i | x i ) | q 0 + | ln f θ f ( x ′ i | x i , a i ) | q 0 } &lt; ∞ for all θ ∈ Θ for q 0 = max { 2 c +1 , 2 } .
- Assumption 7. (a) g ( w,θ ) is d = max { 2 c + 2 , 3 } times partially differentiable with respect to θ on Θ 2 for all w = ( a, x ′ , x ) ∈ A × X × X . (b) sup θ 0 ∈ Θ 1 E θ 0 || h ( w i , θ 0 ) || q 1 &lt; ∞ for some q 1 &gt; 2 c + 2. (c) inf θ 0 ∈ Θ 1 λ min ( V ( θ 0 )) &gt; 0, inf θ 0 ∈ Θ 1 λ min ( D ( θ 0 )) &gt; 0. (d) There is a function C h ( w i ) such that || h ( w i , θ ) -h ( w i , θ 0 ) || ≤ C h ( w i ) || θ -θ 0 || for all θ ∈ Θ 2 and θ 0 ∈ Θ 1 such that || θ -θ 0 || &lt; δ and sup θ 0 ∈ Θ 1 E θ 0 C q 1 h ( w i ) &lt; ∞ for some q 1 &gt; 2 c +2 .
- Assumption 8. (a) For all ε &gt; 0, there exists a positive δ such that for all t ∈ R L h with || t || &gt; ε , | E θ 0 exp( it ′ h ( w i , θ 0 )) | ≤ 1 -δ for all θ 0 ∈ Θ 1 . (b) Var θ 0 ( h ( w i , θ 0 )) has smallest eigenvalue bounded away from 0 over θ 0 ∈ Θ 1 .

The higher-order differentiability of ln P θ ( a | x ) and ln f θ f ( x ′ | x, a ) are satisfied if the density function of the unobserved state variable, /epsilon1 , and the utility function, u θ , are sufficiently smooth. Note that Assumption 4.1(b) of Andrews (2005) is satisfied by the definition of ˆ α and ˆ θ f . Assumption 4.1(c) of Andrews (2005) is satisfied with ρ ( θ, θ 0 ) = E θ 0 ln P θ ( a | x ) and E θ 0 ln f θ f ( x ′ | x, a ). Assumption 4.1(d) of Andrews (2005) is satisfied by Assumption 6(b). Because w i is iid, Assumption 4.3(a), (b), and (d) of Andrews (2005) are trivially satisfied, and his Assumption 4.3(c) reduces to the standard Cram´ er condition. Assumption 4.3(f) of Andrews (2005) follows from our Assumption 8(b) since w i is iid. Assumption 8(a), however, is not satisfied when all elements of the observed state variable have a finite support.

## 6.2 Pivotal Test Statistics

Suppose ˆ θ f is obtained by maximizing l N, 2 ( θ f ). Suppress ( a | x ) and ( x ′ | x, a ) from P θ ( a | x ) and f θ f ( x ′ | x, a ) . Expanding the first order condition for ˆ α and ˆ θ f gives the asymptotic covariance matrix of ˆ θ = (ˆ α ′ , ˆ θ ′ f ) ′ as

<!-- formula-not-decoded -->

where V ( θ ) = E [ ξξ ′ ] with ξ = [( ∂/∂α ′ ) ln P θ , ( ∂/∂θ ′ f ) ln f θ f ] ′ and

<!-- formula-not-decoded -->

We estimate V ( θ ) consistently by its sample analogue, while an outer-product-of-the-gradient estimator based on the information matrix equality is used to estimate D ( θ ) consistently. Note that the information matrix equality from the full MLE based on l N ( θ ) implies -E ( ∂ 2 /∂α∂θ ′ f ) ln P θ 0 = E ( ∂/∂α ) ln P θ 0 ( ∂/∂θ ′ f )(ln P θ 0 +ln f θ 0 f ). Letting V N ( θ ) and D N ( θ ) denote these estimators, then Σ N ( θ ) = D N ( θ ) -1 V N ( θ )( D N ( θ ) -1 ) ′ estimates Σ( θ ) consistently. Alternatively, a consistent estimator can be constructed from the pseudo-likelihood function: we simply need to replace P θ in the definition of D N ( θ ) and V N ( θ ) with Ψ( P, θ ) or Ψ 2 ( P, θ ).

Let θ r , θ 0 r , and ˆ θ r denote the r -th elements of θ , θ 0 , and ˆ θ respectively. Let (Σ N ) rr denote the ( r, r )-th element of Σ N = Σ N ( ˆ θ ). The t -statistic for testing the null hypothesis H 0 : θ r = θ 0 r is

<!-- formula-not-decoded -->

/negationslash

Let η ( θ ) be an R L η -valued function that is continuously differentiable at θ 0 . The Wald statistic for testing H 0 : η ( θ 0 ) = 0 versus H A : η ( θ 0 ) = 0 is

<!-- formula-not-decoded -->

Then T N ( θ 0 r ) → d N (0 , 1) and W N ( θ 0 ) → d χ 2 L η under the null hypotheses.

## 6.3 The MLE Parametric Bootstrap

We begin with analyzing bootstrapping the MLE to provide a benchmark for comparison with the one-step NPL and NMPL bootstraps. The parametric bootstrap sample { w ∗ i : i = 1 , . . . , n } is generated using the parametric density at the MLE, ˆ θ = (ˆ α ′ , ˆ θ ′ f ) ′ . The conditional distribution of the bootstrap sample given ˆ θ is the same as the distribution of the original sample except that the true parameter is ˆ θ rather than θ 0 = ( α 0 ′ , θ 0 ′ f ) ′ . 7

The bootstrap estimator θ ∗ = ( α ∗′ , θ ∗′ f ) ′ is defined exactly as the original estimator ˆ θ but using the bootstrap sample { w ∗ i : i = 1 , . . . , n } . Specifically,

<!-- formula-not-decoded -->

The bootstrap covariance matrix estimator, Σ ∗ N , is defined as Σ ∗ N ( θ ∗ ) where Σ ∗ N ( θ ) has the same definition as Σ N ( θ ) but with the bootstrap sample in place of the original sample. The bootstrap t and Wald statistics are defined as

<!-- formula-not-decoded -->

7 If x i is assumed to be exogenous, then x ∗ i = x i needs to be used. If x i is assumed to be drawn from its stationary distribution λ ( θ ) implied by P θ and f θ f , then x ∗ i is either equal to x i or drawn from λ ( ˆ θ ).

/negationslash where θ ∗ r denotes the r -th element of θ ∗ , and (Σ ∗ N ) rr denotes the ( r, r )-th element of Σ ∗ N . Here, we use the bootstrap Wald statistics to test H 0 : η ( θ 0 ) = 0 versus H A : η ( θ 0 ) = 0.

Let z ∗ | T | ,α , z ∗ T,α , and z ∗ W ,α denote the 1 -α quantiles of | T ∗ N ( ˆ θ r ) | , T ∗ N ( ˆ θ r ), and W ∗ N ( ˆ θ ) , respectively. The symmetric two-sided bootstrap CI for θ 0 r of confidence level 100(1 -α )% is

<!-- formula-not-decoded -->

The equal-tailed two-sided bootstrap CI for θ 0 r of confidence level 100(1 -α )% is

<!-- formula-not-decoded -->

/negationslash

The symmetric two-sided bootstrap t test of H 0 : θ r = θ 0 r versus H 1 : θ r = θ 0 r at significance level α rejects H 0 if | T N ( θ 0 r ) | &gt; z ∗ | T | ,α . The equal-tailed two-sided bootstrap t test at significance level α for the same hypotheses rejects H 0 if T N ( θ 0 r ) &lt; z ∗ T, 1 -α/ 2 or T N ( θ 0 r ) &gt; z ∗ T,α/ 2 . The bootstrap Wald test rejects H 0 if W N ( θ 0 ) &gt; z ∗ W ,α .

The following Lemma establishes the higher-order improvements of the bootstrap MLE.

Lemma 2 Suppose Assumptions 1-8 hold with c in Assumptions 6 and 7. Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The errors in coverage probability of standard delta method CIs are O ( N -1 ) and O ( N -1 / 2 ) for symmetric CIs and equal-tailed CIs, respectively. The errors in rejection probability of a standard Wald test are O ( N -1 ) . 8

## 6.4 One-step NPL and NMPL Parametric Bootstrap

Bootstrapping the MLE is computationally costly because one has to estimate the model repeatedly under different bootstrap samples, where each estimation requires the repeated full

8 Davidson and MacKinnon (1999b) and Kim (2005) analyze an alternative parametric bootstrap procedure that draws the bootstrap sample using the restricted MLE where the null is imposed. Their results indicate that the bootstrap equal-tailed t -test from the restricted parametric bootstrap have smaller errors in rejection probabilities than the unrestricted parametric bootstrap. In this paper, we mainly focus on CIs, but we conjecture that such a refinement from bootstrapping with the restricted MLE is also possible in our context.

solution of the Bellman equation. For this reason, we propose the one-step bootstrap NPL and NMPL algorithms, of which bootstrap estimators are defined as θ ∗ PL k = ( α ∗ PL k ′ , θ ∗ f ′ ) ′ and θ ∗ MPL k = ( α ∗ MPL k ′ , θ ∗ f ′ ) ′ , where θ ∗ f is defined in (12) and ( α ∗ PL k , P ∗ PL k , α ∗ MPL k , P ∗ MPL k ) are defined exactly as (˜ α PL k , ˜ P PL k , ˜ α MPL k , ˜ P MPL k ) but using the bootstrap sample { w ∗ i : i = 1 , . . . , n } .

We estimate θ by the MLE in the original sample and use the fixed point at the MLE, P ˆ θ , as the initial estimate of P for the one-step estimation with the bootstrap samples. Using the NFXP and P ˆ θ does not increase the computational burden significantly, since we are required to estimate θ and compute P ˆ θ only once in the original sample. 9

We use the derivatives of the pseudo-likelihood function for the NPL or NMPL algorithm to construct the covariance matrix estimate. Define D PL N ( P, θ ) and D MPL N ( P, θ ) by replacing P θ in the definition of D N ( θ ) with Ψ( P, θ ) and Ψ 2 ( P, θ ), respectively. Define V PL N ( P, θ ) and V MPL N ( P, θ ) analogously. With the one-step NPL, we use the bootstrap covariance matrix estimator as Σ ∗ N ( P, θ ) = D ∗ PL N ( P, θ ) -1 V ∗ PL N ( P, θ )( D ∗ PL N ( P, θ ) -1 ) ′ , where D ∗ PL N and V ∗ PL N are constructed with the bootstrapped sample. With the one-step NMPL, we use Σ ∗ N ( P, θ ) = D ∗ MPL N ( P, θ ) -1 V ∗ MPL N ( P, θ )( D ∗ MPL N ( P, θ ) -1 ) ′ , with analogous definitions for D ∗ MPL N and V ∗ MPL N .

Evaluating the derivatives of the pseudo-likelihood functions involves a limited number of policy iterations and, under the assumption of extreme-value distributed unobserved state variables, the analytical expression for the first derivatives are available. The computational saving from using the pseudo-covariance matrix estimate can be substantial, since we need to compute the covariance matrix estimates as many times as the number of bootstraps.

The one-step bootstrap t - and Wald statistics, T ∗ N,k ( ˆ θ r ) and W ∗ N,k ( ˆ θ ), are defined as in (13), but with ( θ ∗ , Σ ∗ N ) replaced by ( θ ∗ PL k , Σ ∗ N ( P ∗ PL k , θ ∗ PL k )) or ( θ ∗ MPL k , Σ ∗ N ( P ∗ MPL k , θ ∗ MPL k )). The one-step bootstrap CIs, denoted CI SY M,k , CI ET,k , are defined analogously to (14) and (15) but using the 1 -α quantiles of | T ∗ N,k ( ˆ θ r ) | and T ∗ N,k ( ˆ θ r ) instead of | T ∗ N ( ˆ θ r ) | and T ∗ N ( ˆ θ r ).

Define

µ N,k = N -2 k -1 ln 2 k ( N ) for the one-step NMPL with NR, default NR, and line-search NR, µ N,k = N -( k +1) / 2 ln k +1 ( N ) for the one-step NPL and the one-step NMPL with OPG.

9 Alternatively, we may estimate θ by the NPL or NMPL algorithm (with sufficiently many iterations) instead of the NFXP in the original sample and use ˆ P PL k or ˆ P MPL k as the initial estimate for the bootstrap estimation.

Lemma 3 establishes the higher-order equivalence between the bootstrap MLE and the bootstrap estimator generated from the one-step NPL or NMPL algorithm. Lemma 4 shows, under suitable conditions on c and k , the difference between the bootstrap test statistics constructed using the one-step NPL or NMPL algorithm and those of the MLE is o ( N -c ) .

Lemma 3 Suppose Assumptions 1-8 hold for some c &gt; 0 with 2 c an integer and sup θ ∈ Θ 1 || ( ∂/∂θ ) P θ ( a | x ) || , sup ( P,θ ) ∈ ( B p , Θ 1 ) || D s Ψ( P, θ )( a | x ) || &lt; ∞ uniformly in ( a, x ) for s = 1 , 2 , 3 . Then, for all ε &gt; 0 and s = { PL,MPL } ,

<!-- formula-not-decoded -->

Lemma 4 Suppose the assumptions of Lemma 3 hold and µ N,k = o ( N -( c +1 / 2) ) . Then, for all ε &gt; 0 ,

<!-- formula-not-decoded -->

for Ξ k ( z ) = Pr ∗ ˆ θ ( N 1 / 2 ( θ ∗ s k -ˆ θ ) ≤ z ) -Pr ∗ ˆ θ ( N 1 / 2 ( θ ∗ -ˆ θ ) ≤ z ) with s = { PL,MPL } , Pr ∗ ˆ θ ( T ∗ N,k ( ˆ θ r ) ≤ z ) -Pr ∗ ˆ θ ( T ∗ N ( ˆ θ r ) ≤ z ) , or Pr ∗ ˆ θ ( W ∗ N,k ( ˆ θ ) ≤ z ) -Pr ∗ ˆ θ ( W ∗ N ( ˆ θ ) ≤ z ) .

The following Lemma shows that the errors in coverage probability of the one-step NPL and NMPL bootstrap CIs are the same as those of the MLE bootstrap CIs. Therefore, the one-step bootstrap estimators achieve the same level of higher-order refinement as the bootstrap MLE.

Lemma 5 Suppose the assumptions of Lemma 3 hold.

<!-- formula-not-decoded -->

The condition µ N,k = o ( N -5 / 2 ) requires k ≥ 3 for the one-step NMPL algorithm with the NR, default NR, and line-search NR, and requires k ≥ 5 for the one-step NPL and the one-step NMPL algorithms with the OPG.

When θ is multiplicatively separable in the utility function, the one-step NPL algorithm has an important advantage over the one-step NMPL algorithm: the pseudo likelihood function of the NPL is globally concave. In such a case, one should use the one-step NPL rather than the one-step NMPL algorithm.

The estimators from the NPL and NMPL algorithms yield the same level of higher-order refinement as stated in Lemma 5 except that, reflecting the difference in their convergence rates, the definition of µ N,k for the estimator from the NMPL algorithm is different from that for the estimator from the one-step NMPL algorithm. Specifically, we have µ N,k = N -2 k -1 -1 / 2 ln 2 k +1 ( N ) for the NMPL algorithm with NR, default NR, and line search NR. We omit the proof because it is very similar to the proof of Lemmas 3-5.

## 6.5 Bootstrapping Models with Unobserved Heterogeneity

Applying our bootstrap-based inference method to a finite mixture model is straightforward. We estimate ζ by the NFXP as (9) in the original sample and use ˆ ζ and P ˆ θ m 's as the initial estimates for the bootstrap samples. The bootstrap estimators from the one-step bootstrap NPL and NMPLalgorithms, ( P ∗ PL k , ζ ∗ PL k , P ∗ MPL k , ζ ∗ MPL k ), are defined exactly as ( ˜ P PL k , ˜ ζ PL k , ˜ P MPL k , ˜ ζ MPL k ) but computing from the bootstrap sample. The bootstrap covariance matrix estimator, Σ ∗ PL N ( P ∗ PL k , ζ ∗ PL k ) or Σ ∗ MPL N ( P ∗ MPL k , ζ ∗ MPL k ), is defined analogously to the covariance matrix estimator, Σ N ( ˆ ζ ), except that we use the bootstrap sample and the corresponding pseudo-likelihood function. The one-step bootstrap t - and Wald statistics, T ∗ N,k ( ˆ ζ r ) and W ∗ N,k ( ˆ ζ ), are then defined as in (13), but with ( θ ∗ , Σ ∗ N ) replaced by ( ζ ∗ PL k , Σ ∗ PL N ( P ∗ PL k , ζ ∗ PL k )) or ( ζ ∗ MPL k , Σ ∗ MPL N ( P ∗ MPL k , ζ ∗ MPL k )). The one-step bootstrap CIs are defined similarly to (14) and (15).

Before presenting the final lemma, we define some notation. Let h ζ ( w i , ζ ) ∈ R L hζ denote the vector containing the unique components of ( ∂/∂ζ ) l ( w ; ζ ) and ( ∂/∂ζ ) l ( w ; ζ )( ∂/∂ζ ′ ) l ( w ; ζ ) and their partial derivatives with respect to ζ through order d = max { 2 c +2 , 3 } . We assume the true parameter ζ 0 lies in a subset Θ ζ, 0 of Θ ζ . For some δ &gt; 0, let Θ ζ, 1 = { ζ ∈ Θ ζ : d ( θ, Θ ζ, 0 ) &lt; δ/ 2 } and Θ ζ, 2 = { ζ ∈ Θ ζ : d ( θ, Θ ζ, 0 ) &lt; δ } . The following lemma establishes the higher-order improvements of the one-step bootstrap NPL and NMPL algorithms for a finite mixture model. The proof follows the proof of Lemmas 2-5 and is therefore omitted.

Lemma 6 Suppose Assumptions 1-3, 4(a)-(c), 4(e)-(g), 5, and 4UH hold. Suppose Assump-

tions 6-8 hold with θ, Θ , Θ 1 , Θ 2 , P θ ( a | x ) , h ( w i , θ ) replaced by ζ, Θ ζ , Θ ζ, 1 , Θ ζ, 2 , l ( w ; ζ ) , h ζ ( w i , ζ ) , respectively, for some c &gt; 0 with 2 c an integer. Suppose sup θ ∈ Θ || ( ∂/∂θ ) P θ ( a | x ) || , sup ( P,θ ) || D Ψ( P, θ )( a | x ) || , sup ( P,θ ) || D 2 Ψ( P, θ )( a | x ) || &lt; ∞ with probability one. Then the errors in coverage probability of CI SY M,k ( ˜ ζ r ) and CI ET,k ( ˜ ζ r ) and the errors in rejection probability of the one-step bootstrap Wald test are given by Lemma 5(a)-(c), respectively.

## 6.6 Nonparametric Bootstrap

In practice, applied researchers often use nonparametric bootstrap methods to obtain a consistent estimate of the asymptotic variance-covariance matrix (the so-called percentile method).

It is straightforward to show that an analogue of Propositions 2-5 holds for nonparametrically bootstrapped estimators by using the law of large numbers and information matrix equality for nonparametrically bootstrapped data. For example, an analogue of Proposition 2 holds, in which the O p () terms are replaced with the O p ∗ () terms, and the stated result holds with probability approaching one. Showing the first-order consistency of nonparametric bootstrap of the MLE is also straightforward.

Therefore, we may obtain valid (first-order consistent) standard error estimates and confidence intervals from nonparametrically bootstrapping the NPL, MNPL estimators and their one-step version and using a percentile method. Andrews (2002b) shows the higher-order improvements from applying block (thus nonparametric) bootstrap to extremum estimators with stationary and ergodic data. We conjecture that it is possible to use the techniques in Andrews (2002b) to show the higher-order improvements of the nonparametric bootstrap.

## 7 Monte Carlo Experiments

## 7.1 Experimental Design

The model we consider is a version of the machine replacement models of Rust (1987) and Cooper, Haltiwanger, and Power (1999). There are two observable state variables in the model: machine age s t ∈ N and productivity shock ω t ∈ R . We denote the vector of observed state variables by x t = ( s t , ω t ) ′ and let the variable a t ∈ { 0 , 1 } represent the machine replacement

decision. The profit function is given by u ( x t , a t ) + /epsilon1 ( a t ), where

<!-- formula-not-decoded -->

with rc ( a t ) = θ 0 a t , y ( s t , ω t , a t ) = exp( θ 1 s t (1 -a t ) + ω t ), and mc ( s t , a t ) = θ 2 s t (1 -a t ). Here, y ( s t , ω t , a t ) is a revenue function; c ( s t ) is a machine maintenance cost; rc ( s t ) is a replacement cost; and /epsilon1 ( a t ) is an unobserved state variable which follows an extreme value distribution independently across alternatives. The transition function of s t is given by s t = a t -1 + (1 -a t -1 )( s t -1 +1) and productivity shock ω t follows an AR(1) process ω t = ρω t -1 + η t with η t ∼ N (0 , σ 2 η ). The model requires estimation of the three structural parameters whose true value is given by θ ≡ ( θ 0 , θ 1 , θ 2 ) ′ = (2 . 0 , -0 . 2 , 0 . 1) ′ . We assume that the other parameters in the model, ( β, ρ, σ η ), are known and fixed at ( β, ρ, σ η ) = (0 . 96 , 0 . 8 , 0 . 2).

We generate a cross-sectional data set of sample size N from a parametric model by first randomly drawing the initial states { ( s i , ω i ) : i = 1 , . . . , N } from the stationary distribution of ( s, ω ) under θ and then simulating a i 's using the conditional choice probabilities P θ ( a | s i , ω i ). The data set consists of { ( s i , ω i , a i ) : i = 1 , . . . , N } .

## 7.2 Estimation based on NPL and NMPL algorithms

We first examine the finite sample performance of our proposed estimators. We simulate 1000 samples, each of which consists of N = 500 observations. For the NPL and NMPL algorithms, the initial estimate of conditional choice probabilities, ˆ P 0 , is obtained by the sieve logit estimator with cubic polynomials in the conditioning variables, ( s, ω ). The one-step NPL and NMPL algorithms require, in addition, the initial estimate of θ . We use the estimator with the NPL algorithm at the initial iteration ( k = 1) as the initial estimate of θ , and then we examine the performance of the one-step algorithms starting from the second iteration ( k = 2). In this experiment, we use default NR-step with /epsilon1 = 0 . 001 for the one-step algorithms.

Table 1 reports the average absolute difference between our proposed estimators and the MLE, divided by the MLE to express it as a percentage. In the table, '1-NPL' and '1-NMPL' represent the estimators from the one-step NPL and one-step NMPL algorithms, respectively. Consistent with the results in Sections 3 and 4, all four estimators (NPL, NMPL, 1-NPL, and 1-NMPL) monotonically approach to the MLE as k increases. Furthermore, the NMPL-based

estimator approaches to the MLE faster than the NPL-based estimator. Similarly, the one-step NMPL algorithm approaches to the MLE with fewer iterations than the one-step NPL. The convergence of the one-step NPL and NMPL algorithms to the MLE is not as fast as that of the NPL and NMPL algorithms. However, after five iterations ( k = 5), the distance between the one-step estimators and the MLE becomes as small as 0.2-2.5%. We also note that the performance of the initial consistent estimator ˆ P 0 affects the performance of the NPL/NMPLbased estimators; when we use the true conditional choice probabilities as the initial estimator, the NPL/NMPL estimators converge to the MLE much faster. 10

Table 2 reports the bias and variance. The bias and variance of the estimators from the NPL and NMPL algorithm and their one-step versions approach to those of the MLE with k . When k = 1, the bias and variance of the estimates from the NPL and NMPL differ from those of the MLE, and both have a larger mean squared error than the MLE. The last six rows report the integrated mean squared error (IMSE) of the logarithm of conditional choice probabilities. The IMSE of the initial sieve logit estimator, reported in the row with k = 0, is much larger than the IMSE of the MLE, implying that the NPL and NMPL algorithms start from an imprecise estimate of P 0 . Nonetheless, the three estimators have almost the same IMSE after two iterations ( k = 2).

## 7.3 Bootstrapping

We conduct parametric bootstraps with 1000 simulated samples consisting of N = 1000 observations. For each simulated sample, we obtain the MLE using the NFXP algorithm and draw B=599 bootstrap samples from the parametric model evaluated at the MLE. 11 Then we estimate parameters for each bootstrap sample using the NFXP, NPL, NMPL, one-step NPL, and one-step NMPL algorithms starting from the MLE in the original sample. We use a simple NR-step for the one-step NPL and NMPL algorithms. The covariance matrices of the MLE are constructed by an outer-product-of-the-gradient (OPG) estimator using the derivatives of the likelihood function while those of the estimators for the NPL, NMPL, one-step NPL, and

10 Simulation results are available upon request.

11 We draw the bootstrap samples of { ( s ∗ i , ω ∗ i ) : i = 1 , . . . , N } from the stationary distribution under the MLE ˆ θ . We examine the alternative case in which ( s ∗ i , ω ∗ i ) is set to the original observation ( s i , ω i ) and find that the results are similar. We also experiment with B = 999 in some cases and find that the results do not change substantially.

one-step NMPL algorithms are constructed by the OPG estimator using the derivatives of their pseudo-likelihood functions.

We first compare the performance of the bootstrap Wald test and the asymptotic Wald test. The null hypothesis we test is H 0 : ( θ 1 , θ 2 ) = ( -0 . 2 , 0 . 1). Table 3 reports the rejection frequencies of the asymptotic Wald test for the MLE and the bootstrap Wald test for the MLE and the estimators from the NPL, NMPL, one-step NPL, and one-step NMPL algorithms. As shown in the first row, The asymptotic Wald test substantially overrejects the null hypothesis at all the levels. The bootstrap Wald tests using MLE slightly underreject at .10 and .05 levels but its overall performance is substantially better than that of the asymptotic Wald test. We also conduct the bootstrap Wald test based on the restricted MLE where the null is imposed. Its performance is reported in the row 'MLE-NULL' and is similar to the one based on the unrestricted MLE. The results from the bootstrap Wald tests using the NPL and NMPL algorithms with one iteration (i.e., k = 1) are similar to those using MLE and are better than that of asymptotic Wald test at all three levels. Furthermore, the bootstrap Wald tests using the one-step NPL and one-step NMPL perform well; 1-NPL and 1-NMPL with five iterations (i.e., k = 5) perform better than the asymptotic Wald test at all three levels.

Next, we compare the performance of the asymptotic CIs and the (symmetric) bootstrap CIs for the parameters θ 1 , θ 2 , and θ 3 . Table 4 reports the coverage performance of asymptotic and bootstrap 95% and 90% CIs. For the 95% CIs, the asymptotic CIs either undercover or overcover, with substantial undercovering for θ 1 . This may be due to the difference in the degree of nonlinearity. The parameter θ 1 enters into the profit function through exponential function while θ 0 and θ 1 are linearly related to the profit function; consequently, the degree of nonlinearly in θ 1 is larger than those in θ 0 and θ 2 .

The bootstrap CIs from various estimators have a similar or better coverage probabilities than the asymptotic CIs, in particular showing good coverage probabilities for θ 1 . For the 90% CIs, the miscoverage of the asymptotic CIs is not as severe as that for θ 1 , and the coverage performance of the the bootstrap CIs is similar to, or slightly worse than, that of the asymptotic CIs.

We acknowledge that our experiment has a limited scope. In most dynamic models the likelihood surface is complex. While this means that the asymptotic CIs are likely to be unreliable and, hence, the bootstrap is particularly useful in this context, it also means that a simple

application of the one-step NPL/NMPL may not work well. In practice, one should check the robustness of the results by using a line-search NR step instead of a simple NR step.

## 8 Appendix A: proofs

For an n -linear operator M ( x 1 , . . . , x n ) such as an n -th F-derivative, the operator norm of M is defined as || M || = sup || x 1 || = ··· = || x n || =1 || M ( x 1 , . . . , x n ) || . To simplify the notation, let ψ α ( P, α, θ f ) = N -1 ∑ N i =1 ( ∂/∂α ′ ) ln Ψ( P, α, θ f )( a i | x i ) and ψ 2 α ( P, α, θ f ) = N -1 ∑ N i =1 ( ∂/∂α ′ ) ln Ψ 2 ( P, α, θ f )( a i | x i ).

## 8.1 Proof of Proposition 1

Let ¯ P be an arbitrary set of conditional choice probabilities, and let h = h ( a | x ) be a mapping such that ¯ P + h ∈ B P . From the relation ϕ ( ¯ P )( x ) = u ¯ P ( x ) + βE ¯ P ϕ ( ¯ P )( x ), we obtain

<!-- formula-not-decoded -->

Recall u ¯ P ( x ) = ∑ a ∈ A ¯ P ( a | x ) u ( x, a ) + ∑ a ∈ A ¯ P ( a | x ) e x ( a, ¯ P x ), and note that ∑ a ∈ A h ( a | x ) = 0 because ¯ P, ¯ P + h ∈ B P . Furthermore, Lemmas 1 and 2 of AM hold uniformly in x ∈ X by Assumptions 1 and 2. Consequently, applying Lemma 2 of AM to u ¯ P + h ( x ) -u ¯ P ( x ) gives u ¯ P + h ( x ) -u ¯ P ( x ) = ∑ a ∈ A h ( a | x ) u ( x, a ) -Q -1 x ( ¯ P x ) ′ ¯ h x + o ( || h || ), where ¯ h x = ( h (2 | x ) , . . . , h ( J | x )) ′ , ¯ P x = ( ¯ P x (2) , . . . , ¯ P x ( J )) ′ , and o ( || h || ) term is uniform in x ∈ X.

Let P be the fixed point of Ψ, so that ϕ ( P )( x ) = V ( x ). Then

<!-- formula-not-decoded -->

Because ˜ v x = Q -1 x ( P x ) when P is the fixed point of Ψ, it follows that ϕ ( P + h ) -ϕ ( P ) = o ( || h || ) for any h and hence Dϕ ( P ) = 0. Since Ψ = Λ ◦ ϕ , application of the chain rule in B-spaces gives D Ψ( P ) = D Λ( ϕ ( P )) Dϕ ( P ) = 0. /square

## 8.2 Proof of Proposition 2

We use induction. First, assume ˆ P PL k -1 -P 0 = o p (1). Then ˆ α PL k is consistent, because the consistency proof in the proof of Proposition 4 of AM does not depend on the finiteness of X . The first order condition for ˆ α PL k implies that ψ α ( ˆ P PL k -1 , ˆ α PL k , ˆ θ f ) = 0. On the other hand, from the first order condition for the MLE and DP θ = D θ Ψ( P θ , θ ) (cf. equation (Ap.2) of AM p. 1540), it follows that ψ α ( ˆ P, ˆ α, ˆ θ f ) = 0. Applying the generalized Taylor's theorem [cf., pp.148-149 of Zeidler (1986)] to ψ α ( ˆ P PL k -1 , ˆ α PL k , ˆ θ f ) -ψ α ( ˆ P, ˆ α, ˆ θ f ) = 0 gives

<!-- formula-not-decoded -->

where P τ = τ ˆ P PL k -1 +(1 -τ ) ˆ P and α τ = τ ˆ α PL k +(1 -τ )ˆ α . Note that ˆ P -P 0 = P ˆ θ -P θ 0 = O p ( N -1 / 2 ) because ˆ θ -θ 0 = O p ( N -1 / 2 ) and ∂P θ /∂θ = ∂ Ψ( P θ , θ ) /∂θ = O p (1) from Lemma 7(a). For the first term on the left of (16), ∫ 1 0 ( ∂/∂α ) ψ α ( P τ , α τ , ˆ θ f ) dτ → p E ( ∂ 2 /∂α∂α ′ ) ln Ψ( P 0 , θ 0 ) follows from Lemma 7(d) and the consistency of P τ , ˆ θ f , and α τ . For the second term on the left of (16), we obtain D P ψ α ( P τ , α τ , ˆ θ f ) = O p ( N -1 / 2 ) + O p ( || ˆ P PL k -1 -ˆ P || ) + O p ( || ˆ α PL k -ˆ α || ) uniformly in τ by expanding D P ψ α ( P τ , α τ , ˆ θ f ) around ( ˆ P, ˆ α, ˆ θ f ) and using || P τ -ˆ P || ≤ || ˆ P PL k -1 -ˆ P || , || α τ -ˆ α || ≤ || ˆ α PL k -ˆ α || , Lemma 7(b)(c), and rootN consistency of (ˆ α, ˆ θ f , ˆ P ). Therefore, rearranging the terms in (16) gives [ E ( ∂ 2 /∂α∂α ′ ) ln Ψ( P 0 , θ 0 ) + o p (1) ] (ˆ α PL k -ˆ α ) = O p ( N -1 / 2 || ˆ P PL k -1 -ˆ P || ) + O p ( || ˆ P PL k -1 -ˆ P || 2 ). Then, ˆ α PL k -ˆ α = O p ( N -1 / 2 || ˆ P PL k -1 -ˆ P || + || ˆ P PL k -1 -ˆ P || 2 ) follows because E ( ∂ 2 /∂α∂α ′ ) ln Ψ( P 0 , θ 0 ) is a nonsingular negative definite matrix (see AM p.1541).

For the convergence rate of ˆ P PL k , we obtain ˆ P PL k = Ψ( ˆ P PL k -1 , ˆ α PL k , ˆ θ f ) = ˆ P + O p ( || ˆ α PL k -ˆ α || )+ O p ( || ˆ P PL k -1 -ˆ P || 2 ) by expanding Ψ( ˆ P PL k -1 , ˆ α PL k , ˆ θ f ) around ( ˆ P, ˆ α, ˆ θ f ), applying Ψ( ˆ P, ˆ α, ˆ θ f ) = ˆ P and D P Ψ( ˆ P, ˆ α, ˆ θ f ) = 0, and using Lemma 7(a). The required result for all k follows from induction because ˆ P PL 0 -P 0 = o p (1) by Assumption 4(g). /square

## 8.3 Proof of Proposition 3

We use induction. Assume ˆ P MPL k -1 -P 0 = o p (1). The consistency of ˆ α MPL k follows from an argument similar to the proof of consistency of ˆ α PL k by AM. From the first order conditions for the MLE and the estimator generated from the NMPL algorithm and Lemma 8(a), we have ψ 2 α ( ˆ P MPL k -1 , ˆ α MPL k , ˆ θ f ) = ψ 2 α ( ˆ P, ˆ α, ˆ θ f ) = 0. Applying the generalized Taylor's theorem to these

first order conditions gives

<!-- formula-not-decoded -->

where P τ = τ ˆ P MPL k -1 +(1 -τ ) ˆ P and α τ = τ ˆ α MPL k +(1 -τ )ˆ α . For the first term on the left of (17), ∫ 1 0 ( ∂/∂α ) ψ 2 α ( P τ , α τ , ˆ θ f ) dτ → p E ( ∂ 2 /∂α∂α ′ ) ln Ψ 2 ( P 0 , α 0 , θ 0 f ) = E ( ∂ 2 /∂α∂α ′ ) ln P θ 0 from Lemma 7(d) and the consistency of P τ , α τ , and ˆ θ f . For the second term on the left of (17), recall D P ψ 2 α ( ˆ P, ˆ α, ˆ θ f ) = 0 from Lemma 8(a) because ˆ P is the fixed point of Ψ( · , ˆ α, ˆ θ f ). Thus, applying the generalized Taylor's theorem to D P ψ 2 α ( P τ , α τ , ˆ θ f ) -D P ψ 2 α ( ˆ P, ˆ α, ˆ θ f ) yields

<!-- formula-not-decoded -->

where P b = bP τ +(1 -b ) ˆ P and α b = bα τ +(1 -b )ˆ α . For the right hand side of (18), we obtain D PP ψ 2 α ( P b , α b , ˆ θ f ) , D αP ψ 2 α ( P b , α b , ˆ θ f ) = O p ( N -1 / 2 + || α τ -ˆ α || + || P τ -ˆ P || ) uniformly in b by expanding them around ( P 0 , α 0 , θ 0 f ), applying the triangle inequality to || P b -P 0 || and || α b -α 0 || , and using Lemmas 7(b) and 8(c) and the rootN consistency of (ˆ α, ˆ P, ˆ θ f ). Substituting these bounds into (18) gives, uniformly in τ , D P ψ 2 α ( P τ , α τ , ˆ θ f ) = O p ( N -1 / 2 || ˆ P MPL k -1 -ˆ P || + || ˆ P MPL k -1 -ˆ P || 2 )+ o p ( || ˆ α MPL k -ˆ α || ). Consequently, rearranging the terms in (17) gives [ E ( ∂ 2 /∂α∂α ′ ) ln P θ 0 + o p (1)](ˆ α MPL k -ˆ α ) = O p ( N -1 / 2 || ˆ P MPL k -1 -ˆ P || 2 + || ˆ P MPL k -1 -ˆ P || 3 ), and the stated bound on ˆ α MPL k -ˆ α follows because E ( ∂ 2 /∂α∂α ′ ) ln P θ 0 is a nonsingular negative definite matrix.

For ˆ P MPL k , we have ˆ P MPL k = Ψ( ˆ P MPL k -1 , ˆ α MPL k , ˆ θ f ) = ˆ P + O p ( || ˆ α MPL k -ˆ α || ) + O p ( || ˆ P MPL k -1 -ˆ P || 2 ) from the same argument as the proof of Proposition 2. The required result for all k follows from induction because ˆ P MPL 0 -P 0 = o p (1) by Assumption 5(c). /square

## 8.4 Proof of Proposition 4

We prove the result for only the NR and OPG. The proof for the default NR and line-search NR is essentially the same except for showing Pr( Q D N = Q NR N ) → 0 and Pr( Q LS N = Q NR N ) → 0; see the proof of Lemma 7.1 of Andrews (2005) (A05 hereafter). We suppress the superscript MPL from ˜ α MPL j and ˜ P MPL j , and we suppress ˆ θ f from ψ 2 α ( P, α, ˆ θ f ) and Q N ( P, α, ˆ θ f ).

/negationslash

/negationslash

Since the MLE satisfies the first order condition ψ 2 α ( ˆ P, ˆ α ) = 0, applying the generalized

Taylor's theorem to ψ 2 α ( ˆ P, ˆ α ) -ψ 2 α ( ˜ P j -1 , ˜ α j -1 ) gives

<!-- formula-not-decoded -->

where the first two terms on the right of (19) cancel out, and || R N,j || ≤ 2 sup ( P,α ) ( || D PP ψ 2 α ( P, α ) || + || D αP ψ 2 α ( P, α ) || )( || ˆ P -˜ P j -1 || 2 + || ˆ α -˜ α j -1 || 2 )+sup ( P,α ) ( || D αα ψ 2 α ( P, α ) || )( || ˆ α -˜ α j -1 || 2 ), where the supremum is taken for all the pairs of ( P, α ) that lie between ( ˆ P, ˆ α ) and ( ˜ P j -1 , ˜ α j -1 ).

For the fourth term on the right of (19), the term inside the bracket is 0 in the NR and O p ( || ˆ P -˜ P j -1 || + || ˆ α -˜ α j -1 || + N -1 / 2 ) in the OPG from Lemma 7(d)(e) and the information matrix equality. For the fifth term on the right of (19), we obtain D P ψ 2 α ( ˜ P j -1 , ˜ α j -1 ) = O p ( N -1 / 2 || ˜ α j -1 -ˆ α || + N -1 / 2 || ˜ P j -1 -ˆ P || + || ˜ α j -1 -ˆ α || 2 + || ˜ P j -1 -ˆ P || 2 ) by expanding D P ψ 2 α ( ˜ P j -1 , ˜ α j -1 ) around ( ˆ P, ˆ α ) and applying D P ψ 2 α ( ˆ P, ˆ α ) = 0 and D PP ψ 2 α ( ˆ P, ˆ α ), D αP ψ 2 α ( ˆ P, ˆ α ) = O p ( N -1 / 2 ), which follows from Lemma 8(c), the rootN consistency of ( ˆ P, ˆ θ ), and Lemma 7(b). Finally, for the bound of R N,j , a similar expansion gives sup ( P,α ) || D PP ψ 2 α ( P, α ) || , sup ( P,α ) || D αP ψ 2 α ( P, α ) || = O p ( N -1 / 2 + || ˜ α j -1 -ˆ α || + || ˜ P j -1 -ˆ P || ) with the range of the supremum stated above. Lemma 7(b) gives sup ( P,α ) || D αα ψ 2 α ( P, α ) || = O p (1).

Combining all the bounds in conjunction with Q N ( ˜ P j -1 , ˜ α j -1 ) → p E ( ∂ 2 /∂α∂α ′ ) ln P θ 0 gives ˆ α -˜ α j = O p ( || ˜ α j -1 -ˆ α || 2 + N -1 / 2 || ˜ P j -1 -ˆ P || 2 + || ˜ P j -1 -ˆ P || 3 (+ O p ( N -1 / 2 || ˜ α j -1 -ˆ α || + || ˜ P j -1 -ˆ P || 2 ) for OPG).

We complete the proof by showing the bound of ˜ P j -ˆ P . Similarly to the proof of Proposition 2, expanding ˜ P j = Ψ( ˜ P j -1 , ˜ α j ) around ( ˆ P, ˆ α ) and applying D P Ψ( ˆ P, ˆ α ) = 0 and Assumption 4(g) gives ˜ P j = ˆ P + O p ( || ˜ α j -ˆ α || + || ˜ P j -1 -ˆ P || 2 ) = ˆ P + O p ( || ˜ α j -ˆ α || ). The required result follows by induction. /square

## 8.5 Proof of Proposition 5

The proof follows the proofs of Proposition 2 and 3. Because the MLE maximizes the objective function for the NPL algorithm if P = ˆ P , the first order conditions give D ζ L PL ( ˆ P PL k -1 , ˆ ζ PL k ) = D ζ L PL ( ˆ P , ˆ ζ ) = 0. Assume ˆ P PL k -1 -P 0 = o p (1), then applying the generalized Taylor's theorem and following the argument used to prove Proposition 2 in conjunction with Lemma 1 gives [ E ζ 0 D ζζ l PL ( w i ; P 0 , ζ 0 ) + o p (1)]( ˆ ζ PL k -ˆ ζ ) = O p ( N -1 / 2 || ˆ P PL k -1 -ˆ P || + || ˆ P PL k -1 -ˆ P || 2 ). The stated result follows because E ζ 0 D ζζ l PL ( w i ; P 0 , ζ 0 ) is negative definite. The bound of ˆ P PL k -ˆ P can be shown by expanding Ψ( P PL,m k -1 , ˆ θ PL,m k ) around ( P ˆ θ m , ˆ θ m ) and applying D P Ψ( P ˆ θ m , ˆ θ m ) = 0. The required result follows by induction.

In case of the NMPL algorithm, we have D ζ L MPL ( ˆ P MPL k -1 , ˆ ζ MPL k ) = D ζ L MPL ( ˆ P , ˆ ζ ) = 0 as the first order conditions. The result follows from repeating the argument of the proof of Proposition 3 in conjunction with Lemma 1 and D PP ζ L MPL N ( P 0 , ζ 0 ) , D ζ P ζ L MPL N ( P 0 , ζ 0 ) = O p ( N -1 / 2 ), which holds because E ζ 0 D PP ζ l MPL ( w i ; P 0 , ζ 0 ) = 0 and E ζ 0 D ζ P ζ l MPL ( w i ; P 0 , ζ 0 ) = 0 from the chain rule, D P Ψ ( P 0 , ζ 0 ) = 0, E ζ 0 D PP l PL ( w i ; P 0 , ζ 0 ) = 0 and E ζ 0 D P ζ l PL ( w i ; P 0 , ζ 0 ) = 0. /square

## 8.6 Proof of Lemma 1

First, D P l PL ( w i ; P ζ , ζ ) = D P l MPL ( w i ; P ζ , ζ ) = 0 follows from the chain rule and Proposition 1. We proceed to prove the orthogonality results. D P l PL ( w i ; P ζ , ζ ) = 0 and the information matrix equality imply that E ζ 0 D P ζ l PL ( w i ; P 0 , ζ 0 ) = 0. It follows that D P ζ L PL N ( P 0 , ζ 0 ) = O p ( N -1 / 2 ) since w i is iid. Then, D P ζ L PL N ( P ˆ ζ , ˆ ζ ) = O p ( N -1 / 2 ) follows from expanding D P ζ L PL N ( P ˆ ζ , ˆ ζ ) around ( P 0 , ζ 0 ) and using ˆ P -P 0 , ˆ ζ -ζ 0 = O p ( N -1 / 2 ) and Assumptions 4(g) and 4UH(g). For the NMPL algorithm, D P ζ l MPL ( w i ; P ζ , ζ ) = 0 follows from the chain rule, D P l PL ( w i ; P ζ , ζ ) = 0, and D P Ψ ( P ζ , ζ ) = 0. /square

## 8.7 Proof of Lemma 2

The stated result follows from applying the proof of Theorem 6.1 of A05. Note that only Lemmas A.6, A.7, and A.8 of A05 are used in his proof. Our Lemma 10 corresponds to Lemma A.6 of A05. The results of Lemmas A.7 and A.8 of A05 hold in our case, because we can replace Lemmas A.4 and A.6 of A05 in the proof of Lemmas A.7 and A.8 of A05 with our Lemmas 9 and 10 and the proof carries through. /square

## 8.8 Proof of Lemma 3

The proof follows the same line of approach as the proof of Theorem 7.1 of Andrews (2005). The detail is presented in a supplementary appendix available from the authors upon request.

## 8.9 Proof of Lemma 4 and 5

These lemmas correspond to Theorems 7.1(b) and 7.2 of A05. They are proven by applying the argument of pp. 206-7 of A05. /square

## 9 Appendix B: Auxiliary results

Lemma 7 collects the bounds that are used in the proof of Propositions 2-4 and Lemma 3. Lemma 8 collects the results on the derivatives of ln Ψ 2 ( P, θ ). Lemma 9 is our version (i.e., for ˆ α and ˆ θ f ) of Lemma A.4 of A05. Lemma 10 is our version (i.e., for ˆ α and ˆ θ f ) of Lemma A.6 of A05. Their proofs are presented in a supplementary appendix available from the authors upon request.

Lemma 7 Suppose Assumptions 1-5 hold, ¯ P → p P 0 , and ¯ θ → p θ 0 . Let ψ i ( P, θ ) denote either ln Ψ( P, θ )( a i | x i ) or ln Ψ 2 ( P, θ )( a i | x i ) . Then

<!-- formula-not-decoded -->

If Assumptions 1-8 hold, then (b) holds for ( P, θ ) ∈ B P × Θ 1 .

(f) Suppose Assumptions 1-8 hold. Then, for all ε &gt; 0 and c &gt; 0 , sup θ 0 ∈ Θ 1 Pr( || N -1 ∑ N i =1 D Pα ln Ψ( P 0 , θ 0 )( a i | x i ) || &gt; εN -1 / 2 ln N ) = o ( N -c ) .

Lemma 8 Suppose Assumptions 1-4 hold. Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 9 Suppose Assumptions 1-8 hold. Then, for all ε &gt; 0 ,

<!-- formula-not-decoded -->

Lemma 10 Suppose Assumptions 1-8 hold. Define S N ( θ ) = N -1 ∑ N i =1 h ( w i , θ ) and ˆ θ = (ˆ α ′ , ˆ θ ′ f ) ′ . Let ∆ N ( θ 0 ) denote N 1 / 2 ( ˆ θ -θ 0 ) , T N ( θ 0 r ) , or H N ( ˆ θ, θ 0 ) . Let L denote the dimension of ∆ N ( θ 0 ) . For each definition of ∆ N ( θ 0 ) , there is an infinitely differentiable function G ( · ) that does not depend on θ 0 and that satisfies G ( E θ 0 S N ( θ 0 )) = 0 for all N large and all θ 0 ∈ Θ 1 , and sup θ 0 ∈ Θ 1 sup B ∈B L ∣ ∣ Pr θ 0 (∆ N ( θ 0 ) ∈ B ) -Pr θ 0 ( N 1 / 2 G ( S N ( θ 0 )) ∈ B ) ∣ ∣ = o ( N -c ) , where B L denotes the class of all convex sets in R L .

## References

Aguirregabiria, V., 1999. The dynamics of markups and inventories in retailing firms. Review of Economic Studies 66, 275-308.

Aguirregabiria, V., P. Mira, P., 2002. Swapping the nested fixed point algorithm: a class of estimators for discrete Markov decision models. Econometrica 70, 1519-1543.

Aguirregabiria, V., P. Mira, P., 2007. Sequential estimation of dynamic discrete games. Econometrica 75, 1-53.

Ahn, N., 1995. Measuring the value of children by sex and age using a dynamic programming model. Review of Economic Studies 62, 361-379.

Altug, S., Miller, R. A., 1998. The effect of work experience on female wages and labour supply. Review of Economic Studies 65, 45-85.

Andrews, D. W. K., 2002a. Equivalence of the higher order asymptotic efficiency of k -step and extremum statistics. Econometric Theory 18, 1040-1085.

Andrews, D. W. K., 2002b. Higher-order improvements of a computationally attractive k -step bootstrap for extremum estimators. Econometrica 70, 119-162.

Andrews, D. W. K., 2005. Higher-order improvements of the parametric bootstrap for Markov processes. In: Andrews, D. W. K., Stock, J. H. (Eds.), Identification and Inference for Econometric Models: Essays in Honor of Thomas Rothenberg. Cambridge University Press, pp. 171-215.

Arcidiacono, P., 2004. Ability sorting and the returns to college major. Journal of Econometrics 121, 343-375.

Arcidiacono, P., 2005. Affirmative action in higher education: how do admission and financial aid rules affect future earnings? Econometrica 73, 1477-1524.

Bajari, P., Benkard, C.L., Levin, J., 2007. Estimating dynamic models of imperfect competition. Econometrica 75, 1331-1370.

Bajari, P., Hong, H., 2006. Semiparametric estimation of a dynamic game of incomplete information. NBER Technical Working Paper 320.

Berkovec, J., Stern, S., 1991. Job exit behavior of older man. Econometrica 59, 189-210.

Cooper, R., Haltiwanger, J., Power, L., 1999. Machine replacement and the business cycle: lumps and bumps. American Economic Review 89, 921-946.

Das, S., 1992. A micro-econometric model of capital utilization and retirement: the case of the U.S. cement industry. Review of Economic Studies 59, 277-297.

Davidson, R., MacKinnon, J. G., 1999a. Bootstrap testing in nonlinear models. International Economic Review 40, 487-508.

Davidson, R., MacKinnon, J. G., 1999b. The size distortion of bootstrap tests. Econometric Theory 15, 361-376.

Eckstein, Z., Wolpin, K., 1999. Why youth drop out of high school: the impact of preferences, opportunities and abilities. Econometrica 67, 1295-1339.

Gilleskie, D., 1998. A dynamic stochastic model of medical care use and work absence. Econometrica 66, 1-45.

Gotz, G. A., McCall, J. J., 1980. Estimation in sequential decisionmaking models: a methodological note. Economics Letters 6, 131-136.

G¨ on¨ ul, F. F., 1999. Estimating price expectations in the OTC medicine market: an application of dynamic stochastic discrete choice models to scanner panel data. Journal of Econometrics 89, 41-56.

Heyma, A., 2004. A structural dynamic analysis of retirement behaviour in the Netherlands. Journal of Applied Econometrics 19, 739-759.

Hotz, J., Miller, R. A., 1993. Conditional choice probabilities and the estimation of dynamic models. Review of Economic Studies 60, 497-529.

Ichimura, H., Lee, S., 2006. Characterization of the asymptotic distribution of semiparametric M-estimators. Mimeographed, University of Tokyo.

Imai, S., Jain, N., Ching, A., 2006. Bayesian Estimation of Dynamic Discrete Choice Models. Mimeographed, Queen's University.

Imai, S., Keane, M., 2004. Intertemporal labor supply and human capital accumulation. International Economic Review 45, 601-641.

Janssen, P., Jureckova, J., Veraverbeke, N., 1985. Rate of convergence of one- and two-step Mestimators with applications to maximum likelihood and Pitman estimators. Annals of Statistics 13, 1222-1229.

Kasahara, H., Lapham, B., 2007. Productivity and the decision to import and export: theory and evidence. Mimeographed, University of Western Ontario.

Kasahara, H., Shimotsu, K., 2006. Nonparametric Identification and Estimation of Finite Mixture Models of Dynamic Discrete Choices. Queen's Economics Department Working paper No. 1092.

Kasahara, H., Shimotsu, K., 2007. Nonparametric Identification of Finite Mixture Models of Dynamic Discrete Choices. Mimeographed, Queen's University.

Kennet, M., 1994. A structural model of aircraft engine maintenance. Journal of Applied Econometrics 9, 351-368.

Keane, M. P., Wolpin, K. I., 1997. The career decisions of young men. Journal of Political Economy 105, 473-522.

Kennan, J., Walker, J. R., 2006. The effect of expected income on individual migration decisions. Mimeographed, University of Wisconsin-Madison.

Kim, J. H., 2005. Higher-order improvements of the restricted parametric bootstrap for tests. Mimeographed, National University of Singapore.

Miller, R. A., 1984. Job matching and occupational choice. Journal of Political Economy 92, 1086-1120.

Miller, R. A., Sanders, S. G., 1997. Human capital development and welfare participation. Carnegie-Rochester Conference Series on Public Policy 46, 1-43.

Pakes, A., 1986. Patents as options: some estimates of the value of holding european patent stocks. Econometrica 54, 755-784.

Pakes, A., Ostrovsky, M., Berry, S., 2007. Simple estimators for the parameters of discrete dynamic games (with entry/exit examples). RAND Journal of Economics 38, 373-399.

Pesendorfer, M., Schmidt-Dengler, P., 2006. Asymptotic least squares estimators for dynamic games. Mimeographed, LSE.

Pfanzagl, J., 1974. Asymptotic optimum estimation and test procedures. In: H´ ajek, J. (Ed.), Proceedings of the Prague Symposium on Asymptotics, Vol. 1. Charles University, Prague, pp. 201-272.

Robinson, P. M., 1988. The stochastic difference between econometric statistics. Econometrica 56, 531-548.

Rothwell, G., Rust, J., 1997. On the optimal lifetime of nuclear power plants. Journal of Business and Economic Statistics 15, 195-208.

Rota, P., 2004. Estimating labor demand with fixed costs, International Economic Review 45, 25-48.

Rust, J., 1987. Optimal replacement of GMC bus engines: an empirical model of Harold Zurcher. Econometrica 55, 999-1033.

Rust, J., Phelan, C., 1997. How social security and medicare affect retirement behavior in a world of incomplete markets. Econometrica 65, 781-831.

Slade, M., 1998. Optimal pricing with costly adjustment: evidence from retail-grocery prices. Review of Economic Studies 65, 87-107.

Wolpin, K., 1984. An Estimable Dynamic Stochastic Model of Fertility and Child Mortality. Journal of Political Economy 92, 852-874.

Zeidler, E., 1986. Nonlinear Functional Analysis and its Applications I: Fixed-Point Theorems. Springer-Verlag, New York.

Table 1: Convergence of NPL/NMPL-based estimators to MLE

|     |   NPL |   NMPL |   1-NPL |   1-NMPL |
|-----|-------|--------|---------|----------|
| k=1 | 0.042 |  0.008 |   0.042 |    0.042 |
| k=2 | 0.002 |  0     |   0.021 |    0.021 |
| k=3 | 0     |  0     |   0.009 |    0.007 |
| k=4 | 0     |  0     |   0.006 |    0.004 |
| k=5 | 0     |  0     |   0.004 |    0.002 |
| k=1 | 0.394 |  0.042 |   0.394 |    0.394 |
| k=2 | 0.03  |  0.003 |   0.224 |    0.208 |
| k=3 | 0.003 |  0     |   0.093 |    0.078 |
| k=4 | 0     |  0     |   0.041 |    0.035 |
| k=5 | 0     |  0     |   0.025 |    0.021 |
| k=1 | 0.473 |  0.137 |   0.473 |    0.473 |
| k=2 | 0.03  |  0.01  |   0.246 |    0.294 |
| k=3 | 0.002 |  0     |   0.065 |    0.076 |
| k=4 | 0     |  0     |   0.037 |    0.02  |
| k=5 | 0     |  0     |   0.012 |    0.009 |

Notes: The reported values for the column of 'NPL', for instance, are the mean of | (NPL statistic-MLE)/MLE | for k = 1 , ..., 5 across 1000 replications.

Table 2: Performance of NPL/NMPL-based estimators

|     |     MLE |     NPL |    NMPL |   1-NPL |   1-NMPL |
|-----|---------|---------|---------|---------|----------|
| k=1 |  0.0783 | -0.0075 |  0.0627 | -0.0075 |  -0.0075 |
| k=2 |  0.0783 |  0.078  |  0.0778 |  0.0475 |   0.043  |
| k=3 |  0.0783 |  0.0782 |  0.0783 |  0.0742 |   0.0768 |
| k=4 |  0.0783 |  0.0783 |  0.0783 |  0.0771 |   0.0788 |
| k=5 |  0.0783 |  0.0783 |  0.0783 |  0.0786 |   0.0801 |
| k=1 |  0.1414 |  0.1451 |  0.1443 |  0.1451 |   0.1451 |
| k=2 |  0.1414 |  0.142  |  0.1415 |  0.1484 |   0.1456 |
| k=3 |  0.1414 |  0.1415 |  0.1414 |  0.1465 |   0.1448 |
| k=4 |  0.1414 |  0.1414 |  0.1414 |  0.1441 |   0.143  |
| k=5 |  0.1414 |  0.1414 |  0.1414 |  0.1434 |   0.1434 |
| k=1 | -0.1174 | -0.1243 | -0.1177 | -0.1243 |  -0.1243 |
| k=2 | -0.1174 | -0.1175 | -0.1175 | -0.1069 |  -0.114  |
| k=3 | -0.1174 | -0.1173 | -0.1174 | -0.1144 |  -0.1175 |
| k=4 | -0.1174 | -0.1174 | -0.1174 | -0.1137 |  -0.1168 |
| k=5 | -0.1174 | -0.1174 | -0.1174 | -0.1163 |  -0.1191 |
| k=1 |  0.2251 |  0.1561 |  0.2084 |  0.1561 |   0.1561 |
| k=2 |  0.2251 |  0.2218 |  0.223  |  0.1732 |   0.1747 |
| k=3 |  0.2251 |  0.2244 |  0.2251 |  0.2063 |   0.2099 |
| k=4 |  0.2251 |  0.2251 |  0.2251 |  0.2065 |   0.2149 |
| k=5 |  0.2251 |  0.2251 |  0.2251 |  0.2206 |   0.2267 |
| k=1 |  0.0237 | -0.0096 |  0.0145 | -0.0096 |  -0.0096 |
| k=2 |  0.0237 |  0.0228 |  0.0231 |  0.0103 |   0.0069 |
| k=3 |  0.0237 |  0.0237 |  0.0237 |  0.0201 |   0.0202 |
| k=4 |  0.0237 |  0.0237 |  0.0237 |  0.0224 |   0.0228 |
| k=5 |  0.0237 |  0.0237 |  0.0237 |  0.0233 |   0.0235 |
| k=1 |  0.003  |  0.0058 |  0.0039 |  0.0058 |   0.0058 |
| k=2 |  0.003  |  0.0031 |  0.0031 |  0.004  |   0.0042 |
| k=3 |  0.003  |  0.003  |  0.003  |  0.0034 |   0.0034 |
| k=4 |  0.003  |  0.003  |  0.003  |  0.0031 |   0.0031 |
| k=5 |  0.003  |  0.003  |  0.003  |  0.0031 |   0.0031 |
| k=0 |  0.0157 |  0.5601 |  0.5601 |  0.5601 |   0.5601 |
| k=1 |  0.0157 |  0.0257 |  0.0272 |  0.0259 |   0.0259 |
| k=2 |  0.0157 |  0.0156 |  0.0152 |  0.0239 |   0.0294 |
| k=3 |  0.0157 |  0.0157 |  0.0155 |  0.0196 |   0.0211 |
| k=4 |  0.0157 |  0.0157 |  0.0157 |  0.0163 |   0.0166 |
| k=5 |  0.0157 |  0.0157 |  0.0157 |  0.0158 |   0.0161 |

Notes: The reported values for the last six rows are the average of the mean squared errors of ln ˆ P ( a = 1 | s, ω ) across different states ( s, ω ); k = 0 represents the initial sieve logit estimator.

Table 3: Rejection Frequencies for Bootstrap Wald test at .10, .05, and .01 Levels

|                      |                      |   Significance |   Significance |   Levels |
|----------------------|----------------------|----------------|----------------|----------|
|                      |                      |          0.1   |          0.05  |    0.01  |
| Asymptotic Wald test | Asymptotic Wald test |          0.135 |          0.097 |    0.059 |
| Bootstrap MLE        | Bootstrap MLE        |          0.088 |          0.04  |    0.016 |
| Bootstrap MLE-NULL   | Bootstrap MLE-NULL   |          0.084 |          0.039 |    0.006 |
| Bootstrap NPL        | k = 1                |          0.086 |          0.035 |    0.012 |
| Bootstrap NPL        | k = 3                |          0.09  |          0.042 |    0.016 |
| Bootstrap NMPL       | k = 1                |          0.082 |          0.041 |    0.004 |
| Bootstrap NMPL       | k = 3                |          0.083 |          0.041 |    0.004 |
| Bootstrap 1-NPL      | k = 1                |          0.026 |          0.007 |    0     |
| Bootstrap 1-NPL      | k = 3                |          0.087 |          0.046 |    0.003 |
| Bootstrap 1-NPL      | k = 5                |          0.09  |          0.048 |    0.007 |
| Bootstrap 1-NMPL     | k = 1                |          0.029 |          0.005 |    0     |
| Bootstrap 1-NMPL     | k = 3                |          0.078 |          0.042 |    0.001 |
| Bootstrap 1-NMPL     | k = 5                |          0.08  |          0.044 |    0.011 |

Notes: Based on 1000 simulated samples. The sample size is N = 1000 while the number of bootstrap samples is 599. The null hypothesis is ( θ 1 , θ 2 ) = ( -0 . 2 , 0 . 1).

Table 4: Coverage Performance of Bootstrap 90% and 95% CIs for parameters θ 0 , θ 1 , and θ 2

|                  |                | 95% CIs   | 95% CIs   | 95% CIs   | 90% CIs   | 90% CIs   | 90% CIs   |
|------------------|----------------|-----------|-----------|-----------|-----------|-----------|-----------|
|                  |                | θ 0       | θ 1       | θ 2       | θ 0       | θ 1       | θ 2       |
| Asymptotic CIs   | Asymptotic CIs | 0.935     | 0.922     | 0.962     | 0.898     | 0.891     | 0.909     |
| Bootstrap MLE    | Bootstrap MLE  | 0.929     | 0.949     | 0.961     | 0.893     | 0.903     | 0.909     |
| Bootstrap NPL    | k = 1          | 0.928     | 0.951     | 0.958     | 0.891     | 0.902     | 0.909     |
| Bootstrap NPL    | k = 3          | 0.929     | 0.947     | 0.962     | 0.894     | 0.901     | 0.909     |
| Bootstrap NMPL   | k = 1          | 0.934     | 0.949     | 0.960     | 0.896     | 0.881     | 0.916     |
| Bootstrap NMPL   | k = 3          | 0.934     | 0.949     | 0.960     | 0.896     | 0.881     | 0.916     |
| Bootstrap 1-NPL  | k = 1          | 0.948     | 0.987     | 0.955     | 0.913     | 0.937     | 0.912     |
| Bootstrap 1-NPL  | k = 3          | 0.948     | 0.935     | 0.962     | 0.913     | 0.884     | 0.914     |
| Bootstrap 1-NPL  | k = 5          | 0.950     | 0.935     | 0.967     | 0.916     | 0.886     | 0.914     |
| Bootstrap 1-NMPL | k = 1          | 0.953     | 0.985     | 0.961     | 0.925     | 0.932     | 0.916     |
| Bootstrap 1-NMPL | k = 3          | 0.954     | 0.933     | 0.968     | 0.921     | 0.876     | 0.910     |
| Bootstrap 1-NMPL | k = 5          | 0.953     | 0.930     | 0.968     | 0.925     | 0.879     | 0.912     |

Notes: Based on 1000 simulated samples, each with the sample size of 1000. The number of bootstrap samples is 599.