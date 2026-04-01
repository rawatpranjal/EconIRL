## An Empirical Risk Minimization Approach for Offline Inverse RL and Dynamic Discrete Choice Model

Enoch H. Kang *

Foster School of Business, University of Washington Hema Yoganarasimhan Foster School of Business, University of Washington Lalit Jain Foster School of Business, University of Washington January 16, 2026

## Abstract

We study the problem of estimating Dynamic Discrete Choice (DDC) models, also known as offline Maximum Entropy-Regularized Inverse Reinforcement Learning (offline MaxEnt-IRL) in machine learning. The objective is to recover the reward function that governs agent behavior from offline behavior data. In this paper, we propose a globally convergent gradient-based method for solving these problems without the restrictive assumption of linearly parameterized rewards. The novelty of our approach lies in introducing the Empirical Risk Minimization (ERM) based IRL/DDC framework, which circumvents the need for explicit state transition probability estimation in the Bellman equation. Furthermore, our method is compatible with non-parametric estimation techniques such as neural networks. Therefore, the proposed method has the potential to be scaled to high-dimensional, infinite state spaces. A key theoretical insight underlying our approach is that the Bellman residual satisfies the Polyak-Łojasiewicz (PL) condition -- a property that, while weaker than strong convexity, is sufficient to ensure fast global convergence guarantees. Through a series of synthetic experiments, we demonstrate that our approach consistently outperforms benchmark methods and state-of-the-art alternatives.

Keywords: Dynamic Discrete Choice, Offline Inverse Reinforcement Learning, Gradient-based methods, Empirical Risk Minimization, Neural Networks

* We would like to thank participants of the 2024 University of Washington Marketing PhD Workshop, the 2025 ACM Conference on Economics and Computation, 2025 AIM conference, and the Dynamic Structural Econometrics 2025 Summer School. We also thank Kyoungseok Jang, John Rust, Zikun Ye, and the anonymous reviewers at EC 2025 for their detailed feedback, which has significantly improved the paper. Please ask all correspondance to: ehwkang@uw.edu, hemay@uw.edu and lalitj@uw.edu

## 1 Introduction

Learning from previously collected datasets has become an essential paradigm in sequential decision-making problems where exploration during interactions with the environment is infeasible (e.g., self-driving cars, medical applications) or leveraging large-scale offline data is preferable (e.g., social science, recommendation systems, and industrial automation) (Levine et al. 2020). However, in such cases, defining a reward function (a flow utility function) that accurately captures the underlying decision-making process is often challenging due to the unobservable/sparse rewards (Zolna et al. 2020) and complexity of real-world environments (Foster et al. 2021). To circumvent these limitations, learning from expert demonstrations has gained prominence, motivating approaches such as Imitation Learning (IL) and offline Inverse Reinforcement Learning (offline IRL) or equivalently, Dynamic Discrete Choice (DDC) model estimation 1 .

While IL directly learns a policy by mimicking expert actions, it is susceptible to distribution shift , i.e., when the testing environment (reward, transition function) is different from the training environment. On the other hand, offline IRL (or DDC) aims to infer the underlying reward function that best explains expert behavior. Given that this reward function is identified under a suitable normalization assumption, a new policy can be trained after a change in the environment's transition dynamics (e.g., modifications in recommendation systems) or in the reward function (e.g., marketing interventions). This capability enables offline IRL (or DDC) to be employed in counterfactual simulations, such as evaluating the effects of different policy decisions without direct experimentation. However, an imprecise reward function can lead to suboptimal policy learning and unreliable counterfactual analyses, ultimately undermining its practical utility. As a result, offline IRL (or DDC)'s key metric becomes the precision of reward inference.

While the precise reward function estimation objective has been studied in recent offline IRL literature, theoretically guaranteed existing methods that satisfy the Bellman equation have been limited to explicitly learning a transition model (e.g., Zeng et al. (2023)). However, the statistical complexity for learning a transition model increases exponentially as the state dimension increases. Furthermore, if relearning the transition function is required every time it changes, the premise of IRL for counterfactual simulations may be undermined. In addition, the errors from the transition function estimation can propagate and compound in the later estimation stage. The Dynamic Discrete Choice (DDC) literature in econometrics has separately explored the problem towards the goal of precise reward estimation (Rust 1994; Hotz and Miller 1993; Aguirregabiria and Mira 2007; Su and Judd 2012; Adusumilli and Eckardt 2019; Kristensen et al. 2021; Geng et al. 2023). However, existing methodologies with theoretical precision guarantees suffer from the curse of dimensionality (computational or statistical complexity exponentially grows as state dimension increases (Kristensen et al. 2021)) or algorithmic instability beyond linear reward functions (Adusumilli and Eckardt 2019). This motivates us to ask the following question:

Can we propose a scalable gradient-based method to infer rewards (or Q ∗ function) while provably ensuring global optimality with no assumption on reward structure/transition function knowledge?

1 Refer to Section 3.3 for the equivalence between Offline Maximum Entropy IRL (MaxEnt-IRL) and DDC.

Our contributions. In this paper, we propose an Empirical Risk Minimization (ERM)-based gradient-based method for IRL/DDC as an inverse Q-learning method. This method provably finds the true parameter θ for Q ∗ estimation (up to statistical error, which diminishes at an O ( N -1 / 2 ) rate with N samples) with O ( T -1 / 4 ) rate of convergence, where T is the number of gradient iterations. In addition, the true reward function can be computed from the estimated Q ∗ with no extra statistical or computational cost, given the estimated Q ∗ function. In developing this method, we make the following technical contributions:

- We propose an empirical risk minimization (ERM) problem formulation, which we refer to as ERM-IRL in the IRL literature and ERM-DDC in the DDC literature, reflecting the shared problem. This formulation allows us to circumvent the need for explicit transition function estimation. 2 Notably, this formulation also allows us to conclude that imitation learning (IL) is a strictly easier problem than IRL/DDC estimation problem.
- We show that the objective function of the ERM-IRL satisfies the Polyak-Łojasiewicz (PL) condition, which is a weaker but equally useful alternative to strong convexity for providing theoretical convergence guarantees. This is enabled by showing that each of its two components - expected negative log-likelihood and mean squared Bellman error - satisfies the PL condition 3 .
- Since the mean squared Bellman error term is a solution to a strongly concave inner maximization problem (Dai et al. 2018; Patterson et al. 2022), minimization of the ERM-IRL objective becomes a mini-max problem with two-sided PL condition (Yang et al. 2020). Using this idea, we propose an alternating gradient ascent-descent algorithm that provably converges to the true Q ∗ , which is the unique saddle point of the problem.

In addition to establishing theoretical global convergence guarantees, we demonstrate the empirical effectiveness of the algorithm through standard benchmark simulation experiments. Specifically, we evaluate using a series of simulations: (1) The Rust bus engine replacement problem (Rust 1987), which is the standard framework for evaluation used in the dynamic discrete choice literature, and (2) A high-dimensional variant of the Rust bus-engine problem, where we allow a very large state space. In both settings, we show that our algorithm outperforms/matches the performance of existing approaches. It is particularly valuable in large state-space settings, where many of the standard algorithms become infeasible due to their need to estimate state-transition probabilities. We expect our approach to be applicable to a variety of business and economic problems where the state and action space are infinitely large, and firms/policy-makers do not have a priori knowledge of the parametric form of the reward function and/or state transitions.

The remainder of the paper is organized as follows. In Section 2, we discuss related work in greater detail. Section 3 introduces the problem setup and provides the necessary background. In Section 4, we present

2 Transition function estimation can also be avoided in the counterfactual reasoning stage. Once the estimated reward function ˆ r is in hand, one can perform counterfactual policy evaluation and optimization by running an offline RL routine over a reward info-augmented data of ( s, a, ˆ r ) tuples.

3 The sum of two PL functions is not necessarily PL; in the proof, we show that our case is an exception.

the ERM-IRL framework, followed by an algorithm for solving it in Section 5. Section 6 establishes the global convergence guarantees of the proposed algorithm. Finally, Section 7 presents experimental results demonstrating the effectiveness of our approach.

## 2 Related works

The formulations of DDC and (Maximum Entropy) IRL are fundamentally equivalent (see Section 3.3 for details). In the econometrics literature, stochastic decision-making behaviors are usually considered to come from the random utility model (McFadden 2001), which assumes that the effect of unobserved covariates appears in the form of additive and conditionally independent randomness in agent utilities (Rust 1994). On the other hand, in the computer science literature, stochastic decision-making behaviors are modeled as a 'random choice'. That is, the assumption is that agents play a stochastic strategy (where they randomize their actions based on some probabilities). This model difference, however, is not a critical differentiator between the two literatures. The two modeling choices yield equivalent optimality equations, meaning that the inferred rewards are identical under both formulations (Ermon et al. 2015).

The main difference between DDC and IRL methods stems from their distinct objectives. DDC's objective is to estimate the exact reward function that can be used for subsequent counterfactual policy simulations (e.g., sending a coupon to a customer to change their reward function). Achieving such strong identifiability necessitates a strong anchor action assumption (Assumption 3). A direct methodological consequence of this pursuit for an exact function is the requirement to solve the Bellman equation, which in turn causes significant scalability issues. On the other hand, IRL's objective is to identify a set of reward functions that are compatible with the data. This allows for a weaker identification assumption (Ng et al. 1999) than DDC's assumption (Assumption 3) and avoids the computational burden inherent in the DDC framework caused by the Bellman equation.

Table 1 compares DDC and IRL methods based on several characteristics, which are defined here as they correspond to the table's columns. The first set of characteristics is typically satisfied by IRL methods but not by DDC methods with global optimality. One-shot optimization indicates that a method operates on a single timescale without requiring an inner optimization loop or inner numerical integration like forward roll-outs. Transition Estimation-Free signifies that the method avoids the explicit estimation of a transition function. A method is considered Gradient-based if its primary optimization process relies on gradient descent, and Scalable if it can handle state spaces of at least 20 10 .

Conversely, a different set of characteristics is often satisfied by DDC methods. The Bellman equation criterion is met if the estimation procedure fits the estimated r -function or Q ∗ -function to the Bellman equation; this definition excludes occupancy-matching methods (e.g., IQ-Learn (Garg et al. 2021), Clare (Yue et al. 2023); see Appendix B.6) and the semi-gradient method (Adusumilli and Eckardt 2019), which minimizes the projected squared Bellman error for linear value functions (Sutton and Barto 2018). Global optimality refers to a theoretical guarantee of convergence to the globally optimal r or Q ∗ beyond linear value function approximation. The △ symbol indicates a conditional guarantee; for instance, Approximate Value Iteration (A VI) (Adusumilli and Eckardt 2019) is marked with a △ due to the known instability of fitted

fixed-point methods beyond the linear reward/value class (Wang et al. 2021; Jiang and Xie 2024). ML-IRL is also marked with a △ because its guarantee applies only to linear reward functions. For methods that achieve global optimality, the table also lists their sample complexity. A rate of 1 / √ N implies that the estimated parameter ˆ θ converges to the true parameter θ ∗ at that rate, where N is the sample size.

Table 1: Comparison of DDC and IRL methods.

| Method                                                                    | One-shot Optimization   | Transition Estimation-Free   | Gradient- Based   | Scalability   | Bellman Equation   | Global Optimality   | Statistical Complexity   |
|---------------------------------------------------------------------------|-------------------------|------------------------------|-------------------|---------------|--------------------|---------------------|--------------------------|
| DDC Methods                                                               |                         |                              |                   |               |                    |                     |                          |
| NFXP (Rust 1987)                                                          |                         |                              |                   |               | ✓                  | ✓                   | 1 / √ N                  |
| CCP (Hotz and Miller 1993)                                                |                         |                              |                   |               | ✓                  | ✓                   | 1 / √ N                  |
| MPEC                                                                      |                         |                              |                   |               | ✓                  | ✓                   | 1 / √ N                  |
| (Su and Judd 2012) AVI                                                    |                         | ✓                            |                   |               | ✓                  | △ (unstable)        |                          |
| (Adusumilli and Eckardt 2019) Semi-gradient (Adusumilli and Eckardt 2019) |                         | ✓                            | ✓                 | ✓             |                    |                     |                          |
| RP (Barzegary and Yoganarasimhan 2022)                                    |                         |                              |                   | ✓             | ✓                  |                     |                          |
| SAmQ (Geng et al. 2023)                                                   |                         | ✓                            |                   | ✓             | ✓                  |                     |                          |
| IRL Methods                                                               |                         |                              |                   |               |                    |                     |                          |
| BC (Torabi et al. 2018)                                                   | ✓                       | ✓                            | ✓                 | ✓             |                    |                     |                          |
| IQ-Learn (Garg et al. 2021)                                               | ✓                       | ✓                            | ✓                 | ✓             |                    |                     |                          |
| Clare (Yue et al. 2023)                                                   |                         |                              | ✓                 | ✓             |                    |                     |                          |
| ML-IRL (Zeng et al. 2023) Model-enhanced                                  |                         |                              | ✓                 | ✓             | ✓                  | △ (Linear only)     |                          |
| AIRL (Zhan et al. 2024)                                                   |                         | ✓                            |                   | ✓             |                    |                     |                          |
| Ours                                                                      | ✓                       | ✓                            | ✓                 | ✓             | ✓                  | ✓                   | 1 / √ N                  |

## 2.1 Dynamic discrete choice model estimation literature

The seminal paper by Rust (Rust 1987) pioneered this literature, demonstrating that a DDC model can be solved by solving a maximum likelihood estimation problem that runs above iterative dynamic programming. As previously discussed, this method is computationally intractable as the size of the state space increases.

Hotz and Miller (1993) introduced a method which is often called the two-step method conditional choice probability (CCP) method, where the CCPs and transition probabilities estimation step is followed by the reward estimation step. The reward estimation step avoids dynamic programming by combining simulation with the insight that differences in value function values can be directly inferred from data without solving Bellman equations. However, simulation methods are, in principle, trajectory-based numerical integration methods that also suffer scalability issues. Fortunately, we can sometimes avoid simulation altogether by utilizing the problem structure, such as regenerative/terminal actions (known as finite dependence (Arcidiacono

and Miller 2011)). Still, this method requires explicit estimation of the transition function, which is not the case in our paper. This paper established an insight that there exists a one-to-one correspondence between the CCPs and the differences in Q ∗ -function values, which was formalized as the identification result by Magnac and Thesmar (2002).

Su and Judd (2012) propose that we can avoid dynamic programming or simulation by formulating a nested linear programming problem with Bellman equations as constraints of a linear program. This formulation is based on the observation that Bellman equations constitute a convex polyhedral constraint set. While this linear programming formulation significantly increases the computation speed, it is still not scalable in terms of state dimensions.

As the above methods suffer scalability issues, methods based on parametric/nonparametric approximation have been developed. Parametric policy iteration (Benitez-Silva et al. 2000) and sieve value function iteration (Arcidiacono et al. 2013) parametrize the value function by imposing a flexible functional form. Kristensen et al. (2021) also proposed methods that combine smoothing of the Bellman operator with sievebased approximations, targeting the more well-behaved integrated or expected value functions to improve computational performance. However, standard sieve methods that use tensor product basis functions, such as polynomials, can suffer from a computational curse of dimensionality, as the number of basis functions required for a given accuracy grows exponentially with the number of state variables. Norets (2012) proposed that neural network-based function approximation reduces the computational burden of Markov Chain Monte Carlo (MCMC) estimation, thereby enhancing the efficiency and scalability. Also leveraging Bayesian MCMC techniques, Imai et al. (2009) developed an algorithm that integrates the DP solution and estimation steps, reducing the computational cost associated with repeatedly solving the underlying dynamic programming problem to be comparable to static models, though it still requires transition probabilities for calculating expectations. Arcidiacono et al. (2016) formulates the problem in continuous time, where the sequential nature of state changes simplifies calculations. However, these continuous-time models still rely on specifying transition dynamics via intensity matrices. Geng et al. (2020) proposed that the inversion principle of Hotz and Miller (1993) enables us to avoid reward parameterization and directly (non-parametrically) estimate value functions, along with solving a much smaller number of soft-Bellman equations, which do not require reward parametrization to solve them. Barzegary and Yoganarasimhan (2022) and Geng et al. (2023) independently proposed state aggregation/partition methods that significantly reduce the computational burden of running dynamic programming with the cost of optimality. While Geng et al. (2023) uses k -means clustering (Kodinariya et al. 2013; Sinaga and Yang 2020), Barzegary and Yoganarasimhan (2022) uses recursive partitioning (RP). As discussed earlier, combining approximation with dynamic programming induces unstable convergence except when the value function is linear in state (Jiang and Xie 2024).

Adusumilli and Eckardt (2019) proposed how to adapt two popular temporal difference (TD)-based methods (an approximate dynamic programming-based method and a semi-gradient descent method based on Tsitsiklis and Van Roy (1996a)) for DDC. As discussed earlier, approximate dynamic programming-based methods are known to suffer from a lack of provable convergence beyond linear reward models (Jiang and Xie

2024; Tsitsiklis and Van Roy 1996b; Van Hasselt et al. 2018; Wang et al. 2021) 4 ; the semi-gradient method is a popular, efficient approximation method that has theoretical assurance of convergence to projected squared Bellman error minimizers for linear reward/value functions (Sutton and Barto 2018). Feng et al. (2020) showed global concavity of value function under certain transition functions and monotonicity of value functions in terms of one-dimensional state, both of which are easily satisfied for applications in social science problems. However, those conditions are limitedly satisfied for the problems with larger dimensional state space.

## 2.2 Offline inverse reinforcement learning literature

The most widely used inverse reinforcement learning model, Maximum-Entropy inverse reinforcement learning (MaxEnt-IRL), assumes that the random choice happens due to agents choosing the optimal policy after penalization of the policy by its Shannon entropy (Ermon et al. 2015). In addition to the equivalence of MaxEnt-IRL to DDC (See Ermon et al. (2015) and Web Appendix § 3.3), the identifiability condition for DDC (Magnac and Thesmar 2002) was rediscovered by Cao et al. (2021) for MaxEnt-IRL. Zeng et al. (2023) proposes a two-step maximum likelihood-based method that can be considered as a conservative version of CCP method of Hotz and Miller (1993) 5 . Despite that their method is proven to be convergent, it requires the explicit estimation of the transition function, and its global convergence was only proven for linear reward functions.

Finn et al. (2016) showed that a myopic 6 version of MaxEnt-IRL can be solved by the Generative Adversarial Network (GAN) training framework (Goodfellow et al. 2020). Fu et al. (2017) extended this framework to a non-myopic version, which is proven to recover the reward function and the value function (up to policy invariance) but only under deterministic transitions. Note that GAN approaches identify rewards only up to policy invariance (Ng et al. 1999), which implies that counterfactual analysis is impossible. The GAN approach has been extended to Q -estimation methods that use fixed point iteration (Geng et al. 2020, 2023). Ni et al. (2021) has shown that the idea of training an adversarial network can also be used to calculate the gradient direction for minimizing the myopic version of negative log likelihood 7 . Zhan et al. (2024) proposed a GAN approach that is provably equivalent to negative log likelihood minimization objective without satisfying the Bellman equation (i.e., behavioral cloning (Torabi et al. 2018)). Indeed, GAN approaches are known to work well for Imitation Learning (IL) tasks (Zare et al. 2024). However, as discussed earlier, solving for IL does not allow us to conduct a counterfactual simulation.

A family of methods starting from Ho and Ermon (2016) tries to address the inverse reinforcement learning problem from the perspective of occupancy matching, i.e., finding a policy that best matches the behavior of data. Garg et al. (2021) proposed how to extend the occupancy matching approach of Ho and

4 In fact, 'fitted fixed point methods can diverge even when all of the following hold: infinite data, perfect optimization with infinite computation, 1-dimensional function class (e.g., linear) with realizability (Assumption 4 ) ... (the instability is) not merely a theoretical construction: deep RL algorithms are known for their instability and training divergence... ' (Jiang and Xie 2024)

5 When there is no uncertainty in the transition function, approximated trajectory gradient of Offline IRL method degenerates to forward simulation-based gradient in CCP estimator method of Hotz and Miller (1993).

6 See Cao et al. (2021) for more discussion on this.

7 Minimizing negative log-likelihood is equivalent to minimizing KL divergence. See the Proof of Lemma 20.

Ermon (2016) to directly estimate Q -function instead of r . Given the assumption that the Bellman equation holds, this approach allows a simple gradient-based solution, as the occupancy matching objective function they maximize becomes concave. Yue et al. (2023) modifies Ho and Ermon (2016) to conservatively deal with the uncertainty of transition function. Recently, occupancy matching-based inverse reinforcement learning has been demonstrated at a planetary scale in Google Maps, delivering significant global routing improvements Barnes et al. (2023). Despite their simplicity and scalability, one caveat of occupancy matching approaches is that the estimated Q from solving the occupancy matching objective may not satisfy the Bellman equation (See Appendix B.6). This also implies that computing r from Q using the Bellman equation is not a valid approach.

## 3 Problem set-up and backgrounds

We consider a single-agent Markov Decision Process (MDP) defined as a tuple ( S , A , P, ν 0 , r, β ) where S denotes the state space and A denotes a finite action space, P ∈ ∆ S×A S is a Markovian transition kernel, ν 0 ∈ ∆ S is the initial state distribution over S , r ∈ R S×A is a deterministic reward function and β ∈ (0 , 1) a discount factor. Given a stationary Markov policy π ∈ ∆ S A , an agent starts from initial state s 0 and takes an action a h ∈ A at state s h ∈ S according to a h ∼ π ( · | s h ) at each period h . Given an initial state s 0 ∼ ν 0 , we define the distribution of state-action sequences for policy π over the sample space ( S × A ) ∞ = { ( s 0 , a 0 , s 1 , a 1 , . . . ) : s h ∈ S , a h ∈ A , h ∈ N } as P ν 0 ,π . We also use E ν 0 ,π to denote the expectation with respect to P ν 0 ,π .

## 3.1 Setup: Maximum Entropy-Inverse Reinforcement Learning (MaxEnt-IRL)

Following existing literature (Geng et al. 2020; Fu et al. 2017; Ho and Ermon 2016), we consider the entropy-regularized optimal policy, which is defined as

<!-- formula-not-decoded -->

where H denotes the Shannon entropy and λ is the regularization coefficient. Throughout, we make the following assumption on the agent's decisions.

Assumption 1. When interacting with the MDP ( S , A , P, ν 0 , r, β ) , each agent follows the entropy-regularized optimal stationary policy π ∗ .

Throughout the paper, we use λ = 1 , the setting which is equivalent to dynamic discrete choice (DDC) model with mean zero T1EV distribution (see Web Appendix § 3.3 for details); all the results of this paper easily generalize to other values of λ . Given π ∗ , we define the value function V ∗ as:

<!-- formula-not-decoded -->

Similarly, we define the Q ∗ function as follows:

<!-- formula-not-decoded -->

Given state s and policy π ∗ , let q = [ q 1 . . . q |A| ] denote the probability distribution over the action space A , such that:

<!-- formula-not-decoded -->

Then, according to Assumption 1, the value function V ∗ must satisfy the recursive relationship defined by the Bellman equation as follows:

<!-- formula-not-decoded -->

Further, we can show that (see Web Appendix § B.3):

<!-- formula-not-decoded -->

Throughout, we define a function V Q as

<!-- formula-not-decoded -->

where V Q ∗ = V ∗ .

## 3.2 Setup: Dynamic Discrete Choice (DDC) model

Following the literature (Rust 1994; Magnac and Thesmar 2002), we assume that the reward the agent observes at state s ∈ S and a ∈ A can be expressed as r ( s, a ) + ϵ a , where ϵ a i.i.d. ∼ G ( δ, 1) is the random part of the reward, where G is Type 1 Extreme Value (T1EV) distribution (i.e., Gumbel distribution) 8 . The mean of G ( δ, 1) is δ + γ , where γ is the Euler constant. Throughout the paper, we use δ = -γ , which makes G a mean 0 distribution. 9 Under this setup, we consider the optimal stationary policy and its corresponding value

8 This reward form is often referred to as additive and conditionally independent form.

9 All the results of this paper easily generalize to other values of δ .

function defined as

<!-- formula-not-decoded -->

Throughout, we make the following assumption on agent's decisions.

Assumption 2. When interacting with the MDP ( S , A , P, ν 0 , r, β ) , agent follows the optimal stationary policy π ∗ .

According to Assumption 2, the value function V ∗ must satisfy the recursive relationship, often called the Bellman equation, as follows:

<!-- formula-not-decoded -->

where the second equality is from Lemma 12. We further define the Q ∗ function as

<!-- formula-not-decoded -->

We can show that (see Web Appendix § B.4):

<!-- formula-not-decoded -->

## 3.3 DDC - MaxEnt-IRL Equivalence and unified problem statement

̸

The Bellman equations of MaxEnt-IRL with λ = 1 (Equation (2)) and DDC with δ = -γ (Equation (3)) are equivalent. Consequently, the optimal Q ∗ values obtained from solving these Bellman equations are the same for both MaxEnt-IRL and DDC. Furthermore, the optimal policy induced by Q ∗ is identical in both frameworks. Therefore, we can infer that solving one problem is equivalent to solving the other. Throughout, all the discussions we make for λ = 1 in MaxEnt-IRL and δ = -γ in DDC extend directly to any λ = 1 and δ = -γ , respectively. This equivalence is a folk theorem that was first observed in (Ermon et al. 2015).

In both settings, the goal is to recover the underlying reward function r that explains an agent's demonstrated behavior. Given the equivalence between them, we can now formulate a unified problem statement

̸

that encompasses both Offline Maximum Entropy Inverse Reinforcement Learning (Offline MaxEnt-IRL) and the Dynamic Discrete Choice (DDC) model estimation.

To formalize this, consider a dataset consisting of state-action-next state sequences collected from an agent's behavior: D := (( s 0 , a 0 , s ′ 0 ) , ( s 1 , a 1 , s ′ 1 ) , . . . , ( s N , a N , s ′ N )) . Following Assumption 1, we assume that the data was generated by the agent playing the optimal policy π ∗ when interacting with the MDP ( S , A , P, ν 0 , r, β ) .

Definition 1 (The unified problem statement) . The objective of offline MaxEnt-IRL and DDC can be defined as learning a function ˆ r ∈ R ⊆ R ¯ S×A that minimizes the mean squared prediction error with respect to data distribution (i.e., expert policy's state-action distribution) from offline data D such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

defines the expert policy's coverage, which consists of states that are reachable with nonzero probability under the expert's optimal policy π ∗ . 10

Restricting to ¯ S is essential, as the dataset D only contains information about states visited under π ∗ . Inferring rewards beyond this set would be ill-posed due to a lack of data, making ¯ S the natural domain for learning. Similarly, Computing MSE using the expert policy's state-action distribution is natural since the goal is to recover the reward function that explains the expert's behavior.

Remark (Counterfactual policy optimization without estimating P ). Given logged interaction data D = { ( s i , a i , s ′ i ) } N i =1 and a recovered reward function ˆ r , suppose the analyst wishes to evaluate the effect of a counterfactual intervention on the reward-e.g., modifying incentives or preferences-which induces a new estimated reward function ˆ r cf ( s, a ) , such as ˆ r cf ( s, a ) = ˆ r ( s, a ) + ∆( s, a ) , where ∆ encodes the intervention. Using the augmented dataset

<!-- formula-not-decoded -->

any standard modern offline-RL algorithm (e.g., Conservative Q-Learning (Kumar et al. 2020)) can be applied to ˆ D cf to obtain the counterfactual optimal policy without requiring an explicit estimate of the transition function P . These methods operate via empirical Bellman backups over observed transitions and are fully model-free.

In contrast, when the counterfactual intervention alters the P itself, one must estimate P , model the intervention accordingly, and perform model-based planning (e.g., (Sutton and Barto 2018)).

10 For every s ∈ ¯ S , every action a ∈ A occurs with probability strictly greater than zero, ensuring that the data sufficiently covers the relevant decision-making space.

where

## 3.4 Identification

As we defined in Definition 1, our goal is to learn the agent's reward function r ( s, a ) given offline data D . However, without additional assumptions on the reward structure, this problem is ill-defined because many reward functions can explain the optimal policy (Fu et al. 2017; Ng et al. 1999). To address this issue, following the DDC literature (Rust 1994; Magnac and Thesmar 2002; Hotz and Miller 1993) and recent IRL literature (Geng et al. 2020), we assume that there is an anchor action a s in each state s , such that the reward for each of state-anchor action combination is known.

Assumption 3. For all s ∈ S , there exists an action a s ∈ A such that r ( s, a s ) is known.

Note that the optimal policy remains the same irrespective of the choice of the anchor action a s and the reward value at the anchor action r ( s, a s ) (at any given s ). As such, Assumption 3 only helps with identification and does not materially affect the estimation procedure. That is, for the sake of choosing an arbitrary reward function that is compatible with the optimal policy (i.e., the IRL objective), an arbitrary choice of a s and the r ( s, a s ) value is justified.

In Lemma 1, we formally establish that Assumptions 1 and 3 uniquely identify Q ∗ and r . See Web Appendix § C.2 for the proof.

Lemma 1 (Magnac and Thesmar (2002)) . Given discount factor β , transition kernel P ∈ ∆ S×A S and optimal policy π ∗ ∈ ∆ S A , under Assumptions 1 and 3, the solution to the following system of equations:

<!-- formula-not-decoded -->

identifies Q ∗ up to s ∈ S , a ∈ A . Furthermore, r is obtained up to ∀ s ∈ S , a ∈ A by solving:

<!-- formula-not-decoded -->

for all s ∈ S , a ∈ A .

In the first part of the theorem, we show that, after constraining the reward functions for anchor actions, we can recover the unique Q ∗ -function for the optimal policy from the observed choices and the Bellman equation for the anchor-action (written in terms of log-sum-exp of Q -values). The second step follows naturally, where we can show that reward functions are then uniquely recovered from Q ∗ -functions using the Bellman equation.

## 3.5 Bellman error and Temporal difference (TD) error

There are two key concepts used for describing a gradient-based algorithm for IRL/DDC: the Bellman error and the Temporal Difference (TD) error. In this section, we define each of them and discuss their relationship.

We start by defining Q = { Q : S × A → R | ∥ Q ∥ ∞ &lt; ∞} . By Rust (1994), β &lt; 1 (discount factor less than one) implies Q ∗ ∈ Q . Next, we define the Bellman operator as T : Q ↦→ Q as follows:

<!-- formula-not-decoded -->

According to the Bellman equation shown in Equation (2), Q ∗ satisfies T Q ∗ ( s, a ) -Q ∗ ( s, a ) = 0 ; in fact, Q ∗ is the unique solution to T Q ( s, a ) -Q ( s, a ) = 0 ; see (Rust 1994). Based on this observation, we define the following notions of error.

Definition 2. We define the Bellman error for Q ∈ Q at ( s, a ) as T Q ( s, a ) -Q ( s, a ) . Furthermore, we define the Squared Bellman error and the Expected squared Bellman error as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In practice, we don't have direct access to T unless we know (or have a consistent estimate of) the transition kernel P ∈ ∆ S×A S . Instead, we can compute an empirical Sampled Bellman operator ˆ T , defined as

<!-- formula-not-decoded -->

Definition 3. We define Temporal-Difference (TD) error for Q at the transition ( s, a, s ′ ) , Squared TD error, and Expected squared TD error as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 2 states the relationship between the TD error terms and Bellman error terms.

Lemma 2 (Expectation of TD error is equivalent to BE error) .

<!-- formula-not-decoded -->

## 4 ERM-IRL (ERM-DDC) framework

## 4.1 Identification via expected risk minimization

Given Lemma 1, we would like to find the unique ˆ Q that satisfies

<!-- formula-not-decoded -->

where ¯ S (the reachable states from ν 0 , π ∗ ) was defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Therefore equations (5) becomes

<!-- formula-not-decoded -->

We now propose a one-shot Empirical Risk Minimization framework (ERM-IRL/ERM-DDC) to solve the Equation (6). First, we recast the IRL problem as the following expected risk minimization problem under an infinite data regime.

Definition 4 (Expected risk minimization problem) . The expected risk minimization problem is defined as the problem of finding Q that minimizes the expected risk R exp ( Q ) , which is defined as

<!-- formula-not-decoded -->

Now note that:

where a s is defined in Assumption 3.

Remark. The joint minimization of the Negative Log Likelihood (NLL) term and Bellman Error (BE) term is the key novelty in our approach. Prior work on the IRL and DDC literature (Hotz and Miller 1993; Zeng et al. 2023) typically minimizes the log-likelihood of the observed choice probabilities (the NLL term), given observed or estimated state transition probabilities. The standard solution is to first estimate/assume state transition probabilities, then obtain estimates of future value functions, plug them into the choice probability, and then minimize the NLL term. In contrast, our recast problem avoids the estimation of state-transition probabilities and instead jointly minimizes the NLL term along with the Bellman error term. This is particularly helpful in large state spaces since the estimation of state-transition probabilities can be infeasible/costly in such settings. In Theorem 3, we show that the solution to our recast problem in Equation (7) identifies the reward function.

Theorem 3 (Identification via expected risk minimization) .

The solution to the expected risk minimization problem (Equation (7) ) uniquely identifies Q ∗ up to s ∈ ¯ S and a ∈ A , i.e., finds ̂ Q that satisfies ̂ Q ( s, a ) = Q ∗ ( s, a ) for s ∈ ¯ S and a ∈ A . Furthermore, we can uniquely identify r up to s ∈ ¯ S and a ∈ A by r ( s, a ) = ̂ Q ( s, a ) -β · E s ′ ∼ P ( s,a ) [ V ̂ Q ] .

Proof of Theorem 3. Define ̂ Q as the solution to the expected risk minimization problem

<!-- formula-not-decoded -->

Note that Equation (7) minimizes the sum of two terms that are jointly minimized in Equation (6). Since the solution set to Equation (6) is nonempty by Lemma 1, by Lemma 21, ̂ Q minimizes Equation (6). Again, by Lemma 1, this implies that ̂ Q is equivalent to Q ∗ .

Essentially, Theorem 3 ensures that solving Equation (7) gives the exact r and Q ∗ up to ¯ S and thus provides the solution to the DDC/IRL problem defined in Definition 1.

## Remark. Comparison with Imitation Learning

Having established the identification guarantees for the ERM-IRL/DDC framework, it is natural to compare this formulation to the identification properties of Imitation Learning (IL) (Torabi et al. 2018; Rajaraman et al. 2020; Foster et al. 2024). Unlike IRL, which seeks to infer the underlying reward function that explains expert behavior, IL directly aims to recover the expert policy without modeling the transition dynamics. The objective of imitation learning is often defined as finding a policy ˆ p with min ˆ p E ( s,a ) ∼ π ∗ ,ν 0 [ ℓ (ˆ p ( a | s ) , π ∗ ( a | s ))] , ℓ is the cross-entropy loss or equivalently,

<!-- formula-not-decoded -->

Equation (8) is exactly what a typical Behavioral Cloning (BC) (Torabi et al. 2018) minimizes under entropy regularization, as the objective of BC is

<!-- formula-not-decoded -->

where ˆ p Q ( a | s ) = Q ( s,a ) ∑ ˜ a ∈A Q ( s, ˜ a ) . Note that the solution set of Equation (9) fully contains the solution set of Equation (6), which identifies the Q ∗ for offline IRL/DDC. This means that any solution to the offline IRL/DDC problem also minimizes the imitation learning objective, but not necessarily vice versa. Consequently, under entropy regularization, the IL objective is fundamentally easier to solve than the offline IRL/DDC problem, as it only requires minimizing the negative log-likelihood term without enforcing Bellman consistency. One of the key contributions of this paper is to formally establish and clarify this distinction: IL operates within a strictly simpler optimization landscape than the offline IRL/DDC, making it a computationally and statistically more tractable problem. This distinction further underscores the advantage of Behavioral Cloning (BC) over ERM-IRL/DDC for imitation learning (IL) tasks--since BC does not require modeling transition dynamics or solving an optimization problem involving the Bellman residual, it benefits from significantly lower computational and statistical complexity, making it a more efficient approach for IL.

While behavioral cloning (BC) is often sufficient for IL (Foster et al. 2024), i.e., reproducing the expert's actions under the same dynamics and incentive structure, it fundamentally lacks the ingredients needed for counterfactual reasoning. Because BC learns a direct mapping s ↦→ a without ever inferring the latent reward r ( s, a ) , it has no principled way to predict what an expert would do if (i) the transition kernel P were perturbed (e.g., a new recommendation algorithm) or (ii) the reward landscape itself were altered (e.g., a firm introduces monetary incentives). In such scenarios the state-action pairs generated by the expert no longer follow the original occupancy measure, so a cloned policy is forced to extrapolate outside its training distribution-precisely where imitation learning is known to fail.

Recovering the reward resolves this limitation. Once we have a consistent estimate ˆ r (or, equivalently, ˆ Q ∗ ), we can decouple policy evaluation from the historical data: any hypothetical change to P or the reward can be encoded and handed to a standard offline RL, which recomputes the optimal policy for the new MDP without further demonstrations. In other words, rewards serve as a portable, mechanism-level summary of preferences that supports robust counterfactual simulation, policy optimization, and welfare analysis-capabilities that pure imitation methods cannot provide.

## 4.2 Estimation via minimax-formulated empirical risk minimization

While the idea of expected risk minimization - minimizing Equation (7) - is straightforward, empirically approximating L BE ( Q )( s, a ) = ( T Q ( s, a ) -Q ( s, a )) 2 and its gradient is quite challenging. As discussed in Section 3.5, T Q is not available unless we know the transition function. As a result, we have to rely on an estimate of T . A natural choice, common in TD-methods, is ˆ T Q ( s, a, s ′ ) = r ( s, a ) + β · V Q ( s ′ ) which is

computable given Q and data D . Thus, a natural proxy objective to minimize is:

<!-- formula-not-decoded -->

Temporal Difference (TD) methods typically use stochastic approximation to obtain an estimate of this proxy objective (Tesauro et al. 1995; Adusumilli and Eckardt 2019). However, the issue with TD methods is that minimizing the proxy objective will not minimize the Bellman error in general (see Web Appendix § C.1 for details), because of the extra variance term, as shown below.

<!-- formula-not-decoded -->

̸

As defined, ˆ T is a one-step estimator, and the second term in the above equation does not vanish even in infinite data regimes. So, simply using the TD approach to approximate squared Bellman error provides a biased estimate. Intuitively, this problem happens because expectation and square are not exchangeable, i.e., E s ′ ∼ P ( s,a ) [ δ Q ( s, a, s ′ ) | s, a ] 2 = E s ′ ∼ P ( s,a ) [ δ Q ( s, a, s ′ ) 2 | s, a ] . This bias in the TD approach is a well-known problem called the double sampling problem (Munos 2003; Lagoudakis and Parr 2003; Sutton and Barto 2018; Jiang and Xie 2024). To remove this problematic square term, we employ an approach often referred to as the 'Bi-Conjugate Trick' (Antos et al. 2008; Dai et al. 2018; Patterson et al. 2022) which replaces a square function with a linear function called the bi-conjugate:

<!-- formula-not-decoded -->

By further re-parametrizing h using ζ = h -r ( s, a ) + Q ( s, a ) , after some algebra, we arrive at Lemma 4. See Web Appendix § C.1 for the detailed derivation.

## Lemma 4.

(a) We can express the squared Bellman error as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(b) Define the minimizer (over all states and actions) of objective (11) as

<!-- formula-not-decoded -->

then r ( s, a ) = Q ∗ ( s, a ) -βζ ∗ ( s, a ) .

The reformulation of L BE proposed in Lemma 4 enjoys the advantage of minimizing the squared TD-error ( L TD ) but without bias. Combining Theorem 3 and Lemma 4, we arrive at the following Theorem 5, which gives the expected risk minimization formulation of IRL we propose.

## Theorem 5. Q ∗ is uniquely identified by

<!-- formula-not-decoded -->

Furthermore, r ( s, a ) = Q ∗ ( s, a ) -βζ ∗ ( s, a ) where ζ ∗ is defined in Lemma 4.

Equation (12) in Theorem 5 is a mini-max problem in terms of Q ∈ Q and the introduced dual function ζ ∈ R S × A . To summarize, term 1) is the negative log-likelihood equation, term 2) is the TD error, and term 3) introduces a dual function ζ . The introduction of the dual function ζ in term 3) may seem a bit strange. In particular, note that arg max ζ ∈ R -E s ′ ∼ P ( s,a ) [ ( V Q ( s ′ ) -ζ ) 2 | s, a ] is just ζ = E s ′ ∼ P ( s,a ) [ V ( s ′ ) | s, a ] . However, we do not have access to the transition kernel and the state and action spaces may be large. Instead, we think of ζ as a function of states and actions, ζ ( s, a ) as introduced in Lemma 4. This parametrization allows us to optimize over a class of functions containing ζ ( s, a ) directly.

Given that the minimax resolution for the expected risk minimization problem in Theorem 5 finds Q ∗ under an infinite number of data , we now discuss the case when we are only given a finite dataset D instead. For this, let's first rewrite the Equation (12) as

<!-- formula-not-decoded -->

where ˜ ζ ∈ argmin ζ ∈ R S × A E ( s,a ) ∼ π ∗ ,ν 0 ,s ′ ∼ P ( s,a ) [ ✶ a = a s { ( V Q ( s ′ ) -ζ ( s, a )) 2 }] . But such ˜ ζ also satisfies ˜ ζ ∈ argmin ζ ∈ R S × A E ( s,a ) ∼ π ∗ ,ν 0 ,s ′ ∼ P ( s,a ) [ ( V Q ( s ′ ) -ζ ( s, a )) 2 ] , because

<!-- formula-not-decoded -->

Substituting Equation (13) and Equation (14)'s expectation over the data distribution with the empirical mean over the data, we arrive at the empirical risk minimization formulation defined in Definition 5.

Definition 5 (Empirical risk minimization) . Given N := |D| where D is a finite dataset. An empirical risk minimization problem is defined as the problem of finding Q that minimizes the empirical risk R emp ( Q ; D ) , which is defined as

<!-- formula-not-decoded -->

where ¯ ζ := argmin ζ ∈ R S × A 1 N ∑ ( s,a,s ′ ) ∈D [ ( V Q ( s ′ ) -ζ ( s, a )) 2 ]

As formulated in Equation (15), our empirical risk minimization objective is a minimax optimization problem that involves inner maximization over ζ and outer minimization over Q . This structure is a direct consequence of introducing the dual function ζ to develop an unbiased estimator, thereby circumventing the double sampling problem discussed previously. In the next section, we introduce GLADIUS , a practical, alternating gradient ascent-descent algorithm specifically designed to solve this minimax problem and learn the underlying reward function from the data.

## 5 GLADIUS: Algorithm for ERM-IRL (ERM-DDC)

Algorithm 1 solves the empirical risk minimization problem in Definition 5 through an alternating gradient ascent descent algorithm we call Gradient-based Learning with Ascent-Descent for Inverse Utility learning from Samples (GLADIUS). Given the function class Q of value functions, let Q θ 2 ∈ Q and ζ θ 1 ∈ R S × A denote the functional representation of Q and ζ . Our goal is to learn the parameters θ ∗ = { θ ∗ 1 , θ ∗ 2 } , that together characterize ˆ Q and ˆ ζ . Each iteration in the GLADIUS algorithm consists of the following two steps:

1. Gradient Ascent: For sampled batch data B 1 ⊆ D , take a gradient step for θ 1 while fixing Q θ 2 .
2. Gradient Descent: For sampled batch data B 2 ⊆ D , take a gradient step for θ 2 while fixing ζ θ 1 .

Note that D can be used instead of using batches B 1 and B 2 ; the usage of batches is to keep the computational/memory complexity O ( | B 1 = B 2 | ) instead of O ( |D| ) .

After a fixed number of gradient steps of Q θ 1 and ζ θ 2 (which we can denote as ˆ Q and ˆ ζ ), we can compute the reward prediction ˆ r as ˆ r ( s, a ) = ˆ Q ( s, a ) -β ˆ ζ ( s, a ) due to Theorem 5.

## Special Case: Deterministic Transitions

When the transition function is deterministic (e.g., in Rafailov et al. (2024); Guo et al. (2025); Zhong et al. (2024)) meaning that for any state-action pair ( s, a ) , the next state s ′ is uniquely determined, the ascent step involving ζ is no longer required. This is because the term ( V Q ( s ′ ) -ζ ( s, a )) 2 (highlighted in orange in Equation (12) and (15)) becomes redundant in the empirical ERM-IRL objective, because max ζ ∈ R -E s ′ ∼ P ( s,a ) [ ( V Q ( s ′ ) -ζ ) 2 | s, a ] is always 0 . Consequently, the optimization simplifies to:

## Algorithm 1 G radient-based L earning with A scentD escent for I nverse U tility learning from S amples (GLADIUS)

Require:

Offline dataset D = { ( s, a, s ) } , time horizon T

′

Ensure: ̂ r , ̂ Q

- 1: Initialize Q θ 2 , ζ θ 1 , iteration ← 1
- 2: while t ≤ T do
- 3: Draw batches B 1 , B 2 from D
- 4: [Ascent Step: Update ζ θ 1 , fixing Q θ 2 and V θ 2 ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 7: θ 1 ← θ 1 -τ 1 ∇ θ 1 D θ 1
- 8: [Descent Step: Update Q θ 2 and V θ 2 , fixing ζ θ 1 ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 14: iteration ← iteration +1
- 15: end while

<!-- formula-not-decoded -->

- 17: ̂ Q ← Q θ 2

<!-- formula-not-decoded -->

## Algorithm 2 GLADIUS under Deterministic Transitions

Require: Offline dataset D = { ( s, a, s ′ ) } , time horizon T

Ensure: r ,

- 1: Initialize Q θ , iteration ← 1

̂ ̂ Q

- 2: while t ≤ T do
- 3: Draw batch B from D

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 8: θ ← θ -τ ∇ θ L θ
- 9: iteration ← iteration +1
- 10: end while

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Under deterministic transitions, the GLADIUS algorithm reduces to gradient descents for Q θ , eliminating the need for the alternating ascent-descent update steps. Consequently, the estimated reward function is computed as:

<!-- formula-not-decoded -->

Key Differences in the Deterministic Case:

- No Ascent Step: The ascent step for ζ is removed since the term ( V Q ( s ′ ) -ζ ( s, a )) 2 disappears.
- Gradient Descent: The algorithm updates Q θ via a single gradient descent step per iteration.
- Reward Computation: The reward function is computed as ˆ r ( s, a ) = ˆ Q ( s, a ) -βV Q ( s ′ ) .

This modification makes GLADIUS computationally more efficient when applied to deterministic environments while maintaining the correct theoretical formulation of the Q ∗ and reward functions.

## 6 Theory and analysis of GLADIUS

As discussed in the previous section, Equation (15) represents a mini-max optimization problem. Such problems are known to be globally solvable by a simple gradient ascent-descent algorithm when it is a concaveconvex mini-max problem. However, the challenge is that Equation (12) is not a concave-convex mini-max problem. Given Q , it determines ζ that serves as the Bayes-optimal estimator for E s ′ ∼ P ( s,a ) [ V Q ( s ′ ) | s, a ] . This implies that -E s ′ ∼ P ( s,a ) [ ( V Q ( s ′ ) -ζ ) 2 | s, a ] is strongly concave in ζ . On the other hand, given such an optimal ζ , L BE ( Q )( s, a ) term is not convex in Q (Bas-Serrano et al. 2021). The key result in this section is proving that both L BE ( Q )( s, a ) and L NLL ( Q )( s, a ) = [ -log (ˆ p Q ( a | s ))] , under certain parametrization of Q , satisfies the Polyak-Łojasiewicz (PL) condition, which allows Algorithm 1 to converge to global optima.

## 6.1 Polyak-Łojasiewicz (PL) in terms of parameter θ

Given the state space S with dim( S ) &lt; ∞ and the finite action space A with |A| &lt; ∞ , consider a set of parametrized functions

<!-- formula-not-decoded -->

where F denotes a class of functions such as linear, polynomial, or deep neural network function class that is parametrized by θ . We make the following assumption, often called the realizability assumption.

Assumption 4 (Realizability) . Q contains an optimal function Q ∗ , meaning there exists θ ∗ ∈ Θ such that Q θ ∗ = Q ∗ .

Assumption 4 - an assumption that is often considered minimal in the literature (Chen and Jiang 2019; Xie and Jiang 2021; Zhan et al. 2022; Zanette 2023; Jiang and Xie 2024) - is fairly mild in modern practice. For instance, a fully connected two-layer ReLU network with sufficient width is a universal approximator (Cybenko 1989; Hornik 1991), capable of approximating any function class to arbitrary accuracy. Although such expressive models once raised concerns about overfitting via the bias-variance trade-off, contemporary deep-learning practice routinely embraces heavily over-parameterized architectures to address this challenge (Zhang et al. 2016; Allen-Zhu et al. 2019; Dar et al. 2021). Moreover, model size-the primary determinant of expressivity-is seldom a binding constraint: commodity GPUs comfortably host networks far larger than those typically required for unstructured social-science problems, and even large language models can now run on a single desktop workstation (OLMo et al. 2024).

Under this parametrization, the expected risk (the Equation (7)) becomes

<!-- formula-not-decoded -->

and the empirical risk (the Equation (15)) becomes

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The main question in this section is under which parametrization the Equation (17) and (18) satisfy the Polyak-Łojasiewicz (PL) condition in terms of θ , which is defined as follows.

<!-- formula-not-decoded -->

Definition 6 (Polyak-Łojasiewicz (PL) condition with respect to ℓ 2 norm) . Given Θ ∈ R d θ , a function g : Θ ↦→ R is said to satisfy the Polyak-Łojasiewicz (PL) condition with respect to ℓ 2 norm if g has a nonempty solution set and a finite minimal value g ( θ ∗ ) for θ ∗ ∈ Θ ⊆ R d , and there exists some c &gt; 0 such that 1 2 ∥∇ g ( θ ) ∥ 2 2 ≥ c ( g ( θ ) -g ( θ ∗ )) , ∀ θ ∈ Θ .

Here, we consider the case when the parametrized function class Q satisfies the Assumption 5.

Assumption 5. Suppose that the state space S is compact and the action set A is finite.

1. The state-action value map

is in C ( S × A ) . Moreover,

<!-- formula-not-decoded -->

2. For every θ in this ball, the Jacobian operator DQ θ : R d θ → C ( S × A ) defined by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

satisfies

<!-- formula-not-decoded -->

where the norm on C ( S × A ) is the sup-norm.

Here, note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

How strong is the Assumption 5? The following two lemmas show that Assumption 5 is satisfied for popular function classes such as linear and the neural network function class.

Lemma 6. Let Q θ ( s, a ) = θ ⊤ ϕ ( s, a ) , where the known feature mapping ϕ : R dim( S )+dim( A ) ↦→ R d θ , satisfies ∥ ϕ ( s, a ) ∥ ≤ C almost surely with respect to ( s, a ) ∼ ( π ∗ , ν 0 ) for some C &gt; 0 . Then dataset size |D| ≥ Cd θ implies that Assumption 5 holds with probability at least 1 -e -C |D| .

Lemma 7 (Liu et al. (2022); Sharifi and Fazlyab (2024)) . Let Q θ be a deep nonlinear neural network composed of smooth activation functions (e.g., sigmoid, Exponential Linear Unit (ELU) (Clevert et al. 2015)) and linear layers parameterized by θ , which was initialized randomly (e.g., Gaussian or orthogonal) as θ 0 . Then there exist a network depth L &lt; ∞ and a width m&lt; ∞ , such that with high probability, the network Q θ satisfies Assumption 5 in the entire ball B that contains both θ 0 and θ ∗ .

Under Assumption 4 and 5, we can prove that both L NLL ( Q θ )( s, a ) and L BE ( Q θ )( s, a ) satisfy PL condition in terms of θ . Furthermore, L BE ( Q θ ) and L NLL ( Q θ ) also satisfy the PL condition.

Lemma 8. For any given s ∈ S and a ∈ A , L BE ( Q θ )( s, a ) satisfies PL condition with respect to θ . Furthermore, L BE ( Q θ ) satisfies the PL condition with respect to θ in terms of ℓ 2 norm.

Lemma 9. For any given s ∈ S and a ∈ A , L NLL ( Q θ )( s, a ) satisfies PL condition with respect to θ . Furthermore, L NLL ( Q θ ) satisfies the PL condition with respect to θ in terms of ℓ 2 norm.

We now present Theorem 10, the main result in this section, which will be useful in Section 6.2 for showing Algorithm 1's global convergence. Note that Equation 17 defines the expected risk through Q θ , whereas Equation 18 defines the empirical risk through ζ θ 1 and Q θ 2 .

Theorem 10. Under Assumption 4 and 5, the expected risk (Equation (17) ) satisfies PL condition in terms of θ . Also, the empirical risk (Equation (18) ) satisfy the PL condition in terms of both θ 1 and θ 2 .

## 6.2 Global convergence of GLADIUS

Denote D N be a finite-size dataset with N number of sampled transition pairs (( s, a, s ′ )) . Let Θ ∗ denote the (possibly non-singleton) set of expected risk (Equation (17)) minimizers. Let ˆ θ T be the parameters returned by Algorithm 1 after T SGDA iterations with step-sizes η t = c 1 c 2 + T .

Theorem 11 (Global convergence) . Under Assumption 4 and Assumption 5,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where dist( θ , Θ ∗ ) := min θ ′ ∈ Θ ∗ ∥ θ -θ ′ ∥ 2 , B is from Assumption 5 and the values of constants c and C ′ are specified in the Web Appendix § C.8 where we prove Theorem 11.

Remark. To the best of our knowledge, no prior work has proposed an algorithm that guarantees global optimum convergence of the minimization problem that involves L BE ( Q )( s, a ) term. 11 . In this regard, Lemma 8 has an important implication for Offline reinforcement learning (Offline RL) (Jiang and Xie 2024) Gradient-based Offline reinforcement learning (Antos et al. 2008; Dai et al. 2018), which minimizes L BE ( Q θ ) in the same way as GLADIUS does, has been proven to be convergent. However, its global convergence guarantee has not yet been established. Lemma 8, combined with a suitable distribution assumption, may be used to establish that gradient-based Offline RL is indeed globally convergent for important function classes such as tabular, linear, polynomial, and neural network function classes.

## 7 Offline IRL/DDC experiments

We now present results from simulation experiments, comparing the performance of our approach with that of several benchmark algorithms. In this section, we use the high-dimensional version of the canonical bus engine replacement problem (Rust (1994)) as the setting for our experiments. Later, in § 8, we use the OpenAI gym benchmark environment experiments with a discrete action space (Lunar Lander, Acrobot, and Cartpole) Brockman (2016) as in Garg et al. (2021) for the related but easier problem of imitation learning.

## 7.1 Experimental Setup

This bus engine setting has been extensively used as the standard benchmark for the reward learning problem in the DDC literature in economics (Hotz and Miller 1993; Aguirregabiria and Mira 2002; Kasahara and Shimotsu 2009; Arcidiacono and Miller 2011; Arcidiacono and Ellickson 2011; Su and Judd 2012; Norets 2009; Chiong et al. 2016; Reich 2018; Chernozhukov et al. 2022; Geng et al. 2023; Barzegary and Yoganarasimhan 2022; Yang 2024).

11 Some studies, such as Dai et al. (2018), have demonstrated convergence to a stationary point of this mini-max problem.

The bus engine replacement problem (Rust 1987) is a simple regenerative optimal stopping problem. In this setting, the manager of a bus company operates many identical buses. As a bus accumulates mileage, its per-period maintenance cost increases. The manager can replace the engine in any period (which then becomes as good as new, and this replacement decision resets the mileage to one). However, the replacement decision comes with a high fixed cost. Each period, the manager makes a dynamic trade-off between replacing the engine and continuing with maintenance. We observe the manager's decisions for a fixed set of buses, i.e., a series of states, decisions, and state transitions. Our goal is to learn the manager's reward function from these observed trajectories under the assumption that he made these decisions optimally.

Dataset. There are N independent and identical buses (trajectories) indexed by j , each of which has 100 periods over which we observe them, i.e., h ∈ { 1 . . . 100 } . Each bus's trajectory starts with an initial mileage of 1. The only reward-relevant state variable at period h is the mileage of bus x jh ∈ { 1 , 2 , . . . 20 } .

Decisions and rewards. There are two possible decisions at each period, replacement or continuation, denoted by d jh = { 0 , 1 } . d jh = 1 denotes replacement, and there is a fixed cost θ 1 of replacement. Replacement resets the mileage to 1, i.e., the engine is as good as new. d jh = 0 denotes maintenance, and the cost of maintaining the engine depends on the mileage as follows: θ 0 x jh . Intuitively, the manager can pay a high fixed cost θ 1 for replacing an engine in this period but doing so reduces future maintenance costs since the mileage is reset to 1. In all our experiments, we set θ 0 = 1 (maintenance cost) and θ 1 = 5 (replacement cost). Additionally, we set the discount factor to β = 0 . 95 .

State transitions at each period. If the manager chooses maintenance, the mileage advances by 1, 2, 3, or 4 with a 1 / 4 probability each. If the manager chooses to replace the engine, then the mileage is reset to 1. That is, P ( { x j ( h +1) = x jh + k } | d jh = 0) = 1 / 4 , k ∈ { 1 , 2 , 3 , 4 } and P { x j ( h +1) = 1 | d jh = 1 } = 1 . When the bus reaches the maximum mileage of 20, we assume that mileage remains at 20 even if the manager continues to choose maintenance.

High-dimensional setup. In some simulations, we consider a high-dimensional version of the problem, where we now modify the basic set-up described above to include a set of K high-dimensional state variables, similar to Geng et al. (2023). Assume that we have access to an additional set of K state variables { s 1 jh , s 2 jh , s 3 jh . . . s K jh } , where each s k jh is an i.i.d random draw from {-10 , -9 , . . . , 9 , 10 } . We vary K from 2 to 100 in our empirical experiments to test the sensitivity of our approach to the dimensionality of the problem. Further, we assume that these high-dimensional state variables s k jh s do not affect the reward function or the mileage transition probabilities. However, the researcher does not know this. So, they are included in the state space, and ideally, our algorithm should be able to infer that these state variables do not affect rewards and/or value function and recover the true reward function.

Traing/testing split. Throughout, we keep 80% of the trajectories in any experiment for training/learning the reward function, and the remaining 20% is used for evaluation/testing.

Functional form. For the oracle methods, we use the reward functions' true parametric form, i.e., a linear function. For other methods (including ours), we used a multi-layer perceptron (MLP) with two hidden layers and 10 perceptrons for each hidden layer for the estimation of Q -function.

## 7.2 Benchmark Algorithms

We compare our algorithm against a series of standard, or state-of-art benchmark algorithms in the DDC and IRL settings.

Rust (Oracle) Rust is an oracle-like fixed point iteration baseline that uses the nested fixed point algorithm (Rust 1987). It assumes the knowledge of: (1) linear parametrization of rewards by θ 1 and θ 2 as described above, and (2) the exact transition probabilities.

ML-IRL (Oracle) ML-IRL from Zeng et al. (2023) is the state-of-the-art offline IRL algorithm that minimizes negative log-likelihood of choice (i.e., the first term in Equation (7)). This method requires a separate estimation of transition probabilities, which is challenging in high-dimensional settings. So, we make the same oracle assumptions as we did for Rust (Oracle), i.e., assume that transition probabilities are known. Additionally, to further improve this method, we leverage the finite dependence property of the problem (Arcidiacono and Miller 2011), which helps avoid roll-outs.

SAmQ SAmQ Geng et al. (2023) fits approximated soft-max Value Iteration (VI) to the observed data. We use the SAmQ implementation provided by the authors 12 . However, their code did not scale due to a memory overflow issue, and did not work for scenarios with 2500 (i.e., 250,000 samples) trajectories or more.

IQ-learn IQ-learn is a popular gradient-based method, maximizing the occupancy matching objective (which does not guarantee that the Bellman equation is satisfied; see Web Appendix S B.6) for details.

BC Behavioral Cloning (BC) simply minimizes the expected negative log-likelihood. This simple algorithm outperforms Zeng et al. (2023); Ziniu et al. (2022) many recent algorithms such as ValueDICE Kostrikov et al. (2019). See § 4 for detailed discussions.

## 7.3 Experiment results

## 7.3.1 Performance results for the standard bus engine setting

Table 2 provides a table of simulation experiment results without dummy variables, i.e., with only mileage ( x jh ) as the relevant state variable. The performance of algorithms was compared in terms of mean absolute percentage error (MAPE) of r estimation, which is defined as 1 N ∑ N i =1 ∣ ∣ ∣ ˆ r i -r i r i ∣ ∣ ∣ × 100 , where N is the total number of samples from expert policy π ∗ and ˆ r i is each algorithm's estimator for the true reward r i . 1314

We find that GLADIUS performs much better than non-Oracle baselines and performs at least on par with, or slightly better than, Oracle baselines. A natural question here is: why do the Oracle baselines that leverage the exact transition function and the precise linear parametrization not beat our approach? The main reason for this is the imbalance of state-action distribution from expert policy: (See Table 3 and Web Appendix § A.1)

1. All trajectories start from mileage 1. In addition, the replacement action (action 0) resets the mileage to 1. Therefore, most states observed in the expert data are within mileage 1-5. When data collection policy

12 https://github.com/gengsinong/SAmQ

13 In the simulation we consider, we don't have a state-action pair with true reward near 0.

14 As we assume that the data was collected from agents following (entropy regularized) optimal policy π ∗ (Assumption 1), the distribution of states and actions in the data is the best data distribution choice.

| No. of Tra- jectories (H=100)   | Oracle Baselines   | Oracle Baselines   | Neural Network, No Knowledge of Transition Probabilities   | Neural Network, No Knowledge of Transition Probabilities   | Neural Network, No Knowledge of Transition Probabilities   | Neural Network, No Knowledge of Transition Probabilities   |
|---------------------------------|--------------------|--------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------|
| No. of Tra- jectories (H=100)   | Rust               | ML-IRL             | GLADIUS                                                    | SAmQ                                                       | IQ-learn                                                   | BC                                                         |
| No. of Tra- jectories (H=100)   | MAPE (SE)          | MAPE (SE)          | MAPE (SE)                                                  | MAPE (SE)                                                  | MAPE (SE)                                                  | MAPE (SE)                                                  |
| 50                              | 3.62 (1.70)        | 3.62 (1.74)        | 3.44 (1.28)                                                | 4.92 (1.20)                                                | 114.13 (26.60)                                             | 80.55 (12.82)                                              |
| 250                             | 1.37 (0.77)        | 1.10 (0.78)        | 0.84 (0.51)                                                | 3.65 (1.00)                                                | 112.86 (27.31)                                             | 72.04 (13.21)                                              |
| 500                             | 0.90 (0.56)        | 0.84 (0.59)        | 0.55 (0.20)                                                | 3.13 (0.86)                                                | 113.27 (25.54)                                             | 71.92 (12.44)                                              |
| 1000                            | 0.71 (0.49)        | 0.64 (0.48)        | 0.52 (0.22)                                                | 1.55 (0.46)                                                | 112.98 (24.12)                                             | 72.17 (12.11)                                              |
| 2500                            | 0.68 (0.22)        | 0.62 (0.35)        | 0.13 (0.06)                                                | N/A                                                        | 111.77 (23.99)                                             | 62.61 (10.75)                                              |
| 5000                            | 0.40 (0.06)        | 0.43 (0.26)        | 0.12 (0.06)                                                | N/A                                                        | 119.18 (22.55)                                             | 46.45 (8.22)                                               |

Based on 20 repetitions. Oracle baselines (Rust, MLIRL) were based on bootstrap repetition of 100.

Table 2: Mean Absolute Percentage Error (MAPE) (%) of r Estimation. (# dummy = 0)

visits a few states much more frequently than others, 'the use of a projection operator onto a space of function approximators with respect to a distribution induced by the behavior policy can result in poor performance if that distribution does not sufficiently cover the state space.' (Tsitsiklis and Van Roy 1996a) This makes Oracle baseline predictions for states with mileage 1-5 slightly worse than GLADIUS.

2. Since we evaluate MAPE on the police played in the data, this implies that our evaluation mostly samples mileages 1-5, and GLADIUS's weakness in extrapolation for mileage 6-10 matters less than the slight imprecision of oracle parametric methods in mileages 1-5.

Table 3: Estimated rewards and frequency values for 1,000 trajectories for action 0.

| Mileages    |        1 |        2 |        3 |       4 |       5 |      6 |      7 |      8 |      9 |      10 |
|-------------|----------|----------|----------|---------|---------|--------|--------|--------|--------|---------|
| Frequency   | 7994     | 1409     | 1060     | 543     | 274     | 35     |  8     |  1     |  0     |   0     |
| True reward |   -1     |   -2     |   -3     |  -4     |  -5     | -6     | -7     | -8     | -9     | -10     |
| ML-IRL      |   -1.013 |   -2.026 |   -3.039 |  -4.052 |  -5.065 | -6.078 | -7.091 | -8.104 | -9.117 | -10.13  |
| Rust        |   -1.012 |   -2.023 |   -3.035 |  -4.047 |  -5.058 | -6.07  | -7.082 | -8.093 | -9.105 | -10.117 |
| GLADIUS     |   -1     |   -1.935 |   -2.966 |  -3.998 |  -4.966 | -5.904 | -6.769 | -7.633 | -8.497 |  -9.361 |

Finally, it is not surprising to see IQ-learn and BC underperform in the reward function estimation task since they do not require/ensure that the Bellman condition holds. See Appendix 8 for a detailed discussion.

## 7.3.2 Performance results for the high-dimensional set-up.

Figure 1 (below) presents high-dimensional experiments, where states were appended with dummy variables. Each dummy variable is of dimension 20. Note that a state space of dimensionality 20 10 (10 dummy variables with 20 possible values each) is equivalent to 10 13 , which is infeasible for existing exact methods (e.g., Rust) and methods that require transition probability estimation (e.g., ML-IRL). Therefore, we only present comparisons to the non-oracle methods.

We find that our approach outperforms benchmark algorithms, including SAmQ, IQ-learn, and BC (see Figure 1). Moreover, as illustrated in the right panel of Figure 1, the MAPE grows sub-linearly with the size of the state space (note the logarithmic x -axis). Remarkably, even in the extreme setting with K = 100 dummy

| Dummy Variables   | GLADIUS      | SAmQ        | IQ-learn     | BC           |
|-------------------|--------------|-------------|--------------|--------------|
| Dummy Variables   | MAPE (SE)    | MAPE (SE)   | MAPE (SE)    | MAPE (SE)    |
| 2                 | 1.24 (0.45)  | 1.79 (0.37) | 112.0 (14.8) | 150.9 (29.1) |
| 5                 | 2.51 (1.19)  | 2.77 (0.58) | 192.2 (19.2) | 171.1 (37.3) |
| 20                | 6.07 (3.25)  | N/A         | 180.1 (15.6) | 180.0 (33.7) |
| 50                | 9.76 (3.68)  | N/A         | 282.2 (25.2) | 205.1 (35.3) |
| 100               | 11.35 (4.24) | N/A         | 321.1 (23.1) | 288.8 (42.9) |

Based on 10 repetitions. For SAmQ, N/A means that the algorithm did not scale.

GLADIUS Performance

![Image](data:image/png;base64,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)

Figure 1: MAPE of r estimation. The left panel shows the MAPE values in a tabular format, and the right panel visualizes the GLADIUS's performance on a log-scaled x-axis. 1000 trajectories were used for all experiments. Smaller is better; the best value in each row is highlighted.

variables-an astronomically large discrete state space of 20 100 ≈ 10 130 distinct configurations-GLADIUS's MAPE rises only to about 10%. This limited loss in precision due to such a massive dimensional expansion underscores the method's robustness and practical scalability to very high-dimensional applications.

## 8 Imitation Learning experiments

One of the key contributions of this paper is the characterization of the relationship between imitation learning (IL) and inverse reinforcement learning (IRL)/Dynamic Discrete Choice (DDC) model, particularly through the ERM-IRL/DDC framework. Given that much of the IRL literature has historically focused on providing experimental results for IL tasks, we conduct a series of experiments to empirically validate our theoretical findings. Specifically, we aim to test our prediction in Section 4 that behavioral cloning (BC) should outperform ERM-IRL for IL tasks , as BC directly optimizes the negative log-likelihood objective without the additional complexity of Bellman error minimization. By comparing BC and ERM-IRL across various IL benchmark tasks, we demonstrate that BC consistently achieves better performance in terms of both computational efficiency and policy accuracy, reinforcing our claim that IL is a strictly easier problem than IRL.

## 8.1 Experimental Setup

As in Garg et al. (2021), we employ three OpenAI Gym environments for algorithms with discrete actions (Brockman 2016): Lunar Lander v2, Cartpole v1, and Acrobot v1. These environments are widely used in IL and RL research, providing well-defined optimal policies and performance metrics.

Dataset. For each environment, we generate expert demonstrations using a pre-trained policy. We use publicly available expert policies 15 trained via Proximal Policy Optimization (PPO) Schulman et al. (2017), as implemented in the Stable-Baselines3 library (Raffin et al. 2021). Each expert policy is run to generate demonstration trajectories, and we vary the number of expert trajectories across experiments for training. For

15 https://huggingface.co/sb3/

all experiments, we used the expert policy demonstration data from 10 episodes for testing. Performance Metric. The primary evaluation metric is % optimality, defined as:

<!-- formula-not-decoded -->

For each baseline, we report the mean and standard deviation of 100 evaluation episodes after training. A higher % optimality indicates that the algorithm's policy closely matches the expert. The 1000-episodic mean and standard deviation ([mean ± std]) of the episodic reward of expert policy for each environment was [232 . 77 ± 73 . 77] for Lunar-Lander v2 (larger the better), [ -82 . 80 ± 27 . 55] for Acrobot v1 (smaller the better), and [500 ± 0] for Cartpole v1 (larger the better).

Training Details. All algorithms were trained for 5,000 epochs. Since our goal in this experiment is to show the superiority of BC for IL tasks, we only include ERM-IRL and IQ-learn Garg et al. (2021) as baselines. Specifically, we exclude baselines such as Rust (Rust 1987) and ML-IRL (Zeng et al. 2023), which require explicit estimation of transition probabilities.

## 8.2 Experiment results

Table 4 presents the % optimality results for Lunar Lander v2, Cartpole v1, and Acrobot v1. As predicted in our theoretical analysis, BC consistently outperforms ERM-IRL in terms of % optimality, validating our theoretical claims.

|       | Lunar Lander v2 (%) (Larger %the better)   | Lunar Lander v2 (%) (Larger %the better)   | Lunar Lander v2 (%) (Larger %the better)   | Cartpole v1 (%) (Larger %the better)   | Cartpole v1 (%) (Larger %the better)   | Cartpole v1 (%) (Larger %the better)   | Acrobot v1 (%) (Smaller %the better)   | Acrobot v1 (%) (Smaller %the better)   | Acrobot v1 (%) (Smaller %the better)   |
|-------|--------------------------------------------|--------------------------------------------|--------------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| Trajs | Gladius                                    | IQ-learn                                   | BC                                         | Gladius                                | IQ-learn                               | BC                                     | Gladius                                | IQ-learn                               | BC                                     |
| 1     | 107.30                                     | 83.78                                      | 103.38                                     | 100.00                                 | 100.00                                 | 100.00                                 | 103.67                                 | 103.47                                 | 100.56                                 |
| 3     | 106.64 (11.11)                             | 102.44 (20.66)                             | 104.46 (11.57)                             | 100.00 (0.00)                          | 100.00 (0.00)                          | 100.00 (0.00)                          | 102.19 (22.69)                         | 101.28 (37.51)                         | 101.25 (36.42)                         |
| 7     | 101.10 (16.28)                             | 104.91 (13.98)                             | 105.99 (10.20)                             | 100.00 (0.00)                          | 100.00 (0.00)                          | 100.00 (0.00)                          | 100.67 (22.30)                         | 100.58 (30.09)                         | 98.08 (24.27)                          |
| 10    | 104.46 (13.65)                             | 105.13 (13.83)                             | 107.01 (10.75)                             | 100.00 (0.00)                          | 100.00 (0.00)                          | 100.00 (0.00)                          | 99.07 (20.58)                          | 101.10 (30.40)                         | 97.75 (16.67)                          |
| 15    | 106.11 (10.65)                             | 106.51 (14.10)                             | 107.42 (10.45)                             | 100.00 (0.00)                          | 100.00 (0.00)                          | 100.00 (0.00)                          | 96.50 (18.53)                          | 95.34 (26.92)                          | 95.33 (15.42)                          |

Based on 100 episodes for each baseline. Each baseline was trained for 5000 epochs.

Table 4: Mean and standard deviation of % optimality of 100 episodes

## 9 Conclusion

In this paper, we propose a provably globally convergent empirical risk minimization framework that combines non-parametric estimation methods (e.g., machine learning methods) with IRL/DDC models. This method's convergence to global optima stems from our new theoretical finding that the Bellman error (i.e.,

Bellman residual) satisfies the Polyak-Łojasiewicz (PL) condition, which is a weaker but almost equally useful condition as strong convexity for providing theoretical assurances.

The three key advantages of our method are: (1) it is easily applicable to high-dimensional state spaces, (2) it can operate without the knowledge of (or requiring the estimation of) state-transition probabilities, and (3) it is applicable to infinite state spaces. These three properties make our algorithm practically applicable and useful in high-dimensional, infinite-size state and action spaces that are common in business and economics applications. We demonstrate our approach's empirical performance through extensive simulation experiments (covering both low and high-dimensional settings). We find that, on average, our method performs quite well in recovering rewards in both low and high-dimensional settings. Further, it has better/on-par performance compared to other benchmark algorithms in this area (including algorithms that assume the parametric form of the reward function and knowledge of state transition probabilities) and is able to recover rewards even in settings where other algorithms are not viable.

## References

Baptiste Abeles, Eugenio Clerico, and Gergely Neu. Generalization bounds for mixing processes via delayed onlineto-pac conversions. arXiv preprint arXiv:2406.12600 , 2024.

Karun Adusumilli and Dita Eckardt. Temporal-difference estimation of dynamic discrete choice models. arXiv preprint arXiv:1912.09509 , 2019.

Victor Aguirregabiria and Pedro Mira. Swapping the nested fixed point algorithm: A class of estimators for discrete markov decision models. Econometrica , 70(4):1519-1543, 2002.

Victor Aguirregabiria and Pedro Mira. Sequential estimation of dynamic discrete games. Econometrica , 75(1):1-53, 2007.

Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. A convergence theory for deep learning via over-parameterization. In International conference on machine learning , pages 242-252. PMLR, 2019.

Andr´ as Antos, Csaba Szepesv´ ari, and R´ emi Munos. Learning near-optimal policies with bellman-residual minimization based fitted policy iteration and a single sample path. Machine Learning , 71:89-129, 2008.

Peter Arcidiacono and Paul B Ellickson. Practical methods for estimation of dynamic discrete choice models. Annu. Rev. Econ. , 3(1):363-394, 2011.

Peter Arcidiacono and Robert A Miller. Conditional choice probability estimation of dynamic discrete choice models with unobserved heterogeneity. Econometrica , 79(6):1823-1867, 2011.

Peter Arcidiacono, Patrick Bayer, Federico A Bugni, and Jonathan James. Approximating high-dimensional dynamic models: Sieve value function iteration. In Structural Econometric Models , pages 45-95. Emerald Group Publishing Limited, 2013.

Peter Arcidiacono, Patrick Bayer, Jason R Blevins, and Paul B Ellickson. Estimation of dynamic discrete choice models in continuous time with an application to retail competition. The Review of Economic Studies , 83(3):889-931, 2016.

Matt Barnes, Matthew Abueg, Oliver F Lange, Matt Deeds, Jason Trader, Denali Molitor, Markus Wulfmeier, and Shawn O'Banion. Massively scalable inverse reinforcement learning in google maps. arXiv preprint arXiv:2305.11290 , 2023.

Ebrahim Barzegary and Hema Yoganarasimhan. A recursive partitioning approach for dynamic discrete choice modeling in high dimensional settings. arXiv preprint arXiv:2208.01476 , 2022.

Joan Bas-Serrano, Sebastian Curi, Andreas Krause, and Gergely Neu. Logistic q-learning. In International conference on artificial intelligence and statistics , pages 3610-3618. PMLR, 2021.

Hugo Benitez-Silva, George Hall, G¨ unter J Hitsch, Giorgio Pauletto, and John Rust. A comparison of discrete and parametric approximation methods for continuous-state dynamic programming problems. manuscript, Yale University , 2000.

G Brockman. Openai gym. arXiv preprint arXiv:1606.01540 , 2016.

Haoyang Cao, Samuel Cohen, and Lukasz Szpruch. Identifiability in inverse reinforcement learning. Advances in Neural Information Processing Systems , 34:12362-12373, 2021.

Jinglin Chen and Nan Jiang. Information-theoretic considerations in batch reinforcement learning. In International conference on machine learning , pages 1042-1051. PMLR, 2019.

Victor Chernozhukov, Juan Carlos Escanciano, Hidehiko Ichimura, Whitney K Newey, and James M Robins. Locally robust semiparametric estimation. Econometrica , 90(4):1501-1535, 2022.

Khai Xiang Chiong, Alfred Galichon, and Matt Shum. Duality in dynamic discrete-choice models. Quantitative Economics , 7(1):83-115, 2016.

Djork-Arn´ e Clevert, Thomas Unterthiner, and Sepp Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). arXiv preprint arXiv:1511.07289 , 2015.

George Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of control, signals and systems , 2(4):303-314, 1989.

Bo Dai, Albert Shaw, Lihong Li, Lin Xiao, Niao He, Zhen Liu, Jianshu Chen, and Le Song. Sbeed: Convergent reinforcement learning with nonlinear function approximation. In International conference on machine learning , pages 1125-1134. PMLR, 2018.

Yehuda Dar, Vidya Muthukumar, and Richard G Baraniuk. A farewell to the bias-variance tradeoff? an overview of the theory of overparameterized machine learning. arXiv preprint arXiv:2109.02355 , 2021.

Stefano Ermon, Yexiang Xue, Russell Toth, Bistra Dilkina, Richard Bernstein, Theodoros Damoulas, Patrick Clark, Steve DeGloria, Andrew Mude, Christopher Barrett, et al. Learning large-scale dynamic discrete choice models of spatio-temporal preferences with application to migratory pastoralism in east africa. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 29, 2015.

Yiding Feng, Ekaterina Khmelnitskaya, and Denis Nekipelov. Global concavity and optimization in a class of dynamic discrete choice models. In International Conference on Machine Learning , pages 3082-3091. PMLR, 2020. Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv preprint arXiv:1611.03852 , 2016. Offline reinforcement learning:

Dylan J Foster, Akshay Krishnamurthy, David Simchi-Levi, and Yunzong Xu. Fundamental barriers for value function approximation. arXiv preprint arXiv:2111.10919 , 2021.

Dylan J Foster, Adam Block, and Dipendra Misra. Is behavior cloning all you need? understanding horizon in imitation learning. Advances in Neural Information Processing Systems , 37:120602-120666, 2024.

Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. arXiv preprint arXiv:1710.11248 , 2017.

Shi Fu, Yunwen Lei, Qiong Cao, Xinmei Tian, and Dacheng Tao. Sharper bounds for uniformly stable algorithms with stationary mixing process. In The Eleventh International Conference on Learning Representations , 2023. Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, and Stefano Ermon. Iq-learn: Inverse soft-q learning for imitation. Advances in Neural Information Processing Systems , 34:4028-4039, 2021.

Sinong Geng, Houssam Nassif, Carlos Manzanares, Max Reppen, and Ronnie Sircar. Deep pqr: Solving inverse

reinforcement learning using anchor actions. In International Conference on Machine Learning , pages 3431-3441. PMLR, 2020.

Sinong Geng, Houssam Nassif, and Carlos A Manzanares. A data-driven state aggregation approach for dynamic discrete choice models. In Uncertainty in Artificial Intelligence , pages 647-657. PMLR, 2023.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM , 63(11):139-144, 2020.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.

Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. Advances in neural information processing systems , 29, 2016.

Kurt Hornik. Approximation capabilities of multilayer feedforward networks. Neural networks , 4(2):251-257, 1991. V Joseph Hotz and Robert A Miller. Conditional choice probabilities and the estimation of dynamic models. The Review of Economic Studies , 60(3):497-529, 1993.

Susumu Imai, Neelam Jain, and Andrew Ching. Bayesian estimation of dynamic discrete choice models. Econometrica , 77(6):1865-1899, 2009.

Nan Jiang and Tengyang Xie. Offline reinforcement learning in large state spaces: Algorithms and guarantees. Statistical Science , 2024.

Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning , pages 17041713. PMLR, 2017.

Enoch H. Kang and Kyoungseok Jang. Stability and generalization for bellman residuals, 2025. URL https: //arxiv.org/abs/2508.18741 .

Hiroyuki Kasahara and Katsumi Shimotsu. Nonparametric identification of finite mixture models of dynamic discrete choices. Econometrica , 77(1):135-175, 2009.

Trupti M Kodinariya, Prashant R Makwana, et al. Review on determining number of cluster in k-means clustering. International Journal , 1(6):90-95, 2013.

Ilya Kostrikov, Ofir Nachum, and Jonathan Tompson. Imitation learning via off-policy distribution matching. arXiv preprint arXiv:1912.05032 , 2019.

Dennis Kristensen, Patrick K Mogensen, Jong Myun Moon, and Bertel Schjerning. Solving dynamic discrete choice models using smoothing and sieve methods. Journal of Econometrics , 223(2):328-360, 2021.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative q-learning for offline reinforcement learning. Advances in neural information processing systems , 33:1179-1191, 2020.

Michail G Lagoudakis and Ronald Parr. Least-squares policy iteration. Journal of machine learning research , 4(Dec): 1107-1149, 2003.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643 , 2020.

Feng-Yi Liao, Lijun Ding, and Yang Zheng. Error bounds, pl condition, and quadratic growth for weakly convex functions, and linear convergences of proximal point methods. In 6th Annual Learning for Dynamics &amp; Control Conference , pages 993-1005. PMLR, 2024.

Chaoyue Liu, Libin Zhu, and Mikhail Belkin. Loss landscapes and optimization in over-parameterized non-linear systems and neural networks. Applied and Computational Harmonic Analysis , 59:85-116, 2022.

Thierry Magnac and David Thesmar. Identifying dynamic discrete decision processes. Econometrica , 70(2):801-816, 2002.

Daniel McFadden. Economic choices. American economic review , 91(3):351-378, 2001.

Mehryar Mohri and Afshin Rostamizadeh. Stability bounds for stationary φ -mixing and β -mixing processes. Journal of Machine Learning Research , 11(2), 2010.

R´ emi Munos. Error bounds for approximate policy iteration. In Proceedings of the Twentieth International Conference on International Conference on Machine Learning , pages 560-567, 2003.

Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml , volume 99, pages 278-287, 1999.

Tianwei Ni, Harshit Sikchi, Yufei Wang, Tejus Gupta, Lisa Lee, and Ben Eysenbach. f-irl: Inverse reinforcement learning via state marginal matching. In Conference on Robot Learning , pages 529-551. PMLR, 2021.

Andriy Norets. Inference in dynamic discrete choice models with serially orrelated unobserved state variables. Econometrica , 77(5):1665-1682, 2009.

Andriy Norets. Estimation of dynamic discrete choice models using artificial neural network approximations. Econometric Reviews , 31(1):84-106, 2012.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, et al. 2 olmo 2 furious. arXiv preprint arXiv:2501.00656 , 2024.

Andrew Patterson, Adam White, and Martha White. A generalized projected bellman error for off-policy value estimation in reinforcement learning. Journal of Machine Learning Research , 23(145):1-61, 2022.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From r to q ∗ : Your language model is secretly a q-function. arXiv preprint arXiv:2404.12358 , 2024.

Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stablebaselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research , 22(268):1-8, 2021.

Nived Rajaraman, Lin Yang, Jiantao Jiao, and Kannan Ramchandran. Toward the fundamental limits of imitation learning. Advances in Neural Information Processing Systems , 33:2914-2924, 2020.

Quentin Rebjock and Nicolas Boumal. Fast convergence to non-isolated minima: four equivalent conditions for C 2 functions. arXiv preprint arXiv:2303.00096 , 2023.

Gregor Reich. Divide and conquer: recursive likelihood function integration for hidden markov models with continuous latent variables. Operations research , 66(6):1457-1470, 2018.

Mark Rudelson and Roman Vershynin. Smallest singular value of a random rectangular matrix. Communications on Pure and Applied Mathematics: A Journal Issued by the Courant Institute of Mathematical Sciences , 62(12): 1707-1739, 2009.

John Rust. Optimal replacement of gmc bus engines: An empirical model of harold zurcher. Econometrica: Journal of the Econometric Society , pages 999-1033, 1987.

John Rust. Structural estimation of markov decision processes. Handbook of econometrics , 4:3081-3143, 1994. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.

Sina Sharifi and Mahyar Fazlyab. Provable bounds on the hessian of neural networks: Derivative-preserving reachability analysis. arXiv preprint arXiv:2406.04476 , 2024.

Kristina P Sinaga and Miin-Shen Yang. Unsupervised k-means clustering algorithm. IEEE access , 8:80716-80727, 2020.

Che-Lin Su and Kenneth L Judd. Constrained optimization approaches to estimation of structural models. Econometrica , 80(5):2213-2230, 2012.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction . 2018.

Gerald Tesauro et al. Temporal difference learning and td-gammon. Communications of the ACM , 38(3):58-68, 1995. Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation. arXiv preprint arXiv:1805.01954 , 2018.

John Tsitsiklis and Benjamin Van Roy. Analysis of temporal-diffference learning with function approximation. Advances in neural information processing systems , 9, 1996a.

John N Tsitsiklis and Benjamin Van Roy. Feature-based methods for large scale dynamic programming. Machine Learning , 22(1):59-94, 1996b.

Masatoshi Uehara, Jiawei Huang, and Nan Jiang. Minimax weight and q-function learning for off-policy evaluation. In International Conference on Machine Learning , pages 9659-9668. PMLR, 2020.

Hado Van Hasselt, Yotam Doron, Florian Strub, Matteo Hessel, Nicolas Sonnerat, and Joseph Modayil. Deep reinforcement learning and the deadly triad. arXiv preprint arXiv:1812.02648 , 2018.

Ruosong Wang, Yifan Wu, Ruslan Salakhutdinov, and Sham Kakade. Instabilities of offline rl with pre-trained neural representation. In International Conference on Machine Learning , pages 10948-10960. PMLR, 2021.

Tengyang Xie and Nan Jiang. Batch value-function approximation with only realizability. In International Conference on Machine Learning , pages 11404-11413. PMLR, 2021.

Junchi Yang, Negar Kiyavash, and Niao He. Global convergence and variance-reduced optimization for a class of nonconvex-nonconcave minimax problems. arXiv preprint arXiv:2002.09621 , 2020.

Yingyao Hu Fangzhu Yang. Estimation of dynamic discrete choice models with unobserved state variables using reinforcement learning. 2024.

Sheng Yue, Guanbo Wang, Wei Shao, Zhaofeng Zhang, Sen Lin, Ju Ren, and Junshan Zhang. Clare: Conservative model-based reward learning for offline inverse reinforcement learning. arXiv preprint arXiv:2302.04782 , 2023.

Andrea Zanette. When is realizability sufficient for off-policy reinforcement learning? In International Conference on Machine Learning , pages 40637-40668. PMLR, 2023.

Maryam Zare, Parham M Kebria, Abbas Khosravi, and Saeid Nahavandi. A survey of imitation learning: Algorithms, recent developments, and challenges. IEEE Transactions on Cybernetics , 2024.

Siliang Zeng, Chenliang Li, Alfredo Garcia, and Mingyi Hong. Understanding expertise through demonstrations: A

maximum likelihood framework for offline inverse reinforcement learning. arXiv preprint arXiv:2302.07457 , 2023.

Simon Sinong Zhan, Qingyuan Wu, Philip Wang, Yixuan Wang, Ruochen Jiao, Chao Huang, and Qi Zhu. Modelbased reward shaping for adversarial inverse reinforcement learning in stochastic environments. arXiv preprint arXiv:2410.03847 , 2024.

Wenhao Zhan, Baihe Huang, Audrey Huang, Nan Jiang, and Jason Lee. Offline reinforcement learning with realizability and single-policy concentrability. In Conference on Learning Theory , pages 2730-2775. PMLR, 2022.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. arXiv preprint arXiv:1611.03530 , 2016.

Han Zhong, Guhao Feng, Wei Xiong, Xinle Cheng, Li Zhao, Di He, Jiang Bian, and Liwei Wang. Dpo meets ppo: Reinforced token optimization for rlhf. arXiv preprint arXiv:2404.18922 , 2024.

Li Ziniu, Xu Tian, Yu Yang, and Luo Zhi-Quan. Rethinking valuedice - does it really improve performance? In ICLR Blog Track , 2022. URL https://iclr-blog-track.github.io/2022/03/25/ rethinking-valuedice/ . https://iclr-blog-track.github.io/2022/03/25/rethinking-valuedice/.

Konrad Zolna, Alexander Novikov, Ksenia Konyushkova, Caglar Gulcehre, Ziyu Wang, Yusuf Aytar, Misha Denil, Nando de Freitas, and Scott Reed. Offline learning from demonstrations and unlabeled experience. arXiv preprint arXiv:2011.13885 , 2020.

3.5 -

3.0 -

2.5 -

2.0 -

1.5 -

1.0 -

0.5 -

0.0

Absolute Error for Action ao (50 Trajectories)

• CCP ao.

—•- Rust ao.

-*- GLADIUS ao

## Web Appendix

2

0.20

0.10

## A Extended experiment discussions

## A.1 More discussions on Bus engine replacement experiments

Figure A1, A2 and Table A1 - A4 shown below present the estimated results for reward and Q ∗ using 50 trajectories (5,000 transitions) and 1,000 trajectories (100,000 transitions). As you can see in Figure A1 and A2, Rust and ML-IRL , which know the exact transition probabilities and employ correct parameterization (i.e., linear), demonstrate strong extrapolation capabilities for [Mileage, action] pairs that are rarely observed or entirely missing from the dataset (mileage 6-10). In contrast, GLADIUS, a neural network-based method, struggles with these underrepresented pairs. However, as we saw in the main text's Table 2, GLADIUS achieves par or lower Mean Absolute Percentage Error (MAPE), which is defined as 1 N ∑ N i =1 ∣ ∣ ∣ ˆ r i -r i r i ∣ ∣ ∣ × 100 where N is the total number of samples from expert policy π ∗ and ˆ r i is each algorithm's estimator for the true reward r i . This is because it overall outperforms predicting r values for the [Mileage, action] pairs that appear most frequently and therefore contribute most significantly to the error calculation, as indicated by the visibility of the yellow shading in the tables below. (Higher visibility implies larger frequency.)

## Results for 50 trajectories (absolute error plot, r prediction, Q ∗ prediction)

Figure A1: Reward estimation error comparison for 50 trajectories. Closer to 0 (black line) is better.

![Image](data:image/png;base64,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)

Results for 1000 trajectories (absolute error plot, r prediction, Q ∗ prediction)

Absolute Error for Action a1 (50 Trajectories)

----I----=

------I------

------

• CCP a11

-=- Rust a1|

GLADIUS a1

Ab:

0.6

0.5

0.2

0.1 -

0.0

Absolute Error for Action ao (1000 Trajectories)

• CCP ao.

-=- Rust ao

→* - GLADIUS ao

4

Absolute Error for Action a1 (1000 Trajectories)

0.04 -

0.03

| Mileage   | Frequency   | Frequency   | Ground Truth r   | Ground Truth r   | ML-IRL   | ML-IRL   | Rust   | Rust   | GLADIUS   | GLADIUS   |
|-----------|-------------|-------------|------------------|------------------|----------|----------|--------|--------|-----------|-----------|
| Mileage   | a 0         | a 1         | a 0              | a 1              | a 0      | a 1      | a 0    | a 1    | a 0       | a 1       |
| 1         | 412         | 37          | -1.000           | -5.000           | -0.959   | -4.777   | -0.965 | -4.812 | -1.074    | -4.999    |
| 2         | 65          | 18          | -2.000           | -5.000           | -1.918   | -4.777   | -1.931 | -4.812 | -1.978    | -5.001    |
| 3         | 43          | 80          | -3.000           | -5.000           | -2.877   | -4.777   | -2.896 | -4.812 | -3.105    | -5.000    |
| 4         | 24          | 101         | -4.000           | -5.000           | -3.836   | -4.777   | -3.861 | -4.812 | -3.844    | -5.001    |
| 5         | 8           | 134         | -5.000           | -5.000           | -4.795   | -4.777   | -4.827 | -4.812 | -4.878    | -5.001    |
| 6         | 4           | 37          | -6.000           | -5.000           | -5.753   | -4.777   | -5.792 | -4.812 | -6.642    | -5.001    |
| 7         | 1           | 26          | -7.000           | -5.000           | -6.712   | -4.777   | -6.757 | -4.812 | -8.406    | -5.001    |
| 8         | 0           | 7           | -8.000           | -5.000           | -7.671   | -4.777   | -7.722 | -4.812 | -10.170   | -5.001    |
| 9         | 0           | 2           | -9.000           | -5.000           | -8.630   | -4.777   | -8.688 | -4.812 | -11.934   | -5.001    |
| 10        | 0           | 1           | -10.000          | -5.000           | -9.589   | -4.777   | -9.653 | -4.812 | -13.684   | -5.002    |

Table A1: Reward estimation for 50 trajectories. Color indicates appearance frequencies.

Table A2: Q ∗ estimation for 50 trajectories. Color indicates appearance frequencies.

| Mileage   | Frequency   | Frequency   | Ground Truth Q   | Ground Truth Q   | ML-IRL Q   | ML-IRL Q   | Rust Q   | Rust Q   | GLADIUS Q   | GLADIUS Q   |
|-----------|-------------|-------------|------------------|------------------|------------|------------|----------|----------|-------------|-------------|
| Mileage   | a 0         | a 1         | a 0              | a 1              | a 0        | a 1        | a 0      | a 1      | a 0         | a 1         |
| 1         | 412         | 37          | -52.534          | -54.815          | -49.916    | -52.096    | -50.327  | -52.523  | -53.059     | -55.311     |
| 2         | 65          | 18          | -53.834          | -54.815          | -51.165    | -52.096    | -51.584  | -52.523  | -54.270     | -55.312     |
| 3         | 43          | 80          | -54.977          | -54.815          | -52.266    | -52.096    | -52.691  | -52.523  | -55.548     | -55.312     |
| 4         | 24          | 101         | -56.037          | -54.815          | -53.286    | -52.096    | -53.718  | -52.523  | -56.356     | -55.312     |
| 5         | 8           | 134         | -57.060          | -54.815          | -54.270    | -52.096    | -54.708  | -52.523  | -57.419     | -55.312     |
| 6         | 4           | 37          | -58.069          | -54.815          | -55.239    | -52.096    | -55.683  | -52.523  | -59.184     | -55.312     |
| 7         | 1           | 26          | -59.072          | -54.815          | -56.202    | -52.096    | -56.652  | -52.523  | -60.950     | -55.312     |
| 8         | 0           | 7           | -60.074          | -54.815          | -57.162    | -52.096    | -57.619  | -52.523  | -62.715     | -55.312     |
| 9         | 0           | 2           | -61.074          | -54.815          | -58.122    | -52.096    | -58.585  | -52.523  | -64.481     | -55.312     |
| 10        | 0           | 1           | -62.074          | -54.815          | -59.081    | -52.096    | -59.550  | -52.523  | -66.228     | -55.308     |

Figure A2: Reward estimation error comparison for 1,000 trajectories. Closer to 0 (black line) is better.

![Image](data:image/png;base64,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)

Table A3: Reward estimation for 1,000 trajectories. Color indicates appearance frequencies.

| Mileage   | Frequency   | Frequency   | Ground Truth r   | Ground Truth r   | ML-IRL   | ML-IRL   | Rust    | Rust   | GLADIUS   | GLADIUS   |
|-----------|-------------|-------------|------------------|------------------|----------|----------|---------|--------|-----------|-----------|
| Mileage   | a 0         | a 1         | a 0              | a 1              | a 0      | a 1      | a 0     | a 1    | a 0       | a 1       |
| 1         | 7994        | 804         | -1.000           | -5.000           | -1.013   | -5.043   | -1.012  | -5.033 | -1.000    | -5.013    |
| 2         | 1409        | 541         | -2.000           | -5.000           | -2.026   | -5.043   | -2.023  | -5.033 | -1.935    | -5.001    |
| 3         | 1060        | 1296        | -3.000           | -5.000           | -3.039   | -5.043   | -3.035  | -5.033 | -2.966    | -5.000    |
| 4         | 543         | 1991        | -4.000           | -5.000           | -4.052   | -5.043   | -4.047  | -5.033 | -3.998    | -5.002    |
| 5         | 274         | 2435        | -5.000           | -5.000           | -5.065   | -5.043   | -5.058  | -5.033 | -4.966    | -5.002    |
| 6         | 35          | 829         | -6.000           | -5.000           | -6.078   | -5.043   | -6.070  | -5.033 | -5.904    | -5.002    |
| 7         | 8           | 476         | -7.000           | -5.000           | -7.091   | -5.043   | -7.082  | -5.033 | -6.769    | -5.002    |
| 8         | 1           | 218         | -8.000           | -5.000           | -8.104   | -5.043   | -8.093  | -5.033 | -7.633    | -5.003    |
| 9         | 0           | 73          | -9.000           | -5.000           | -9.117   | -5.043   | -9.105  | -5.033 | -8.497    | -5.003    |
| 10        | 0           | 10          | -10.000          | -5.000           | -10.130  | -5.043   | -10.117 | -5.033 | -9.361    | -5.004    |

Table A4: Q ∗ estimation for 1,000 trajectories. Color indicates appearance frequencies.

| Mileage   | Frequency   | Frequency   | Ground Truth Q   | Ground Truth Q   | ML-IRL Q   | ML-IRL Q   | Rust Q   | Rust Q   | GLADIUS Q   | GLADIUS Q   |
|-----------|-------------|-------------|------------------|------------------|------------|------------|----------|----------|-------------|-------------|
| Mileage   | a 0         | a 1         | a 0              | a 1              | a 0        | a 1        | a 0      | a 1      | a 0         | a 1         |
| 1         | 7994        | 804         | -52.534          | -54.815          | -53.110    | -55.405    | -53.019  | -55.309  | -52.431     | -54.733     |
| 2         | 1409        | 541         | -53.834          | -54.815          | -54.423    | -55.405    | -54.330  | -55.309  | -53.680     | -54.720     |
| 3         | 1060        | 1296        | -54.977          | -54.815          | -55.578    | -55.405    | -55.483  | -55.309  | -54.852     | -54.721     |
| 4         | 543         | 1991        | -56.037          | -54.815          | -56.649    | -55.405    | -56.554  | -55.309  | -55.942     | -54.721     |
| 5         | 274         | 2435        | -57.060          | -54.815          | -57.684    | -55.405    | -57.588  | -55.309  | -56.932     | -54.721     |
| 6         | 35          | 829         | -58.069          | -54.815          | -58.705    | -55.405    | -58.608  | -55.309  | -57.886     | -54.721     |
| 7         | 8           | 476         | -59.072          | -54.815          | -59.721    | -55.405    | -59.623  | -55.309  | -58.745     | -54.721     |
| 8         | 1           | 218         | -60.074          | -54.815          | -60.735    | -55.405    | -60.636  | -55.309  | -59.604     | -54.722     |
| 9         | 0           | 73          | -61.074          | -54.815          | -61.748    | -55.405    | -61.648  | -55.309  | -60.463     | -54.722     |
| 10        | 0           | 10          | -62.074          | -54.815          | -62.760    | -55.405    | -62.660  | -55.309  | -61.322     | -54.722     |

## B Equivalence between Dynamic Discrete choice and Entropy regularized Inverse Reinforcement learning

## B.1 Properties of Type 1 Extreme Value (T1EV) distribution

Type 1 Extreme Value (T1EV), or Gumbel distribution, has a location parameter and a scale parameter. The T1EV distribution with location parameter ν and scale parameter 1 is denoted as Gumbel ( ν, 1) and has its CDF, PDF, and mean as follows:

<!-- formula-not-decoded -->

Suppose that we are given a set of N independent Gumbel random variables G i , each with their own parameter ν i , i.e. G i ∼ Gumbel( ν i , 1) .

Lemma 12. Let Z = max G i . Then Z ∼ Gumbel( ν Z = log ∑ i e ν i , 1) .

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

̸

Lemma14. Let G 1 ∼ Gumbel ( ν 1 , 1) and G 2 ∼ Gumbel ( ν 2 , 1) . Then E [ G 1 | G 1 ≥ G 2 ] = γ +log ( 1 + e ( -( ν 1 -ν 2 )) ) holds.

Proof. Let ν 1 -ν 2 = c . Then E [ G 1 | G 1 ≥ G 2 ] is equivalent to ν 1 + ∫ + ∞ -∞ xF ( x + c ) f ( x )d x ∫ + ∞ -∞ F ( x + c ) f ( x )d x , where the pdf f and cdf F are associated with Gumbel (0 , 1) , because

<!-- formula-not-decoded -->

Now note that

Corollary 13. P ( G k &gt; max i = k G i ) = e ν k ∑ i e ν i .

Proof.

̸

̸

<!-- formula-not-decoded -->

Therefore, E [ G 1 | G 1 ≥ G 2 ] = γ + ν k +log ( 1 + e ( -( ν 1 -ν 2 )) ) holds.

Corollary 15. E [ G k | G k = max G i ] = γ + ν k -log ( e ν k ∑ i e ν i ) Proof.

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## B.2 Properties of entropy regularization

Suppose we have a choice out of discrete choice set A = { x i } |A| i =1 . A choice policy can be a deterministic policy such as argmax i ∈ 1 ,..., |A| x i , or stochastic policy that is characterized by q ∈ △ A . When we want to enforce smoothness in choice, we can regularize choice by newly defining the choice rule

<!-- formula-not-decoded -->

̸

̸

where Ω is a regularizing function.

Lemma16. When the regularizing function is constant -τ multiple of Shannon entropy H ( q ) = -∑ |A| i =1 q i log ( q i ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. In the following, I will assume τ &gt; 0 . Let

<!-- formula-not-decoded -->

We are going to find the max by computing the gradient and setting it to 0 . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and

This last equation states that the Hessian matrix is negative definite (since it is diagonal and -τ q 1 &lt; 0 ), and thus ensures that the stationary point we compute is actually the maximum. Setting the gradient to 0 yields q ∗ i = exp ( x i τ -1 ) , however the resulting q ∗ might not be a probability distribution. To ensure ∑ n i =1 q ∗ i = 1 , we add a normalization:

<!-- formula-not-decoded -->

This new q ∗ is still a stationary point and belongs to the probability simplex, so it must be the maximum. Hence, you get

<!-- formula-not-decoded -->

and

as desired.

## B.3 IRL with entropy regularization

## Markov decision processes

Consider an MDP defined by the tuple ( S , A , P, ν 0 , r, β ) :

- S and A denote finite state and action spaces
- P ∈ ∆ S×A S is a Markovian transition kernel, and ν 0 ∈ ∆ S is the initial state distribution.
- r ∈ R S×A is a reward function.
- β ∈ (0 , 1) a discount factor

## B.3.1 Agent behaviors

Denote the distribution of the agent's initial state s 0 ∈ S as ν 0 . Given a stationary Markov policy π ∈ ∆ S A , an agent starts from initial state s 0 and make an action a h ∈ A at state s h ∈ S according to a h ∼ π ( · | s h ) at each period h . We use P π ν 0 to denote the distribution over the sample space ( S × A ) ∞ = { ( s 0 , a 0 , s 1 , a 1 , . . . ) : s h ∈ S , a h ∈ A , h ∈ N } induced by the policy π and the initial distribution ν 0 . We also use E π to denote the expectation with respect to P π ν 0 . Maximum entropy inverse reinforcement learning (MaxEnt-IRL) makes the following assumption:

Assumption 6 (Assumption 1) . Agent follows the policy

<!-- formula-not-decoded -->

where H denotes the Shannon entropy and λ is the regularization parameter.

For the rest of the section, we use λ = 1 . Wethen define the function V as V ( s h ′ ) = E π ∗ [∑ ∞ h = h ′ β h ( r ( s h , a h ) + H ( π ∗ ( and call it the value function . According to Assumption 1, the value function V must satisfy the Bellman equation, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and q ∗ := arg max q ∈△ A { E a ∼ q [ r ( s, a ) + β · E [ V ( s ′ ) | s, a ]] + H ( q ) } is characterized by

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

- Equality in Equation (A2) and equality in Equation (A4) is from Lemma 16

This implies that π ∗ ( a | s ) = q ∗ a = exp( Q ( s,a )) ∑ a ′ ∈A exp( Q ( s,a ′ )) for a ∈ A . In addition to the Bellman equation in terms of value function V , Bellman equation in terms of choice-specific value function Q ( s, a ) can be derived by combining Q ( s, a ) := r ( s, a ) + β · E [ V ( s ′ ) | s, a ] and Equation (A3):

<!-- formula-not-decoded -->

We can also derive an alternative form of choice-specific value function Q ( s, a ) by combining Q ( s, a ) := r ( s, a ) + β · E s ′ ∼ P ( s,a ) [ V ( s ′ ) | s, a ] and Equation (A1):

<!-- formula-not-decoded -->

The last line comes from the fact that Q ( s ′ , a ′ ) -log π ∗ ( a ′ | s ′ ) is equivalent to log (∑ a ′ ∈A exp( Q ( s ′ , a ′ )) ) , which is a quantity that does not depend on the realization of specific action a ′ .

## B.4 Single agent Dynamic Discrete Choice (DDC) model

## Markov decision processes

Consider an MDP τ := ( S , A , P, ν 0 , r, G ( δ, 1) , β ) :

- S and A denote finite state and action spaces
- P ∈ ∆ S×A S is a Markovian transition kernel, and ν 0 ∈ ∆ S is the initial state distribution.

- r ( s h , a h ) + ϵ ah is the immediate reward (called the flow utility in the Discrete Choice Model literature) from taking action a h at state s h at time-step h , where:
- -r ∈ R S×A is a deterministic reward function
- -ϵ ah i.i.d. ∼ G ( δ, 1) is the random part of the reward, where G is Type 1 Extreme Value (T1EV) distribution (a.k.a. Gumbel distribution). The mean of G ( δ, 1) is δ + γ , where γ is the Euler constant.
- -In the econometrics literature, this reward setting is a result of the combination of two assumptions: conditional independence (CI) and additive separability (AS) Magnac and Thesmar (2002).
- β ∈ (0 , 1) a discount factor

Figure A3: Gumbel distribution G ( -γ, 1)

![Image](data:image/png;base64,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)

## B.4.1 Agent behaviors

Denote the distribution of agent's initial state s 0 ∈ S as ν 0 . Given a stationary Markov policy π ∈ ∆ S A , an agent starts from initial state s 0 and make an action a h ∈ A at state s h ∈ S according to a h ∼ π ( · | s h ) at each period h . We use P π ν 0 to denote the distribution over the sample space ( S × A ) ∞ = { ( s 0 , a 0 , s 1 , a 1 , . . . ) : s h ∈ S , a h ∈ A , h ∈ N } induced by the policy π and the initial distribution ν 0 . We also use E π to denote the expectation with respect to P π ν 0 . As in Inverse Reinforcement learning (IRL), a Dynamic Discrete Choice (DDC) model makes the following assumption:

Assumption 7. Agent makes decision according to the policy argmax π ∈ ∆ S A E π [∑ ∞ h =0 β h ( r ( s h , a h ) + ϵ ah ) ] .

As Assumption 7 specifies the agent's policy, we omit π in the notations from now on. Define ϵ h = [ ϵ 1 h . . . ϵ |A| h ] , where ϵ ih i.i.d ∼ G ( δ, 1) for i = 1 . . . |A| . We define a function V as

<!-- formula-not-decoded -->

and call it the value function. According to Assumption 7, the value function V must satisfy the Bellman equation, i.e.,

<!-- formula-not-decoded -->

Define

<!-- formula-not-decoded -->

We call ¯ V the expected value function, and Q ( s, a ) as the choice-specific value function. Then the Bellman equation can be written as

<!-- formula-not-decoded -->

Furthermore, Corollary 13 characterizes that the agent's optimal policy is characterized by

<!-- formula-not-decoded -->

In addition to Bellman equation in terms of value function V in Equation (A6), Bellman equation in terms of choice-specific value function Q comes from combining Equation (A7) and Equation (A9):

<!-- formula-not-decoded -->

When δ = -γ (i.e., the Gumbel noise is mean 0), we have

<!-- formula-not-decoded -->

This Bellman equation can be also written in another form.

<!-- formula-not-decoded -->

where π ∗ ( s, a ) = ( Q ( s,a ) ∑ a ′ ∈A Q ( s,a ′ ) ) .

## B.5 Equivalence between DDC and Entropy regularized IRL

Equation A2, Equation A4 and Equation A5 characterizes the choice-specific value function's Bellman equation and optimal policy in entropy regularized IRL setting when regularizing coefficient is 1:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation A10, Equation 3, and Equation A13 (when δ = -γ ) characterizes the choice-specific value function's Bellman equation and optimal policy of Dynamic Discrete Choice setting:

<!-- formula-not-decoded -->

Q that satisfies A10 is unique Rust (1994), and Q -r forms a one-to-one relationship. Therefore, the exact equivalence between these two setups implies that the same reward function r and discount factor β will lead to the same choice-specific value function Q and the same optimal policy for the two problems.

## B.6 IRL with occupancy matching

Ho and Ermon (2016) defines another inverse reinforcement learning problem that is based on the notion of occupancy matching. Let ν 0 be the initial state distribution and d π be the discounted state-action occupancy of π which is defined as d π = (1 -β ) ∑ ∞ t =0 β t d π t , with d π t ( s, a ) = P π,ν 0 [ s t = s, a t = a ] . Note that Q π ( s, a ) := E π [∑ ∞ t =0 β t r ( s t , a t ) | s 0 = s, a 0 = a ] = ∑ ∞ t =0 β t E (˜ s, ˜ a ) ∼ d π t [ r (˜ s, ˜ a ) | s 0 = a, a 0 = a ] . Defining the discounted state-action occupancy of the expert policy π ∗ as d ∗ , Ho and Ermon (2016) defines the inverse reinforcement learning problem as the following max-min problem:

<!-- formula-not-decoded -->

where H is the Shannon entropy we used in MaxEnt-IRL formulation and ψ is the regularizer imposed on the reward model r .

Would occupancy matching find Q that satisfies the Bellman equation? Denote the policy as π ∗ and its corresponding discounted state-action occupancy measure as d ∗ = (1 -β ) ∑ ∞ t =0 β t d ∗ t , with d ∗ t ( s, a ) = P π ∗ ,ν 0 [ s t = s, a t = a ] . Wedefine the expert's action-value function as Q ∗ ( s, a ) := E π ∗ [∑ ∞ t =0 β t r ( s t , a t ) | s 0 = s, a 0 = a ] and the Bellman operator of π ∗ as T ∗ . Then we have the following Lemma 17 showing that occupancy matching (even without regularization) may not minimize Bellman error for every state and action.

Lemma 17 (Occupancy matching is equivalent to naive weighted Bellman error sum) . The perfect occupancy matching given the same ( s 0 , a 0 ) satisfies

<!-- formula-not-decoded -->

Proof. Note that E ( s,a ) ∼ d ∗ [ r ( s, a ) | s 0 , a 0 ] = ∑ ∞ t =0 β t E ( s,a ) ∼ d ∗ t [ r ( s, a ) | s 0 , a 0 ] = Q ∗ ( s, a ) and E ( s,a ) ∼ d π [ r ( s , a ) | s 0 , a 0 ] = ∑ ∞ t =0 β t E ( s,a ) ∼ d π t [ r ( s, a ) | s 0 , a 0 ] = Q π ( s, a ) . Therefore

<!-- formula-not-decoded -->

Lemma 17 implies that occupancy measure matching, even without reward regularization, does not necessarily imply Bellman errors being 0 for every state and action. In fact, what they minimize is the average Bellman error Jiang et al. (2017); Uehara et al. (2020). This implies that r cannot be inferred from Q using the Bellman equation after deriving Q using occupancy matching.

Lemma 18 (Bellman Error Telescoping) . Let the Bellman operator T π is defined to map f ∈ R S × A to

T π f := r ( s, a ) + E s ′ ∼ P ( s,a ) ,a ′ ∼ π ( ·| s ′ ) [ f ( s ′ , a ′ ) | s, a ] . For any π , and any f ∈ R S × A ,

<!-- formula-not-decoded -->

Proof. Note that the right-hand side of the statement can be expanded as

<!-- formula-not-decoded -->

which is the left-hand side of the statement.

## C Technical Proofs

## C.1 Theory of TD correction using biconjugate trick

Proof of Lemma 4.

<!-- formula-not-decoded -->

where the unique maximum is with

<!-- formula-not-decoded -->

and where the equality of A15 is from

<!-- formula-not-decoded -->

Now note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the equality of Equation (A17) comes from the fact that the ζ that maximize Equation (A16) is ζ ∗ := E s ′ ∼ P ( s,a ) [ ˆ V ( s ′ ) | s, a ] , because

<!-- formula-not-decoded -->

For Q ∗ , T Q ∗ = Q ∗ holds. Therefore, we get

<!-- formula-not-decoded -->

## C.2 Proof of Lemma 1

Proof. Suppose that the system of equations (Equation (5))

<!-- formula-not-decoded -->

is satisfied for Q ∈ Q , where Q denote the space of all Q functions. Then we have the following equivalent recharacterization of the second condition ∀ s ∈ S ,

<!-- formula-not-decoded -->

We will now show the existence and uniqueness of a solution using a standard fixed point argument on a Bellman operator. Let F be the space of functions f : S → R induced by elements of Q , where each Q ∈ Q defines an element of F via

<!-- formula-not-decoded -->

and define an operator T f : F → F that acts on functions f Q :

<!-- formula-not-decoded -->

Then for Q 1 , Q 2 ∈ Q , We have

<!-- formula-not-decoded -->

Subtracting the two, we get

<!-- formula-not-decoded -->

Taking supremum norm over s ∈ S , we get

<!-- formula-not-decoded -->

This implies that T f is a contraction mapping under supremum norm, with β ∈ (0 , 1) . Since Q is a Banach space under sup norm (Lemma 19), we can apply Banach fixed point theorem to show that there exists a unique f Q that satisfies T f ( f Q ) = f Q , and by definition of f Q there exists a unique Q that satisfies T f ( f Q ) = f Q , i.e.,

<!-- formula-not-decoded -->

Since Q ∗ satisfies the system of equations (5), Q ∗ is the only solution to the system of equations.

Also, since Q ∗ = T Q ∗ = r ( s, a ) + β · E s ′ ∼ P ( s,a ) [ log( ∑ a ′ ∈A exp Q ∗ ( s ′ , a ′ )) | s, a ] holds, we can identify r as

<!-- formula-not-decoded -->

Lemma 19. Suppose that Q consists of bounded functions on S × A . Then Q is a Banach space with the supremum norm as the induced norm.

Proof. Suppose a sequence of functions { Q n } in Q is Cauchy in the supremum norm. We must show that Q n → Q ∗ as n → ∞ for some Q ∗ and Q ∗ is also bounded. Note that Q n being Cauchy in sup norm implies that for every ( s, a ) , the sequence { Q n ( s, a ) } is Cauchy in R . Since R is a complete space, every Cauchy sequence of real numbers has a limit; this allows us to define function Q ∗ : S × A ↦→ R such that Q ∗ ( s, a ) = lim n →∞ Q n ( s, a ) . Then we can say that Q n ( s, a ) → Q ∗ ( s, a ) for every ( s, a ) ∈ S × A . Since each Q n is bounded, we take the limit and obtain:

<!-- formula-not-decoded -->

which implies Q ∗ ∈ Q .

Now what's left is to show that the supremum norm

<!-- formula-not-decoded -->

induces the metric, i.e.,

<!-- formula-not-decoded -->

The function d satisfies the properties of a metric:

- Non-negativity: d ( Q 1 , Q 2 ) ≥ 0 and d ( Q 1 , Q 2 ) = 0 if and only if Q 1 = Q 2 .
- Symmetry: d ( Q 1 , Q 2 ) = d ( Q 2 , Q 1 ) by the absolute difference.
- Triangle inequality:

<!-- formula-not-decoded -->

which shows d ( Q 1 , Q 3 ) ≤ d ( Q 1 , Q 2 ) + d ( Q 2 , Q 3 ) .

## C.3 Proof of Theorem 3

Define ˆ Q as

<!-- formula-not-decoded -->

From Lemma 1, it is sufficient to show that ˆ Q satisfies the equations (5) of Lemma 1, i.e.,

<!-- formula-not-decoded -->

where ¯ S (the reachable states from ν 0 , π ∗ ) was defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now note that:

and

Therefore what we want to prove, Equations (5), becomes the following Equation (6):

<!-- formula-not-decoded -->

̸

Under this non-emptiness, according to Lemma 21, ˆ Q satisfies Equation (6). This implies that ˆ Q ( s, a ) = Q ∗ ( s, a ) for s ∈ ¯ S and a ∈ A , as the solution to set of Equations (5) is Q ∗ . This implies that where its solution set is nonempty by Lemma 1, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for s ∈ ¯ S and a ∈ A .

## Lemma 20.

<!-- formula-not-decoded -->

## Proof of Lemma 20.

<!-- formula-not-decoded -->

□

Therefore,

<!-- formula-not-decoded -->

Lemma 21. Let f 1 : X → R and f 2 : X → R be two functions defined on a common domain X . Suppose the sets of minimizers of f 1 and f 2 intersect, i.e.,

̸

<!-- formula-not-decoded -->

Then, any minimizer of the sum f 1 + f 2 is also a minimizer of both f 1 and f 2 individually. That is, if

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

Proof. Since arg min f 1 ∩ arg min f 2 = ∅ , let x † be a common minimizer such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now, let x ∗ be any minimizer of f 1 + f 2 , so

<!-- formula-not-decoded -->

then

This implies that

Evaluating this at x † , we obtain

But then

Thus, we conclude

## C.4 Proof of Lemma 6

For the given dataset D , define

<!-- formula-not-decoded -->

Now, suppose for contradiction that x ∗ / ∈ arg min f 1 , meaning

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This contradicts the fact that m 2 = min f 2 , so x ∗ must satisfy

<!-- formula-not-decoded -->

By symmetry, assuming x ∗ / ∈ arg min f 2 leads to the same contradiction, forcing

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that DQ θ = M . Without loss of generality, we can assume that ϕ is isotropic ( E [ ϕ ] = 0 , E [ ϕϕ ⊤ ] = I d ) ; otherwise, define

<!-- formula-not-decoded -->

Then ˜ ϕ is isotropic ( E [ ˜ ϕ ] = 0 , E [ ˜ ϕ ˜ ϕ ⊤ ] = I d ) and satisfies

<!-- formula-not-decoded -->

because λ min (Σ) = λ ∗ and ∥ ϕ ∥ 2 ≤ B .

Instantiate

<!-- formula-not-decoded -->

Subtracting the deterministic rank-one term cannot decrease the smallest singular value, so

<!-- formula-not-decoded -->

Now let's consider the function class

Therefore we have

<!-- formula-not-decoded -->

where ϕ : S × A → R d is a known feature map with ∥ ϕ ( s, a ) ∥ ≤ B almost surely and θ ∈ R d is the parameter vector. Then for any unit vector u ∈ R d ,

<!-- formula-not-decoded -->

Then by using Hoeffding's Lemma, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then by Rudelson and Vershynin (2009), we have

<!-- formula-not-decoded -->

provided that the dataset size satisfies |D| ≥ Cd with C &gt; 1 .

## C.5 Proof of Lemma 8 (Bellman error terms satisfying the PL condition)

By Lemma 22, for all s ∈ S and a ∈ A , L BE ( Q θ )( s, a ) satisfies PL condition with respect to θ in terms of euclidean norm. Now we would like to show that L BE ( Q θ ) := E ( s,a ) ∼ π ∗ ,ν 0 [ L BE ( Q θ )( s, a )] is PL with respect to θ .

First, L BE ( Q θ )( s, a ) is of C 2 because

L ( Q )( s, a ) is of C 2

<!-- formula-not-decoded -->

By Rebjock and Boumal (2023), L BE ( Q θ )( s, a ) being of C 2 implies that the PL condition is equivalent to the Quadratic Growth (QG) condition, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

L BE ( Q θ ) is of C 2 ( ∵ Lemma 23). This implies that L BE ( Q θ ) satisfies QG condition with respect to θ , which is equivalent to L BE ( Q θ ) satisfying PL condition with respect to θ . □

Lemma 22. Suppose that Assumption 5 holds. Then for any given fixed s ∈ S and a ∈ A , L BE ( Q θ )( s, a ) satisfies PL condition with respect to θ in terms of euclidean norm.

Proof of Lemma 22. Define

<!-- formula-not-decoded -->

Also define

Therefore,

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Now what's left is lower bounding ∥∇ θ Ψ θ ( s, a ) ∥ 2 2 . Let ( s, a ) ∈ S × A be fixed. We consider the scalar function θ ↦→ Ψ θ ( s, a ) . We want to compute its gradient with respect to θ . Define the evaluation functional δ ( s,a ) : C ( S × A ) → R such that δ ( s,a ) [ Q ] = Q ( s, a ) and

<!-- formula-not-decoded -->

Using the chain rule for differentiation, the gradient ∇ θ Ψ θ ( s, a ) can be expressed using the Fr´ echet derivatives of the involved maps. Let G ( θ ) = Q θ and H ( Q ) = T Q -Q . Then Ψ θ ( s, a ) = δ ( s,a ) [ H ( G ( θ ))] and the derivative of G at θ is DG θ = DQ θ . The Fr´ echet derivative of H at Q θ , denoted DH Q θ : C ( S × A ) → C ( S × A ) , is given by DH Q θ = J T ( Q θ ) -I , where J T ( Q θ ) is the Fr´ echet derivative of the soft Bellman operator T evaluated at Q θ , and I is the identity operator on C ( S × A ) .

The gradient of the scalar function Ψ θ ( s, a ) with respect to θ ∈ R d θ is given by the action of the adjoint of the composition DH Q θ ◦ DQ θ on the evaluation functional δ ( s,a ) .

<!-- formula-not-decoded -->

Using the property ( A ◦ B ) ⊤ = B ⊤ ◦ A ⊤ , where ⊤ denotes the adjoint operator:

<!-- formula-not-decoded -->

Substituting DH Q θ = J T ( Q θ ) -I :

<!-- formula-not-decoded -->

Let J ⊤ T denote the adjoint of J T ( Q θ ) and I ⊤ = I .

<!-- formula-not-decoded -->

Now, we take the squared Euclidean norm ( ℓ 2 norm in R d θ ):

<!-- formula-not-decoded -->

We use the property that for any compatible matrix (or operator) M and vector v, ∥ Mv ∥ 2 ≥ σ min ( M ) ∥ v ∥ 2 .

Applying this with M = DQ ⊤ θ :

<!-- formula-not-decoded -->

where ∥ · ∥ H ∗ denotes the norm in the dual space to H = C ( S × A ) appropriate for defining the adjoints and singular values. From the assumption, we can rewrite this as

<!-- formula-not-decoded -->

Now what's left is lower bounding and therefore

<!-- formula-not-decoded -->

As we discussed above, T is a contraction with respect to the supremum norm ∥ · ∥ ∞ on C ( S × A ) . This means its Fr´ echet derivative J T ( Q θ ) satisfies ∥ J T ( Q θ ) ∥ op ≤ β , where ∥ · ∥ op is the operator norm induced by the ∥ · ∥ ∞ norm. The operator I -J T ( Q θ ) is therefore invertible on C ( S × A ) , and its inverse can be expressed as a Neumann series ∑ ∞ k =0 ( J T ( Q θ )) k . The operator norm of the inverse is bounded by ∥ ( I -J T ( Q θ )) -1 ∥ op ≤ 1 1 -∥ J T ( Q θ ) ∥ op ≤ 1 1 -β . The minimum singular value of an invertible operator K is related to the operator norm of its inverse: σ min ( K ) = 1 / ∥ ∥ K -1 ∥ ∥ op ∗ . Therefore, σ min ( I -J T ( Q θ ) ⊤ ) = 1 / ∥ ∥ ∥ ∥ ( I -J T ( Q θ ) ⊤ ) -1 ∥ ∥ ∥ ∥ op ∗ ≥ 1 -β . Now we can lower bound the norm:

<!-- formula-not-decoded -->

where the last inequality comes from assuming the dual norm is normalized such that ∥ ∥ δ ( s,a ) ∥ ∥ H ∗ = 1 . 16 Substituting back, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

16 Here, H ∗ = ( C ( S × A )) ∗ is the space of Radon measures endowed with the total-variation norm, and δ ( s,a ) is the Dirac point-mass.

## C.6 Proof of Lemma 9 (NLL loss satisfying the PL condition)

First, note that γ ∈ (0 , 1) and the Bellman equation holds. Therefore, there exist Q min , Q max ∈ R such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let be the softmax probability. The NLL loss is

<!-- formula-not-decoded -->

The functional gradient of f s,a w.r.t. Q θ , denoted G θ ∈ ( C ( S × A )) ∗ , corresponds to

<!-- formula-not-decoded -->

So G θ is a signed measure supported on { s } × A , and its total variation norm is

̸

<!-- formula-not-decoded -->

using ∑ b π θ,b = 1 . By the chain rule:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then:

and squaring both sides yields:

Now define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This function is continuous, strictly positive on the interval, and

<!-- formula-not-decoded -->

So,

Therefore

<!-- formula-not-decoded -->

Combining ( a ) and ( b ) , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This proves the PL inequality with µ PL = 2 µc ψ &gt; 0 . That is, L NLL ( Q θ ) ( s, a ) satisfies the PL condition with respect to θ .

Now we would like to show that L NLL ( Q θ ) satisfies the PL condition with respect to θ in terms of ℓ 2 norm. First, L NLL ( Q θ ) ( s, a ) is C 2 because

<!-- formula-not-decoded -->

By Rebjock and Boumal (2023), L NLL ( Q θ )( s, a ) being of C 2 implies that the PL condition is equivalent to the Quadratic Growth (QG) condition, i.e.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now note that L NLL ( Q θ ) ( s, a ) being C 2 implies that L NLL ( Q θ ) being C 2 due to Lemma 23. This implies that L NLL ( Q θ ) satisfies QG condition with respect to θ , which is equivalent to L NLL ( Q θ ) satisfying PL condition with respect to θ . □

Therefore,

## Lemma 23. Let

<!-- formula-not-decoded -->

where Θ ⊂ R d is an open set. For every ( s, a ) , the map θ ↦→ Q θ ( s, a ) is C 2 on Θ .

Suppose the following assumptions hold:

State and Action Spaces. The state space S is a compact topological space and the action set A is finite.

Uniform C 2 Bounds. The map ( s, a, θ ) ↦→ Q θ ( s, a ) and its first two derivatives with respect to θ are continuous on S × A × Θ . Furthermore, there exist constants B &gt; 0 and B 2 &gt; 0 such that for all θ ∈ Θ :

<!-- formula-not-decoded -->

Define the averaged loss as L NLL ( θ ) := E ( s,a ) ∼ µ [ f s,a ( θ ) ] , where µ is any probability measure on the product space S × A .

Then, L NLL is twice continuously differentiable on Θ , and its derivatives are given by

<!-- formula-not-decoded -->

for every θ ∈ Θ .

Proof of Lemma 23. We establish uniform integrable bounds on the first and second derivatives of f s,a ( θ ) to justify invoking the Leibniz integral rule for differentiating under the expectation.

First, observe that for any fixed ( s, a ) ∈ S × A , the map θ ↦→ f s,a ( θ ) is C 2 (Θ) . This is a direct consequence of the assumption that θ ↦→ Q θ ( s, a ) is C 2 and the fact that the log-sum-exp function is analytic, hence smooth.

Let π θ ( b | s ) := exp( Q θ ( s, b )) / ∑ c ∈A exp( Q θ ( s, c )) be the softmax probability. The first derivative of f s,a ( θ ) is

<!-- formula-not-decoded -->

Using the triangle inequality and the uniform bound on the gradient of Q θ , we obtain a uniform bound on the norm of the gradient of f s,a ( θ ) :

<!-- formula-not-decoded -->

Let M 1 := 2 B . This bound is independent of θ , s , and a .

The Hessian of f s,a ( θ ) is given by

<!-- formula-not-decoded -->

The second term is the Hessian of a log-partition function, which is the covariance matrix of the gradients ∇ θ Q θ ( s, b ) with respect to the distribution π θ ( ·| s ) . Specifically,

<!-- formula-not-decoded -->

We bound the operator norm of the Hessian of f s,a ( θ ) using the triangle inequality and properties of the operator norm:

<!-- formula-not-decoded -->

where X b := ∇ θ Q θ ( s, b ) . The first two terms are bounded by B 2 . For the covariance term, we note that ∥ X b ∥ 2 ≤ B , so ∥ X b X ⊤ b ∥ op = ∥ X b ∥ 2 2 ≤ B 2 . Thus, ∥ E [ X b X ⊤ b ] ∥ op ≤ E [ ∥ X b X ⊤ b ∥ op ] ≤ B 2 , and since the covariance is positive semidefinite, its operator norm is bounded by the maximum eigenvalue, which is in turn bounded by the trace. A simpler bound is ∥ Cov ( X ) ∥ op ≤ E [ ∥ X ∥ 2 2 ] ≤ B 2 . This yields a uniform bound:

<!-- formula-not-decoded -->

Let M 2 := 2 B 2 + B 2 . This constant is also independent of θ , s , and a .

The functions θ ↦→ f s,a ( θ ) , θ ↦→ ∇ θ f s,a ( θ ) , and θ ↦→ ∇ 2 θ f s,a ( θ ) are continuous on Θ for each ( s, a ) . The norms of the first and second derivatives are dominated by the constants M 1 and M 2 , respectively. Since constants are integrable with respect to any probability measure µ , the conditions for the Leibniz integral rule are satisfied. Therefore, L NLL is twice differentiable and we can interchange differentiation and expectation:

<!-- formula-not-decoded -->

To establish that L NLL is C 2 , we must show the Hessian is continuous. By the 'Uniform C 2 Bounds' assumption, the map ( θ, s, a ) ↦→∇ 2 θ f s,a ( θ ) is continuous on Θ ×S × A . Let ( θ n ) n ∈ N be a sequence in Θ converging to θ ∈ Θ . Then for each ( s, a ) , ∇ 2 θ f s,a ( θ n ) →∇ 2 θ f s,a ( θ ) as n →∞ . The sequence of functions g n ( s, a ) := ∇ 2 θ f s,a ( θ n ) is uniformly dominated in norm by the µ -integrable constant M 2 . By the Dominated Convergence Theorem, applied component-wise to the Hessian matrix,

<!-- formula-not-decoded -->

This confirms the continuity of the Hessian at any θ ∈ Θ . The same argument with the dominating function M 1 shows the gradient is continuous. Therefore, L NLL is twice continuously differentiable on Θ .

## C.7 Proof of Theorem 10

We first prove that the expected risk (Equation (17)) satisfies the PL condition in terms of θ . When we were proving Theorem 3 in Appendix C.3, we saw that true Q is identified by solving the following system of equations:

<!-- formula-not-decoded -->

Also, we saw that both E ( s,a ) ∼ π ∗ ,ν 0 [ -log (ˆ p Q ( · | s ))] and E ( s,a ) ∼ π ∗ ,ν 0 [ ✶ a = a s L BE ( Q )( s, a )] are of C 2 when we proved Lemma 8 and 9. Therefore, by Lemma 25, we conclude that the sum of the two terms, the expected risk, is PL in terms of θ .

Now we prove that the empirical risk (Equation (18)) satisfies the PL condition in terms of θ . Note that when we were proving that the expected risk

<!-- formula-not-decoded -->

is PL in terms of θ in previous steps, the specific π ∗ , ν 0 , and P did not matter. That is, if we used ˜ π , ˜ ν , and ˜ P instead of π ∗ , ν 0 , and P in the above proof procedures,

<!-- formula-not-decoded -->

must be PL in terms of θ . Now let's choose ˜ π , ˜ ν , and ˜ P as empirical distributions from dataset D . Then the above equation becomes

<!-- formula-not-decoded -->

which is the empirical risk R emp ( Q ) . Lastly, by Lemma 24, E ( s,a ) ∼ ˜ π, ˜ ν,s ′ ∼ ˜ P ( s,a ) [ ( V Qθ 2 ( s ′ ) -ζ θ 1 ( s, a )) 2 ] satisfies PL condition with respect to θ 1 . Therefore, We can conclude that R emp ( Q ) satisfies PL condition for both θ 1 and θ 2 . □

Lemma 24 (PL condition w.r.t. θ 1 ) . Assume the parametric function class { ζ θ 1 : θ 1 ∈ Θ ζ } satisfies the same

Assumption 5 that Q θ 2 satisfies. Define the (negative) inner objective

<!-- formula-not-decoded -->

Then there exists a constant µ ζ = 2 β 2 µ &gt; 0 such that

<!-- formula-not-decoded -->

where θ ∗ 1 maximises L ζ ( · ; θ 2 ) for this fixed θ 2 . The empirical counterpart

<!-- formula-not-decoded -->

satisfies the same µ ζ -PL inequality.

Proof. The lemma states that the function L ζ ( θ 1 ; θ 2 ) satisfies the Polyak-Łojasiewicz (PL) condition for maximization with respect to θ 1 . This is equivalent to showing that its negative, let's define it as F ( θ 1 ) , satisfies the PL condition for minimization.

<!-- formula-not-decoded -->

We will prove that F ( θ 1 ) satisfies the PL condition. The logic applies identically to the empirical version, ̂ F ( θ 1 ) = -̂ L ζ ( θ 1 ; θ 2 ) , by replacing the expectation E ( s,a,s ′ ) ∼ D with the empirical mean 1 N ∑ ( s,a,s ′ ) ∈D .

The proof consists of three steps. First, we express the gradient using the adjoint of the Jacobian operator from Assumption 5 Second, we relate the function suboptimality to the error in function space. Finally, we combine these results to derive the PL inequality.

## 1. Expressing the Gradient

Let θ ∗ 1 be the minimizer of F ( θ 1 ) . The optimal function ζ θ ∗ 1 ( s, a ) that minimizes the mean squared error is the conditional expectation, ζ θ ∗ 1 ( s, a ) = E s ′ | s,a [ V Q θ 2 ( s ′ )] . We define the error as ∆ θ 1 ( s, a ) := ζ θ 1 ( s, a ) -ζ θ ∗ 1 ( s, a ) . Then the gradient of F ( θ 1 ) can be written as

<!-- formula-not-decoded -->

Using the adjoint of the Jacobian operator from Assumption 5, ( Dζ θ 1 ) ∗ g = E [ g ( s, a ) ∇ θ 1 ζ θ 1 ( s, a )] , we can write the gradient as:

<!-- formula-not-decoded -->

## 2. Expressing the Suboptimality

The suboptimality gap F ( θ 1 ) -F ( θ ∗ 1 ) can be expressed as the squared error in function space.

<!-- formula-not-decoded -->

The cross-term vanishes because ζ θ ∗ 1 is the conditional expectation. The suboptimality is thus the squared L 2 norm of the error, denoted ∥ ∆ θ 1 ∥ 2 L 2 :

<!-- formula-not-decoded -->

## 3. Deriving the PL Condition

We connect the gradient norm to the suboptimality using the property ∥ A ∗ v ∥ ≥ σ min ( A ) ∥ v ∥ .

<!-- formula-not-decoded -->

Assumption 5 states that σ min ( Dζ θ 1 ) ≥ √ µ . Substituting this in gives:

<!-- formula-not-decoded -->

Now, we replace the error norm with the suboptimality from Step 2:

<!-- formula-not-decoded -->

Finally, dividing by 2 gives the PL condition for F ( θ 1 ) = -L ζ :

<!-- formula-not-decoded -->

This is the PL inequality with a PL constant of µ ζ = 2 β 2 µ &gt; 0 .

Lemma 25. Suppose that f 1 and f 2 are both PL in θ and of C 2 . Furthermore, the minimizer of f 1 + f 2 lies in the intersection of minimizer of f 1 and the minimizer f 2 . Then f 1 + f 2 satisfies PL condition.

Proof of Lemma 25. Recall that we say f satisfies µ -PL condition if 2 µ ( f ( θ ) -f ( θ ∗ )) ≤ ∥∇ f ( θ ) ∥ 2 .

<!-- formula-not-decoded -->

The last inequality is not trivial, and therefore requires Lemma 26.

Lemma 26. Suppose that f 1 and f 2 are both PL in θ , of C 2 and minimizer of f 1 + f 2 lies in the intersection of the minimizer of f 1 and the minimizer f 2 . Then for all θ ∈ Θ , ⟨∇ f 1 ( θ ) , ∇ f 2 ( θ ) ⟩ ≥ 0 .

Proof. Let M 1 := { θ ∈ Θ : f 1 ( θ ) = f ∗ 1 } , and M 2 := { θ ∈ Θ : f 2 ( θ ) = f ∗ 2 } . From what is assumed, f 1 + f 2 has a minimizer θ ∗ that belongs to both M 1 and M 2 .

Since f 1 and f 2 are both of C 2 and satisfy PL condition, they both satisfy Quadratic Growth (QG) condition (Liao et al. 2024), i.e., there exists α 1 , α 2 &gt; 0 such that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now suppose, for the purpose of contradiction, that there exists ˆ θ ∈ Θ such that 〈 ∇ f 1 ( ˆ θ ) , ∇ f 2 ( ˆ θ ) 〉 &lt; 0 . Consider the direction d := -g 1 = -∇ f 1 ( ˆ θ ) . Then ∇ f 1 ( ˆ θ ) ⊤ d = g ⊤ 1 ( -g 1 ) = -∥ g 1 ∥ 2 &lt; 0 holds. This implies that f 1 ( ˆ θ + ηd ) &lt; f 1 ( ˆ θ ) . Then QG condition for f 1 implies that

<!-- formula-not-decoded -->

Now, note that ∇ f 2 ( ˆ θ ) ⊤ d = g ⊤ 2 ( -g 1 ) = -g ⊤ 1 g 2 . Since g ⊤ 1 g 2 &lt; 0 , ∇ f 2 ( ˆ θ ) ⊤ d &gt; 0 . Therefore, f 2 ( ˆ θ + ηd ) &gt; f 2 ( ˆ θ ) for sufficiently small η &gt; 0 . That is, f 2 ( ˆ θ + ηd ) -f 2 ( θ ∗ ) &gt; f 2 ( ˆ θ ) -f 2 ( θ ∗ ) . By QG condition, this implies that ∥ ∥ ∥ ˆ θ + ηd -θ ∗ ∥ ∥ ∥ &gt; ∥ ∥ ∥ ˆ θ -θ ∗ ∥ ∥ ∥ . Contradiction.

## C.8 Proof of Theorem 11 (Global optima convergence under ERM-IRL)

## 1. Optimization error analysis

In Theorem 5, we proved that the empirical risk (Equation (12)), which is minimized by Algorithm 1, satisfies PL in terms of θ . Since the inner maximization problem of the empirical risk is trivially strongly

convex, which implies PL, the empirical risk satisfies a two-sided PL (Yang et al. 2020). Denote Equation (12) as f ( θ , ζ ) , and define g , g ∗ as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any ( θ , ζ ) . Furthermore, they are both equal to 0 if and only if ( θ , ζ ) is a minimax point, which is θ ∗ and ζ ∗ . More precisely, we have

<!-- formula-not-decoded -->

Therefore, we would like to find θ , ζ that a ( θ ) + αb ( θ , ζ ) = 0 for α &gt; 0 , where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

At iteration 0, algorithm starts from ˆ θ 0 and ζ = ζ 0 . We denote the θ , ζ value at the Algorithm iteration T as ˆ θ T and ˆ ζ T . Also, we define P T as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now define

Denote that f ( θ , ζ ) satisfies µ θ -PL for θ and µ ζ -PL for ζ . Let α = 1 / 10 , τ T 1 = c γ + T , τ T 2 = 18 l 2 c µ 2 η ( γ + T ) for some c &gt; 2 /µ θ , and γ &gt; 0 such that τ 1 1 ≤ min { 1 /L, µ 2 η / 18 l 2 } . Then the following Theorem holds.

Theorem 27 (Theorem 3.3 of Yang et al. (2020)) . Applying Algorithm 1 using stochastic gradient ascent

Note that

descent (SGAD), P T satisfies

<!-- formula-not-decoded -->

Therefore, a ( ˆ θ T ) ≤ P T ≤ ν γ + T . Note that a ( θ ) satisfies the PL condition with respect to θ and of C 2 since subtracting a constant from PL is PL. Therefore, a ( θ ) satisfies Quadratic Growth (QG) condition by Liao et al. (2024), i.e.,

<!-- formula-not-decoded -->

where S D N is the set of minimizers of empirical risk, and dist ( θ , S ) := min θ ′ ∈ S ∥ θ -θ ′ ∥ . Therefore, we can conclude that dist 2 ( ˆ θ T , S D N ) ≤ O (1 /T ) , or dist ( ˆ θ T , S D N ) is O (1 / √ T ) .

## 2. Global convergence analysis.

The challenge of statistical convergence analysis using data from a Markov Chain, which is noni.i.d sampled data, is well known (Mohri and Rostamizadeh 2010; Fu et al. 2023; Abeles et al. 2024). The following Lemma 28 by Kang and Jang (2025) shows that applying Algorithm 1 for Bellman residual minimization has sample complexity of O (1 /N ) .

Lemma 28 (Sample complexity of Bellman residual minimization using SGDA (Kang and Jang 2025)) . ( ̂ θ ( T ) 1 , ̂ θ ( T ) 2 ) be the parameters returned by ALGORITHM 1 after T SGDA iterations on the empirical objective (15) with stepsizes η t = c 1 c 2 + T . Define the population risk

<!-- formula-not-decoded -->

where ˜ θ ⋆ 1 is the minimizer in

<!-- formula-not-decoded -->

Let Θ ∗ 2 := argmin θ 2 ∈ Θ R ( θ 2 ) denote the (possibly non-singleton) set of population minimizers, and fix any θ ∗ 2 ∈ Θ ∗ 2 . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

Here, L is the Lipschitz smoothness constant from Lemma 29, ρ is the strongly concavity constant, G

is the upper bound on the norm of the stochastic gradients, c 1 and c 2 are from step-size choice, c is min( µ PL / 2 , ρ/ 2) , and C is a constant defined in Kang and Jang (2025).

Using E [ R ( ̂ θ ( T ) 2 ) -R ( θ ∗ 2 ) ] ≤ (1 + L/ρ ) Gε T + ν γ + T and the fact that R satisfies PL condition, which is equivalent to QG condition under C 2 , we arrive at the final global convergence analysis result (to the solution set):

<!-- formula-not-decoded -->

for some C ′ , where dist( θ, S ) := min θ ′ ∈ S ∥ θ -θ ′ ∥ 2 .

Next, we convert the above bound into the bound in terms of E D N [ ∥ ∥ ∥ Q θ ⋆ 2 -Q ̂ θ ( T ) 2 ∥ ∥ ∥ 2 ∞ ] . The key idea is to utilize sup ( s,a ) ∈S×A ∥∇ θ Q θ ( s, a ) ∥ 2 ≤ B from Assumption 5. Fix any θ 1 , θ 2 ∈ Θ and any ( s, a ) . Consider the line segment

<!-- formula-not-decoded -->

By the fundamental theorem of calculus (equivalently, the mean value theorem in integral form),

<!-- formula-not-decoded -->

Take absolute values and apply Cauchy-Schwarz:

<!-- formula-not-decoded -->

where the both inequalities are from sup ( s,a ) ∈S×A ∥∇ θ Q θ ( s, a ) ∥ 2 ≤ B of Assumption 5. Now take supremum over ( s, a ) :

<!-- formula-not-decoded -->

So the map θ ↦→ Q θ is B -Lipschitz from (Θ , ∥ · ∥ 2 ) into ( S × A → R , ∥ · ∥ ∞ ) . Therefore, for any θ ∗ ∈ Θ ∗ 2 ,

<!-- formula-not-decoded -->

Take inf over θ ∗ ∈ Θ ∗ 2 on both sides:

<!-- formula-not-decoded -->

Since

Therefore, we have

Take expectations and squares:

<!-- formula-not-decoded -->

Therefore, we achieve

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then the set { Q θ ∗ : θ ∗ ∈ Θ ∗ 2 } is a singleton { Q ∗ } . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

□

Lemma 29 (Lipschitzness of R emp) . Suppose that ∥∇ θ Q θ ∥ 2 ≤ B and M =: ( R max + βV max + Q max ) is defined as above. Then R emp ( θ ) is Lipschitz continuous with respect to θ with constant (1 + 2 M (1 + β )) B .

Proof. If ∇ θ L NLL ( Q θ ) ( s, a ) and ∇ θ L BE ( Q θ ) ( s, a ) is bounded, then

<!-- formula-not-decoded -->

which implies that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∇ θ Ψ θ ( s, a ) = ∇ θ ( T Q θ ( s, a )) -∇ θ Q θ ( s, a ) . Since gradient of Q θ is bounded, we have ∇ θ ( T Q θ ( s, a )) = ∇ θ ( r ( s, a ) + β E s ′ [ V Q θ ( s ′ )]) = β E s ′ [ ∇ θ V Q θ ( s ′ )] , where ∇ θ V Q θ ( s ′ ) = ∑ c ∈A exp( Q θ ( s ′ ,c )) ∑ b ∈A exp( Q θ ( s ′ ,b )) ∇ θ Q θ ( s ′ , c ) = ∑ c ∈A ˆ p Q θ ( c | s ′ ) ∇ θ Q θ ( s ′ , c ) ≤ B ( ∵ ∥∇ θ Q θ ∥ 2 ≤ B due to Assumption 5). Now, applying Jensen's inequality for norms ( ∥ E [ X ] ∥ ≤ E [ ∥ X ∥ ]) :

<!-- formula-not-decoded -->

Now note that

Therefore

Also, note that

<!-- formula-not-decoded -->

where its gradient is ∇ θ L NLL ( s, a ) = -∇ θ Q θ ( s, a ) + ∑ b ∈A ˆ p Q θ ( b | s ) ∇ θ Q θ ( s, b ) . By Assumption 6.2, ∥∇ θ Q θ ( s, a ) ∥ 2 ≤ B . Since ∑ b ˆ p Q θ ( b | s ) = 1 , the triangle inequality yields

<!-- formula-not-decoded -->

Since E [ ✶ a = a s ] ≤ 1 , we have

<!-- formula-not-decoded -->

By the Mean Value Theorem, for any θ 1 , θ 2 ∈ Θ , there exists ˜ θ on the line segment connecting θ 1 and θ 2 such that R exp ( θ 1 ) -R exp ( θ 2 ) = ∇ θ R exp ( ˜ θ ) · ( θ 1 -θ 2 ) . Taking the absolute value and applying the Cauchy-Schwarz inequality:

<!-- formula-not-decoded -->

Since ∥ ∥ ∥ ∇ θ R exp ( ˜ θ ) ∥ ∥ ∥ 2 ≤ L , we conclude:

<!-- formula-not-decoded -->

Therefore, R exp ( θ ) is Lipschitz continuous with constant (1 + 2 M (1 + β )) B . With the exactly same logic introduced in the Lemma C.7, this implies that R emp ( θ ) is also Lipschitz continuous with respect to θ with constant (1 + 2 M (1 + β )) B .

Lemma 30 (Lipschitz smoothness of R emp) . We continue from what we derived during the proof of Lemma 29. If ∇ θ L NLL ( Q θ ) ( s, a ) and ∇ θ L BE ( Q θ ) ( s, a ) is bounded, then

<!-- formula-not-decoded -->

Recall that ∇ θ L NLL ( s, a ) = -∇ θ Q θ ( s, a ) + ∇ θ V Q θ ( s ) where ∇ θ V Q θ ( s ) = ∑ b ∈A ˆ p Q θ ( b | s ) ∇ θ Q θ ( s, b ) and ∥∇ θ L NLL ( s, a ) ∥ 2 ≤ 2 B . With ∥ ∥ ∇ 2 θ Q θ ( s, a ) ∥ ∥ 2 ≤ B 2 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∇ θ p θ ( b | s ) = p θ ( b | s ) ( ∇ θ Q θ ( s, b ) -∑ c p θ ( c | s ) ∇ θ Q θ ( s, c )) . Distributing, we have

<!-- formula-not-decoded -->

Now we have where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For T 1 and T 2 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For T 3 , from ∥ ∥ ∇ θ Q θ ( s, b ) ∇ θ Q θ ( s, b ) ⊤ ∥ ∥ 2 = ∥∇ θ Q θ ( s, b ) ∥ 2 2 ≤ B 2 , we have

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

Also, continuing from

<!-- formula-not-decoded -->

where ∇ θ Ψ θ ( s, a ) = ∇ θ ( T Q θ ( s, a )) -∇ θ Q θ ( s, a ) = β E s ′ [ ∇ θ V Q θ ( s ′ )] -∇ θ Q θ ( s, a ) , we have

<!-- formula-not-decoded -->

From ∇ 2 θ Ψ θ ( s, a ) = β E s ′ [ ∇ 2 θ V Q θ ( s ′ ) | s, a ] -∇ 2 θ Q θ ( s, a ) where ∇ 2 θ V Q θ ( s ) = T 2 + T 3 . Therefore

<!-- formula-not-decoded -->

Therefore, we get

<!-- formula-not-decoded -->

For ∇ 2 ζ R exp ( ζ ) , it is a simple squared function and therefore the Lipschitz smoothness in β is trivial. Now with the same mean value theorem argument as in Lemma 29, we have that

<!-- formula-not-decoded -->

where L θ = 2 B 2 + |A| 4 B 2 + [ 2(1 + 2 β ) MB 2 + ( 2(1 + β ) 2 + β |A| 2 M ) B 2 ] and L ζ = β . Now with the exactly same logic introduced in the Lemma C.7, this implies that R emp is also Lipschitz smooth with respect to θ with constant L θ = 2 B 2 + |A| 4 B 2 + [ 2(1 + 2 β ) MB 2 + ( 2(1 + β ) 2 + β |A| 2 M ) B 2 ] and Lipschitz smooth with respect to ζ with constant L ζ = β .