## APrimer on Maximum Causal Entropy Inverse Reinforcement Learning

Adam Gleave ∗ gleave@berkeley.edu

## Abstract

Inverse Reinforcement Learning (IRL) algorithms [17, 1] infer a reward function that explains demonstrations provided by an expert acting in the environment. Maximum Causal Entropy (MCE) IRL [31, 29] is currently the most popular formulation of IRL, with numerous extensions [5, 8, 21]. In this tutorial, we present a compressed derivation of MCE IRL and the key results from contemporary implementations of MCE IRL algorithms. We hope this will serve both as an introductory resource for those new to the field, and as a concise reference for those already familiar with these topics.

## Contents

| 1 Introduction                          | 1 Introduction                                                           | 1 Introduction                                                                      |   2 |
|-----------------------------------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----|
| 2 Background                            | 2 Background                                                             | 2 Background                                                                        |   2 |
|                                         | 2.1                                                                      | Markov decision processes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . |   3 |
|                                         | 2.2                                                                      | Imitation as feature expectation matching . . . . . . . . . . . . . . . . . . . .   |   3 |
|                                         | 2.3                                                                      | Maximum causal entropy . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    |   4 |
| Maximum Causal Entropy (MCE) IRL        | Maximum Causal Entropy (MCE) IRL                                         | Maximum Causal Entropy (MCE) IRL                                                    |   5 |
| 3.1                                     | MCE                                                                      | IRL as feature expectation matching . . . . . . . . . . . . . . . . . . . .         |   5 |
|                                         | 3.1.1                                                                    | The Lagrangian and the dual problem . . . . . . . . . . . . . . . . . .             |   6 |
|                                         | 3.1.2                                                                    | The dual function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       |   6 |
|                                         | 3.1.3                                                                    | Learning a reward function with dual ascent . . . . . . . . . . . . . .             |  11 |
| 3.2                                     | MCE IRL as maximum likelihood estimation .                               | . . . . . . . . . . . . . . . . .                                                   |  11 |
| 3.3                                     |                                                                          | Maximum Entropy (ME) IRL: MCE IRL for deterministic MDPs . . . . . . .              |  14 |
|                                         | 3.3.1                                                                    | MEIRL implies risk-seeking behaviour in stochastic environments .                   |  15 |
| 4 Dynamics-free approximations to MEIRL | 4 Dynamics-free approximations to MEIRL                                  | 4 Dynamics-free approximations to MEIRL                                             |  16 |
| 4.1                                     |                                                                          | Notation and Assumptions . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  |  16 |
|                                         |                                                                          | 4.1.1 Stationary policies and infinite-horizon MDPs . . . . . . . . . . . . .       |  16 |
| 4.2                                     | Guided cost learning: Approximation via importance sampling . . . .      | . . .                                                                               |  17 |
| 4.3                                     | An interlude on generative adversarial networks . . .                    | . . . . . . . . . . . . .                                                           |  19 |
| 4.4                                     | Adversarial IRL: Astate-centric, GAN-based approximation . . . . . . . . | .                                                                                   |  20 |
|                                         | 4.4.1                                                                    | Policy objective is entropy-regularised f . . . . . . . . . . . . . . . . .         |  20 |
|                                         | 4.4.2                                                                    | f recovers the optimal advantage . . . . . . . . . . . . . . . . . . . . .          |  21 |
|                                         | 4.4.3                                                                    | Reward shaping in MCE RL . . . . . . . . . . . . . . . . . . . . . . . .            |  21 |
|                                         | 4.4.4                                                                    | Discriminator objective . . . . . . . . . . . . . . . . . . . . . . . . . .         |  22 |
|                                         | 4.4.5                                                                    | Recovering rewards . . . . . . . . . . . . . . . . . . . . . . . . . . . .          |  24 |
| 5 Conclusion                            | 5 Conclusion                                                             | 5 Conclusion                                                                        |  27 |

∗ Equal contribution.

Sam Toyer ∗ sdt@berkeley.edu

## 1 Introduction

The most direct approach to automating a task is to manually specify the steps required to complete the task in the form of a policy . However, it is often easier to specify a reward function that captures the overarching task objective, and then use reinforcement learning (RL) to obtain a policy that carries out the steps to achieve that objective [23]. Unfortunately, procedurally specifying a reward function can also be challenging. Even a task as simple as peg insertion from pixels has a non-trivial reward function [27, Section IV.A]. Most real-world tasks have far more complex reward functions than this, especially when they involve human interaction.

A natural solution is to learn the reward function itself. A common approach is Inverse Reinforcement Learning (IRL) [17, 1]: inferring a reward function from a set of demonstrations of a particular task. This is well-suited to tasks that humans can easily perform but would find difficult to directly specify the reward function for, such as walking or driving. An additional benefit is that demonstrations can be cheaply collected at scale: for example, vehicle manufacturers can learn from their customers' driving behaviour [24, 14].

A key challenge for IRL is that the problem is underconstrained : many different reward functions are consistent with the observed expert behaviour [2, 15, 4, 22]. Some of these differences, such as scale or potential shaping, never change the optimal policy and so may be ignored [18, 10]. However, many differences do change the optimal policy-yet perhaps only in states that were never observed in the expert demonstrations. By contrast, alternative modalities such as actively querying the user for preference comparisons [20] can avoid this ambiguity, at the cost of a higher cognitive workload for the user.

Maximum Causal Entropy (MCE) IRL is a popular framework for IRL. Introduced by Ziebart [31, 29], MCE IRL models the demonstrator as maximising return achieved (like an RL algorithm) plus an entropy bonus that rewards randomising between actions. The entropy bonus allows the algorithm to model suboptimal demonstrator actions as random mistakes. In particular, it means that any set of sampled demonstrations has support under the demonstrator model, even if the trajectories are not perfectly optimal for any non-trivial reward function. This is important for modelling humans, who frequently deviate from optimality in complex tasks.

An alternative framework, Bayesian IRL [19], goes beyond finding a point estimate of the reward function and instead infers a posterior distribution over reward functions. It therefore assigns probability mass to all reward functions compatible with the demonstrations (so long as they have support in the prior). Unfortunately, Bayesian IRL is difficult to scale, and has to date only been demonstrated in relatively simple environments such as small, discrete MDPs.

In contrast to Bayesian IRL, algorithms based on MCE IRL have scaled to high-dimensional environments. Maximum Entropy Deep IRL [28] was one of the first extensions, and is able to learn rewards in gridworlds from pixel observations. More recently, Guided Cost Learning [6] and Adversarial IRL [8] have scaled to MuJoCo continuous control tasks. Given its popularity and accomplishments we focus on MCE IRL in the remainder of this document; we refer the reader to Jeon et al. [13] for a broader overview of reward learning.

## 2 Background

Before describing the MCE IRL algorithm itself, we first need to introduce some notation and concepts. First, we define Markov Decision Processes (MDPs). Then, we outline IRL based on feature matching . Finally, we introduce the notion of causal entropy .

## 2.1 Markov decision processes

In this tutorial, we consider Markov decision processes that are either discounted, or have a finite horizon, or both. Below we give the definition and notation that we use throughout. Note that when doing IRL, we drop the assumption that the true reward A is known, and instead infer it from data.

Definition 2.1. A Markov Decision Process (MDP) " = 'S GLYPH&lt;150&gt; A GLYPH&lt;150&gt; GLYPH&lt;15&gt; GLYPH&lt;150&gt; )GLYPH&lt;150&gt; I GLYPH&lt;150&gt; T GLYPH&lt;150&gt; A ' consists of a set of states S and a set of actions A ; a discount factor GLYPH&lt;15&gt; 2 » 0 GLYPH&lt;150&gt; 1 … ; a horizon ) 2 N [f1g ; an initial state distribution I' B ' ; a transition distribution T' B 0 j BGLYPH&lt;150&gt; 0 ' specifying the probability of transitioning to B 0 from B after taking action 0 ; and a reward function A ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' specifying the reward upon taking action 0 in state B and transitioning to state B 0 . It must be true that either the discount factor satisfies GLYPH&lt;15&gt; 5 1 or that the horizon is finite ( ) 5 1 ).

Given an MDP, we can define a (stochastic) policy GLYPH&lt;27&gt; C ' 0 C j B C ' that assigns a probability to taking action 0 C 2 A in state B C 2 S at time step C . The probability of a policy acting in the MDP producing a trajectory fragment GLYPH&lt;31&gt; = ' B 0 GLYPH&lt;150&gt; 0 0 GLYPH&lt;150&gt; B 1 GLYPH&lt;150&gt; 0 1 GLYPH&lt;150&gt; GLYPH&lt;149&gt; GLYPH&lt;149&gt; GLYPH&lt;149&gt; GLYPH&lt;150&gt; B : GLYPH&lt;0&gt; 1 GLYPH&lt;150&gt; 0 : GLYPH&lt;0&gt; 1 GLYPH&lt;150&gt; B : ' of length : 2 N is given by:

<!-- formula-not-decoded -->

Note in a finite-horizon MDP ( ) 2 N ) the policy may be non-stationary , i.e. it can depend on the time step C . In the infinite-horizon case ( ) = 1 ), the MDP is symmetric over time, and so we assume the policy is stationary. We drop the subscript and write only GLYPH&lt;27&gt; when the policy is acting across multiple timesteps or is known to be stationary.

The objective of an agent is to maximise the expected return:

<!-- formula-not-decoded -->

An optimal policy GLYPH&lt;27&gt; GLYPH&lt;3&gt; attains the highest possible expected return: GLYPH&lt;27&gt; GLYPH&lt;3&gt; 2 arg max GLYPH&lt;27&gt; GLYPH&lt;28&gt; ' GLYPH&lt;27&gt; ' .

## 2.2 Imitation as feature expectation matching

In IRL, our objective is to recover a reward function that-when maximised by a reinforcement learner-will lead to similar behaviour to the demonstrations. One way to formalise 'similar behaviour' is by feature expectation matching . Suppose the demonstrator is optimising some unknown linear reward function A GLYPH&lt;20&gt;8 ' B C GLYPH&lt;150&gt; 0 C ' = GLYPH&lt;20&gt; ) 8 ) ' B C GLYPH&lt;150&gt; 0 C ' , where ) ' B C GLYPH&lt;150&gt; 0 C ' 2 R 3 is some fixed feature mapping. In this case, the demonstrator's expected return under its behaviour distribution D will be linear in the expected sum of discounted feature vectors observed by the agent:

<!-- formula-not-decoded -->

Saythat we recover some imitation policy GLYPH&lt;27&gt; withidentical expected feature counts to the demonstrator:

<!-- formula-not-decoded -->

Because expected reward is linear in the (matched) feature expectations above, the reward obtained by GLYPH&lt;27&gt; under the unknown true reward function A GLYPH&lt;20&gt;8 ' B C GLYPH&lt;150&gt; 0 C ' must be the same as the reward obtained by the demonstrator D under that reward function [1]. If our imitation policy GLYPH&lt;27&gt; is optimal under reward function parameters GLYPH&lt;160&gt; GLYPH&lt;20&gt; , then it is reasonable to say that GLYPH&lt;160&gt; GLYPH&lt;20&gt; is an estimate of the demonstrator's true reward parameters. However, in general there will be many choices of GLYPH&lt;160&gt; GLYPH&lt;20&gt; that produce imitation policies GLYPH&lt;27&gt; with the same expected feature counts as the demonstrator. In the next section, we will see how we can apply the principle of maximum entropy to break ties between these reward functions.

## 2.3 Maximum causal entropy

The principle of maximum entropy holds that when choosing between many probability distributions that are consistent with the data, one should pick the distribution that has highest entropy . Intuitively, such a distribution is the 'most uncertain' among those that meet the data-fitting constraints. This principle can also be formally justified with an appeal to games: choosing the maximum entropy distribution minimises one's expected log loss in the setting where an adversary is able to choose the true distribution from those consistent with the data [25].

In an IRL setting, this principle leads to a simple and effective algorithm for simultaneously recovering a reward function and corresponding imitation policy from demonstrations. In particular, in MCE IRL we choose the reward function whose corresponding policy has maximal causal entropy :

Definition 2.2. Let ( 0: ) and GLYPH&lt;22&gt; 0: ) be random variables representing states and actions induced by following a policy GLYPH&lt;27&gt; in an MDP and sampled according to Eq. (1) . Then the causal entropy 1 GLYPH&lt;29&gt; ' GLYPH&lt;22&gt; 0: ) GLYPH&lt;0&gt; 1 k ( 0: ) GLYPH&lt;0&gt; 1 ' is the sum of the entropies of the policy action selection conditioned on the state at that timestep:

<!-- formula-not-decoded -->

Note the sum in Definition 2.2 is discounted, effectively valuing entropy of later actions less. This is needed for consistency with discounted returns, and for convergence in infinite-horizon problems; see Haarnoja et al. [12, appendix A] for more information. We will revisit the subtleties of infinite horizons in Section 4.1.1.

Remark 2.1. The causal entropy GLYPH&lt;29&gt; ' GLYPH&lt;22&gt; 0: ) GLYPH&lt;0&gt; 1 k ( 0: ) GLYPH&lt;0&gt; 1 ' has the useful property that at each time step, it conditions only on information available to the agent (that is, the current state, as well as prior states and actions). By contrast, the conditional entropy of actions given states GLYPH&lt;29&gt; ' GLYPH&lt;22&gt; 0: ) GLYPH&lt;0&gt; 1 j ( 0: ) GLYPH&lt;0&gt; 1 ' conditions on states that arise after each action was taken. Moreover, conventional Shannon entropy GLYPH&lt;29&gt; ' ( 0: ) GLYPH&lt;0&gt; 1 GLYPH&lt;150&gt; GLYPH&lt;22&gt; 0: ) GLYPH&lt;0&gt; 1 ' calculates the entropy over the entire trajectory distribution, introducing an unwanted dependency on transition dynamics via terms GLYPH&lt;29&gt; ' (C j (C GLYPH&lt;0&gt; 1 GLYPH&lt;150&gt; GLYPH&lt;22&gt; C GLYPH&lt;0&gt; 1 ' :

<!-- formula-not-decoded -->

Maximising Shannon entropy therefore introduces a bias towards taking actions with uncertain (and possibly risky) outcomes. For this reason, maximum causal entropy should be used rather than maximum (Shannon) entropy in stochastic MDPs. In deterministic MDPs, GLYPH&lt;29&gt; ' (C j (C GLYPH&lt;0&gt; 1 GLYPH&lt;150&gt; GLYPH&lt;22&gt; C GLYPH&lt;0&gt; 1 ' = 0 and the methods are equivalent (Section 3.3).

1 The definition of causal entropy can also be generalised to non-Markovian contexts [29, section 4.2].

## 3 Maximum Causal Entropy (MCE) IRL

In this section, we start by reviewing two complementary ways of formalising the MCE IRL objective as an optimisation problem. First, we will consider MCE IRL as a way of solving a particular constrained optimisation problem. Second, we will describe MCE IRL in terms of maximum likelihood estimation, which will allow us to replace the linear reward function with a non-linear one. Finally, we will discuss how MCE IRL simplifies to Maximum Entropy (ME) IRL under deterministic dynamics.

## 3.1 MCEIRL as feature expectation matching

MCE IRL was originally introduced by Ziebart et al. [31], who considered the general setting of nonMarkovian dynamics and trajectory-centric features. They then derived simplifications for the special case of Markovian dynamics and policies, as well as feature functions that decompose across stateaction transitions. This primer only considers the simpler case (decomposed features, Markovian policies), and is thus substantially shorter. Moreover, we will assume throughout that the horizon ) is finite, and that the policy is non-stationary (time-dependent); removing these assumptions is discussed in Section 4.1.1

Optimisation Problem 3.1 (MCE IRL primal problem) . Given an expert data distribution D , the optimisation problem of finding a time-dependent stochastic policy GLYPH&lt;27&gt; C ' B C GLYPH&lt;150&gt; 0 C ' that matches feature expectations while maximising causal entropy can be written as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where GLYPH&lt;15&gt; denotes the set of policies where the action distribution at each state falls within the standard policy simplex, so that

<!-- formula-not-decoded -->

and ( 0: ) GLYPH&lt;0&gt; 1 and GLYPH&lt;22&gt; 0: ) GLYPH&lt;0&gt; 1 are random sequences of states and actions, induced by GLYPH&lt;27&gt; C and sampled according to Eq. (1) on the left, and D on the right.

This primal problem is not convex in general, but we will nevertheless approach solving it using a standard recipe for convex optimisation problems [3, Chap. 5]:

1. Form the Lagrangian. First, we will convert the feature expectation matching constraint into a weighted penalty term, producing the Lagrangian for the problem.
2. Derive the dual problem. Next, we will use the Lagrangian to form a dual problem that is equivalent to Optimisation Problem 3.1. The dual problem will introduce an extra set of parameterscalled Lagrange multipliers-which in this case will be interpretable as weights of a reward function.
3. Dual ascent. Finally, we will solve the dual problem with a simple procedure called dual ascent , which alternates between exactly maximising the Lagrangian with respect to the policy parameters (the primal variables) and taking a single gradient step to minimise the Lagrangian with respect to the reward weights (the dual variables). This process is repeated until the dual variables converge.

## 3.1.1 The Lagrangian and the dual problem

We begin our derivation of the dual problem by forming the Lagrangian GLYPH&lt;3&gt; : GLYPH&lt;15&gt; GLYPH&lt;2&gt; R 3 ! R for Optimisation Problem 3.1:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The dual variable vector GLYPH&lt;20&gt; 2 R 3 can be interpreted as a set of penalty weights to enforce the feature expectation matching constraint in Eq. (13b). Importantly, GLYPH&lt;27&gt; is constrained to the simplex GLYPH&lt;15&gt; ; later, we will need be careful with this constraint when we form a separate, nested optimisation problem to compute GLYPH&lt;27&gt; .

Equipped with the Lagrangian, we can now express the dual problem to Optimisation Problem 3.1

Optimisation Problem 3.2 (Dual MCE IRL problem) . Define the dual function 6 ' GLYPH&lt;20&gt; ' as

<!-- formula-not-decoded -->

The dual MCE IRL problem is the problem of finding a GLYPH&lt;27&gt; GLYPH&lt;3&gt; and GLYPH&lt;20&gt; GLYPH&lt;3&gt; such that GLYPH&lt;20&gt; GLYPH&lt;3&gt; attains

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

while GLYPH&lt;27&gt; GLYPH&lt;3&gt; attains

Instead of optimising the primal (Optimisation Problem 3.1), we will instead optimise the dual (Optimisation Problem 3.2), and treat the recovered GLYPH&lt;27&gt; GLYPH&lt;3&gt; as a solution for the primal imitation learning problem. Moreover, we will later see that the recovered GLYPH&lt;20&gt; GLYPH&lt;3&gt; can be interpreted as parameters of a reward function that incentivises reproducing the expert demonstrations.

## 3.1.2 The dual function

Observe that Optimisation Problem 3.2 is expressed in terms of a dual function 6 ' GLYPH&lt;20&gt; ' = max GLYPH&lt;27&gt; GLYPH&lt;3&gt; ' GLYPH&lt;27&gt; GLYPH&lt;150&gt; GLYPH&lt;20&gt; ' . Computing 6 ' GLYPH&lt;20&gt; ' can therefore be viewed as a nested optimisation over GLYPH&lt;27&gt; subject to the probability simplex constraints defining GLYPH&lt;15&gt; .

We will begin by putting the optimisation defining the dual function into the familiar form. First recall that the policy simplex GLYPH&lt;15&gt; is defined by two constraints:

<!-- formula-not-decoded -->

Since the causal entropy term in GLYPH&lt;3&gt; is undefined when GLYPH&lt;27&gt; has negative elements, we will treat the non-negativity constraint as implicit, and focus only on the normalisation constraint. We will rewrite the normalisation constraint as GLYPH&lt;17&gt; B C ' GLYPH&lt;27&gt; ' = 0, where the function GLYPH&lt;17&gt; B C is defined as

<!-- formula-not-decoded -->

This gives rise the following optimisation problem:

Optimisation Problem 3.3 (Dual function primal problem) . The problem of computing the dual function can be equivalently expressed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Lagrangian for Optimisation Problem 3.3 is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where GLYPH&lt;8&gt; GLYPH&lt;24&gt; B C GLYPH&lt;9&gt; B C 2S GLYPH&lt;150&gt; 0 GLYPH&lt;20&gt; C 5 ) are dual variables for the normalisation constraints, and we have left the dependence on GLYPH&lt;20&gt; implicit. We will attempt to find a minimiser of Optimisation Problem 3.3 by solving the Karush-Kuhn-Tucker (KKT) conditions for the problem [3, p.224], which are 2

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that GLYPH&lt;24&gt; only appears in the first equation. We will later see that we can set it to a value which ensures that the normalisation constraints are satisfied and the gradient vanishes. First, however, we will compute the gradient of the Lagrangian for Optimisation Problem 3.3:

Lemma 3.1. The gradient of GLYPH&lt;9&gt; ' GLYPH&lt;27&gt; GLYPH&lt;150&gt; GLYPH&lt;24&gt; ; GLYPH&lt;20&gt; ' is given by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the discounted probability that the agent will be in state B C at time C if it follows policy GLYPH&lt;27&gt; .

Proof. We will derive this gradient term-by-term. Taking the derivative of the third term (normalisation) with respect to some arbitrary GLYPH&lt;27&gt; C ' 0 C j B C ' yields

<!-- formula-not-decoded -->

Note that we are differentiating with respect to the action selection probability GLYPH&lt;27&gt; C ' 0 C j B C ' , which is a variable specific to time C ; the policy need not be stationary, so we may have GLYPH&lt;27&gt; C ' 0 j B ' &lt; GLYPH&lt;27&gt; C 0' 0 j B ' .

2 This problem does not have explicit inequality constraints, so we do not require complementary slackness conditions.

The derivative of the middle term (feature expectation matching) is

<!-- formula-not-decoded -->

Finally, the derivative of the first term (causal entropy) can be derived in a similar manner as

<!-- formula-not-decoded -->

Putting it all together, the derivative of GLYPH&lt;9&gt; ' GLYPH&lt;27&gt; GLYPH&lt;150&gt; GLYPH&lt;24&gt; ; GLYPH&lt;20&gt; ' with respect to our policy is

<!-- formula-not-decoded -->

We will now solve the first KKT condition, Eq. (24), by setting Eq. (26) equal to zero and solving for GLYPH&lt;27&gt; . This will give us the GLYPH&lt;27&gt; which attains 6 ' GLYPH&lt;20&gt; ' = GLYPH&lt;3&gt; ' GLYPH&lt;27&gt; GLYPH&lt;150&gt; GLYPH&lt;20&gt; ' , thereby allowing us to achieve our goal (for this sub-section) of computing the dual function 6 ' GLYPH&lt;20&gt; ' . Conveniently, the resulting GLYPH&lt;27&gt; has a form which resembles the optimal policy under value iteration-indeed, the recursion defining GLYPH&lt;27&gt; is often called soft value iteration :

Lemma 3.2. The KKT condition r GLYPH&lt;27&gt; GLYPH&lt;9&gt; ' GLYPH&lt;27&gt; GLYPH&lt;150&gt; GLYPH&lt;24&gt; ; GLYPH&lt;20&gt; ' = 0 is satisfied by a policy

<!-- formula-not-decoded -->

satisfying the following recursion:

<!-- formula-not-decoded -->

Proof. We begin by setting the derivative of the Lagrangian GLYPH&lt;9&gt; to zero and rearranging. Through this, we find that at optimality the policy (i.e. primal variables) must take the form

<!-- formula-not-decoded -->

We abuse notation by assuming that the last term vanishes when C = ) GLYPH&lt;0&gt; 1, so that at the end of the trajectory we have

<!-- formula-not-decoded -->

Naively calculating the optimal policy from Eq. (42) would require enumeration of exponentially many trajectories to obtain the inner expectation. We will instead show that GLYPH&lt;27&gt; can be recovered by soft value iteration. First we decompose GLYPH&lt;27&gt; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' 0 C j B C ' using Eq. (40), where &amp; soft and + soft functions are defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We can show that &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' and + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C ' satisfy a 'softened' version of the Bellman equations. Note that by setting the GLYPH&lt;24&gt; B C variables to be appropriate normalising constants, we ensure that the normalisation KKT conditions in Eq. (25) are satisfied. Since we wish to constrain ourselves to

normalised policies GLYPH&lt;27&gt; 2 GLYPH&lt;15&gt; , we can assume the GLYPH&lt;24&gt; B C are chosen such that ˝ 0 2A GLYPH&lt;27&gt; C ' 0 j B C ' = 1 for all B C 2 S . It then follows that + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C ' must be a soft maximum over &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' values in B C :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Likewise, we can use the definitions of &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C and + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C to show that for C 5 ) GLYPH&lt;0&gt; 1, we have

<!-- formula-not-decoded -->

where the penultimate step follows from substituting GLYPH&lt;27&gt; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' 0 C j B C ' = exp GLYPH&lt;16&gt; &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' GLYPH&lt;0&gt; + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C ' GLYPH&lt;17&gt; . Putting it all together, we have the soft VI equations:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

GLYPH&lt;3&gt;

Remark 3.1 (Soft VI) . Observe that the soft VI equations are analogous to traditional value iteration, but with the hard maximum over actions replaced with a log-sum-exp, which acts as a 'soft' maximum. In the

## Algorithm 1 Maximum Causal Intropy (MCE) IRL on demonstrator D

- 1: Initialise some reward parameter estimate GLYPH&lt;20&gt; 0, and set : 0
- 2: repeat
- 3: Apply soft value iteration to obtain optimal policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; w.r.t GLYPH&lt;20&gt; : (Eq. (41))

:

<!-- formula-not-decoded -->

- 5: : : , 1
- 6: until Stopping condition satisfied

finite-horizon case, these equations can be applied recursively from time C = ) GLYPH&lt;0&gt; 1 down to C = 0 , yielding an action distribution GLYPH&lt;27&gt; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' 0 C j B C ' = exp ' &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' GLYPH&lt;0&gt; + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C '' at each time step C and state B C . In the infinite-horizon case, we drop the subscripts C and search for a fixed point + soft GLYPH&lt;20&gt; and &amp; soft GLYPH&lt;20&gt; to the soft VI equations with corresponding stationary policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; .

In both cases, the agent chooses actions with probability exponential in the soft advantage GLYPH&lt;22&gt; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' , &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' GLYPH&lt;0&gt; + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C ' . Thus, computing the dual function 6 ' GLYPH&lt;20&gt; ' reduces to solving a planning problem. The similarity between the soft VI equations and the ordinary Bellman equations means that we are (somewhat) justified in interpreting the dual variable vector GLYPH&lt;20&gt; as weights for a reward function A GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' = GLYPH&lt;20&gt; ) ) ' B C GLYPH&lt;150&gt; 0 C ' .

## 3.1.3 Learning a reward function with dual ascent

In the previous subsection, we determined how to find the policy GLYPH&lt;27&gt; which attains the value of the dual function 6 ' GLYPH&lt;20&gt; ' , so that 6 ' GLYPH&lt;20&gt; ' = GLYPH&lt;3&gt; ' GLYPH&lt;27&gt; GLYPH&lt;150&gt; GLYPH&lt;20&gt; ' . All that remains is to determine the step that we need to take on the dual variables GLYPH&lt;20&gt; (which we are interpreting as reward parameters). The gradient of the dual w.r.t GLYPH&lt;20&gt; is given by

<!-- formula-not-decoded -->

The first (policy GLYPH&lt;27&gt; ) term can be computed by applying soft VI to GLYPH&lt;20&gt; and then rolling the optimal policy forward to obtain occupancy measures. The second (demonstrator D ) term can be computed directly from demonstration trajectories. This cycle of performing soft VI followed by a gradient step on GLYPH&lt;20&gt; is illustrated in Algorithm 1. In the next section, we'll see an alternative perspective on the same algorithm that does not appeal to feature expectation matching or duality, and which can be extended to non-linear reward functions.

## 3.2 MCEIRL as maximum likelihood estimation

In the previous section, we saw that IRL can be reduced to dual ascent on a feature expectation matching problem. We can also interpret this method as maximum likelihood estimation of reward parameters GLYPH&lt;20&gt; subject to the distribution over trajectories induced by the policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' 0 C j B C ' = exp ' &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' GLYPH&lt;0&gt; + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C '' under the soft VI recursion in Eq. (41). This perspective has the advantage of allowing us to replace the linear reward function A GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' = GLYPH&lt;20&gt; ) ) ' B C GLYPH&lt;150&gt; 0 C ' with a general non-linear reward function A GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' . If A GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' is non-linear, then the resulting problem no longer has the same maximum-entropy feature-matching justification as before, although it does still appear to work well in practice.

To show that the feature expectation matching and maximum likelihood views are equivalent when A GLYPH&lt;20&gt; is linear, we will show that the gradient for the dual function 6 ' GLYPH&lt;20&gt; ' and the gradient for the dataset expected log likelihood are equivalent. First, we will introduce the notion of the discounted likelihood of a trajectory.

Definition3.1 (Discountedlikelihood) . Thediscountedlikelihood of a trajectory GLYPH&lt;31&gt; = ' B 0 GLYPH&lt;150&gt; 0 0 GLYPH&lt;150&gt; B 1 GLYPH&lt;150&gt; 0 1 GLYPH&lt;150&gt; GLYPH&lt;149&gt; GLYPH&lt;149&gt; GLYPH&lt;149&gt; GLYPH&lt;150&gt; 0 ) GLYPH&lt;0&gt; 1 GLYPH&lt;150&gt; B ) ' under policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; is

<!-- formula-not-decoded -->

where GLYPH&lt;15&gt; 2 » 0 GLYPH&lt;150&gt; 1 … is a discount factor.

When GLYPH&lt;15&gt; = 1, the discounted likelihood is simply the likelihood of GLYPH&lt;31&gt; under the policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; . When GLYPH&lt;15&gt; 5 1, probabilities of actions later in the trajectory will be regularised towards 1 by raising them to GLYPH&lt;15&gt; C . This will not correspond to a normalised probability distribution, although we will see that it allows us to draw a connection between maximum likelihood inference and discounted MCE IRL.

Now consider the log discounted likelihood of a single trajectory GLYPH&lt;31&gt; under a model in which the agent follows the soft VI policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' 0 C j B C ' with respect to reward parameters GLYPH&lt;20&gt; :

<!-- formula-not-decoded -->

Here 5 ' GLYPH&lt;31&gt; ' contains all the log T and log I (dynamics) terms, which are constant in GLYPH&lt;20&gt; . Although the expression above only matches the log likelihood when GLYPH&lt;15&gt; = 1, we have nevertheless added explicit discount factors so that the gradient equivalence with 6 ' GLYPH&lt;20&gt; ' holds when GLYPH&lt;15&gt; 5 1. The expected log likelihood of a demonstrator's trajectories, sampled from demonstration distribution D , is then

<!-- formula-not-decoded -->

where 2 is a constant that does not depend on GLYPH&lt;20&gt; .

The transitions in the demonstration distribution D will follow T provided the demonstrator acts in the same MDP, and so we can drop the E T . Critically, this would not be possible were D instead an empirical distribution from a finite number of samples, as then D might not exactly follow T . However, this is a reasonable approximation for sufficiently large demonstration datasets.

After dropping E T , we simplify the telescoping sums:

<!-- formula-not-decoded -->

All that remains to show is that the gradient of this expression is equal to the gradient of the dual function 6 ' GLYPH&lt;20&gt; ' that we saw in Section 3.

Lemma 3.3. Assume that the reward function is linear in its parameters GLYPH&lt;20&gt; , so that A GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' = GLYPH&lt;20&gt; ) ) ' BGLYPH&lt;150&gt; 0 ' . Gradient ascent on the expected log likelihood from Eq. (68) updates parameters in the same direction as gradient descent on 6 ' GLYPH&lt;20&gt; ' from Optimisation Problem 3.2; that is,

<!-- formula-not-decoded -->

where the negative on the right arises from the fact that we are doing gradient ascent on the expected log likelihood, and gradient descent on the dual function 6 ' GLYPH&lt;20&gt; ' .

Proof. We already know the gradient of 6 ' GLYPH&lt;20&gt; ' from Eq. (59), so we only need to derive the gradient of L'D ; GLYPH&lt;20&gt; ' . Computing the gradient of the first term of L'D ; GLYPH&lt;20&gt; ' is trivial. We push the differentiation operator inside the expectation so that we are averaging r A GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' over the states and actions in our dataset of samples from D :

<!-- formula-not-decoded -->

The second term is slightly more involved, but can be derived recursively as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step unrolled the recursion to time ) . Armed with this derivative, the gradient of our log likelihood is

<!-- formula-not-decoded -->

In the special case of a linear reward A GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' = GLYPH&lt;20&gt; ) ) ' BGLYPH&lt;150&gt; 0 ' , this gradient is equal to GLYPH&lt;0&gt;r GLYPH&lt;20&gt; 6 ' GLYPH&lt;20&gt; ' (where r GLYPH&lt;20&gt; 6 ' GLYPH&lt;20&gt; ' is given in Eq. (59)). This shows that maximising the log likelihood L is equivalent to minimising the dual function 6 ' GLYPH&lt;20&gt; ' from the previous section. GLYPH&lt;3&gt;

## 3.3 Maximum Entropy (ME) IRL: MCE IRL for deterministic MDPs

So far we've considered the general setting of stochastic transition dynamics, where for any given state-action pair ' B C GLYPH&lt;150&gt; 0 C ' 2 S GLYPH&lt;2&gt;A there may be many possible successor states B C , 1 2 S for which T' B C , 1 j B C GLYPH&lt;150&gt; 0 C ' 7 0. Consider what happens if we act under the Maximum Causal Entropy (MCE) policy GLYPH&lt;27&gt; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' 0 C j B C ' = exp ' &amp; GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C GLYPH&lt;150&gt; 0 C ' GLYPH&lt;0&gt; + GLYPH&lt;20&gt; GLYPH&lt;150&gt;C ' B C '' for some GLYPH&lt;20&gt; , but restrict ourselves to the case of deterministic dynamics, where T' B C , 1 j B C GLYPH&lt;150&gt; 0 C ' = 1 for one successor state B C , 1 and zero for all others. Here the discounted likelihood of some feasible trajectory GLYPH&lt;31&gt; -in which the state transitions agree with the dynamics-simplifies substantially.

Lemma 3.4. In an MDP with deterministic state transitions and a deterministic initial state distribution, the MCE IRL density ? GLYPH&lt;20&gt; GLYPH&lt;150&gt; GLYPH&lt;15&gt; ' GLYPH&lt;31&gt; ' takes the form

<!-- formula-not-decoded -->

where / GLYPH&lt;20&gt; = exp + GLYPH&lt;20&gt; GLYPH&lt;150&gt; 0 ' B 0 ' is a normalising constant dependent on the (deterministic) initial state B 0 .

Proof. Assuming deterministic dynamics, the discounted likelihood for a feasible GLYPH&lt;31&gt; becomes:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Wemade use of determinism to drop the dynamics factors in Eq. (80), and to collapse the expectation over successor states that appeared in Eq. (82). If we now assume that the initial state distribution I is deterministic (with all weight on the actual initial state B 0 at the beginning of the trajectory GLYPH&lt;31&gt; ), then we get I' B 0 ' = 1, from which the result follows. GLYPH&lt;3&gt;

If we perform IRL under the assumption that our trajectory distribution follows the form of Eq. (78), then we recover the Maximum Entropy IRL algorithm (abbreviated MaxEnt or ME ) that was first

Figure 1: RiskyPath : The agent can either take a long but sure path to the goal ( B 0 ! B 1 ! B 2), or attempt to take a shortcut ( B 0 ! B 2 ' , with the risk of receiving a low reward ( B 0 ! B 3). This example is simplified from Ziebart's thesis [29, Figure 6.4].

![Image](data:image/png;base64,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)

proposed by Ziebart [30]. As we will see in Section 4, this simplified algorithm lends itself well to approximation in environments with unknown (but deterministic) dynamics [6, 5, 8].

## 3.3.1 MEIRL implies risk-seeking behaviour in stochastic environments

Observe that the Maximum Entropy (ME) IRL discounted likelihood in Eq. (78) takes a much simpler form than the likelihood induced by MCE IRL, which requires recursive evaluation of the soft VI equations. Given that ME IRL has a simpler discounted likelihood, one might be tempted to use ME IRL in environments with stochastic dynamics by inserting dynamics factors to get a discounted likelihood of the form

<!-- formula-not-decoded -->

Unfortunately, 'generalising' ME IRL in this way may not produce appropriate behaviour in environments with truly non-deterministic dynamics. The intuitive reason for this is that the ME IRL discounted likelihood in Eq. (78) can assign arbitrarily high likelihood to any feasible trajectory, so long as the return associated with that trajectory is high enough. Indeed, by moving the dynamics factors into the exp, we can see that the discounted likelihood 3 takes the form

<!-- formula-not-decoded -->

where ' GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' = ˝ ) GLYPH&lt;0&gt; 1 C = 0 GLYPH&lt;15&gt; C A GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' . This form suggests that the agent can simply pay a 'log cost' in order to choose outcomes that would give it better return. A trajectory GLYPH&lt;31&gt; with extremely high return ' GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' relative to other trajectories may have high likelihood even when the associated transition probabilities are low. In reality, if a non-deterministic environment has a state transition ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' with very low (but non-zero) probability &amp; , then no trajectory containing that transition can have likelihood more than &amp; under any physically realisable policy, even if the trajectory has extremely high return.

This mismatch between real MDP dynamics and the ME IRL model can manifest as risk-taking behaviour in the presence of stochastic transitions. For example, in Fig. 1, an agent can obtain B 0 ! B 1 ! B 2 with 100% probability, or B 0 ! B 2 and B 0 ! B 3 with 50% probability each. The ME IRL transition model would wrongly believe the agent could obtain B 0 ! B 2 with arbitrarily high probability by paying a small cost of log T' B 2 j B 0 GLYPH&lt;150&gt; GLYPH&lt;1&gt;' = log 1 2 = GLYPH&lt;0&gt; log 2. It would therefore conclude

3 Previously, we defined the discounted likelihood as the ordinary trajectory likelihood with discount factors applied at each time step. In order to obtain a valid probability distribution, the discounted likelihood will have to be normalised. The reasoning in this section still applies after normalisation: as the return of a trajectory increases, its probability under the model will go to 1, even if stochastic transitions make it impossible to find a policy that actually achieves such a trajectory distribution.

that an agent given rewards A ' B 2 ' = 1 and A ' B 3 ' = GLYPH&lt;0&gt; 100 would favour the risky path for a sufficiently small discount factor GLYPH&lt;15&gt; , despite the true expected return always being higher for the safe path.

## 4 Dynamics-free approximations to ME IRL

Real-world environments are often too complex to be described in the tabular form required by MCE IRL. To perform IRL in these settings, we can use 'dynamics-free' approximations to MCE IRL which do not require a known world model. In this section we describe several such approximations, all of which assume the environment is deterministic (simplifying MCE to ME IRL) and that the horizon is effectively infinite (simplifying to a stationary policy).

We begin by describing the assumptions made by these algorithms in more detail, and adapting our previous definitions to the non-tabular setting. We then consider Guided Cost Learning (GCL), which directly approximates the gradient of the ME IRL objective using importance-weighted samples [6]. Finally, we will describe Adversarial IRL (AIRL) [8], a GAN-based approximation to ME IRL, that distinguishes between state-action pairs instead of distinguishing between trajectories. AIRL has been shown to be successful in standard RL benchmark environments, including some MuJoCo-based control tasks [8] and some Atari games [26].

## 4.1 Notation and Assumptions

Determinism GCL is based on ME IRL, which Section 3.3 showed is only well-founded for deterministic dynamics. AIRL can be derived using MCE IRL, but many of the key results only hold for deterministic environments. Thus we will assume that environments are deterministic throughout this section.

Function approximation Since we'll be discussing results about non-parametric and non-linear reward functions, we will overload our &amp; soft and + soft notation from earlier to refer to arbitrary reward functions. For example, given some reward function A ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' , we use &amp; soft A ' BGLYPH&lt;150&gt; 0 ' to denote the soft Q-function under A . We'll also be making use of the soft advantage function , defined as

<!-- formula-not-decoded -->

## 4.1.1 Stationary policies and infinite-horizon MDPs

In addition to nonlinear function approximation, the algorithms which we will consider in the next few sections are designed to work with stationary (time-independent) policies. Theoretically, the use of stationary policies can be justified by appealing to infinite horizons. In an infinite horizon problem, the value of a given state is independent of the current time step, since there will always be an infinite number of time steps following it. Consequently, the value function + soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C and Q-function &amp; soft GLYPH&lt;20&gt; GLYPH&lt;150&gt;C lose their time-dependence, and can instead be recovered as the fixed point of the following recurrence:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This yields a stationary policy GLYPH&lt;27&gt; ' 0 j B ' = exp GLYPH&lt;22&gt; soft A ' BGLYPH&lt;150&gt; 0 ' .

The notion of an 'infinite' horizon might sound abstruse, given that real-world demonstration trajectories GLYPH&lt;31&gt; GLYPH&lt;24&gt; D must be finite, but we can conceptually view them as being infinite trajectories padded with a virtual absorbing state. This helps us model situations where the demonstrator always completes the task in finite time, after which the demonstration ends.

We can also use this padding approach to model simulated tasks, where episodes often end when some termination condition is satisfied, such as a robot falling over or a vehicle reaching its destination. For the purpose of IRL, we can again imagine such trajectories as ending with an infinite sequence of repeated absorbing states, each with the same learned absorbing state reward value.

Note the presence of a termination condition may simplify the IRL problem considerably when episode termination is (positively or negatively) correlated with task success. Indeed, Kostrikov et al. [16] observe it is possible to get seemingly-competent behaviour in the Hopper environment by simply giving the agent a fixed positive reward at all time steps until termination, then zero afterwards. In general, the absorbing state reward tends to be low in tasks which require the agent to stay 'alive' and high in goal-oriented tasks where the agent needs to finish an episode quickly.

Implementation termination bias In practice, implementations of IRL algorithms often wrongly treat the episodes as being of finite but variable length, implicitly assuming that the absorbing state reward is exactly zero. To make matters worse, certain choices of neural network architecture can force a learned reward function to always be positive (or negative) up until termination. This can lead to situations where the algorithm 'solves' an environment regardless of what reward parameters it learns, or where the algorithm cannot ever solve the environment because its reward function has the wrong sign. It is possible to fix this problem by explicitly learning an absorbing state reward, but for the purpose of comparing existing algorithms we instead recommend using fixed-horizon environments, which side-steps this problem entirely [9].

Convergence of soft VI Soft value iteration still converges to a unique value in the infinite horizon case, provided the discount factor GLYPH&lt;15&gt; is less than 1 [12, Appendix A.2]. In particular, Haarnoja et al. [12]-adaptinganearlierproofbyFoxetal.[7]-showthatsoftvalueiterationisacontractionmapping. It then follows from the Banach fixed point theorem that soft value iteration has a unique fixed point.

## 4.2 Guided cost learning: Approximation via importance sampling

Consider the gradient of the log discounted likelihood of a demonstration distribution D under the MEIRL model, which was previously shown in Eq. (77) to be

<!-- formula-not-decoded -->

Sample-based approximation of Term (90a) is straightforward: we can just sample # trajectories GLYPH&lt;31&gt; 1 GLYPH&lt;150&gt; GLYPH&lt;31&gt; 2 GLYPH&lt;150&gt; GLYPH&lt;149&gt; GLYPH&lt;149&gt; GLYPH&lt;149&gt; GLYPH&lt;150&gt; GLYPH&lt;31&gt; # GLYPH&lt;24&gt; D from the demonstration distribution D , then approximate the expectation with sample mean

<!-- formula-not-decoded -->

Approximating Term (90b) is harder, since recovering GLYPH&lt;27&gt; GLYPH&lt;20&gt; requires planning against the estimated reward function A GLYPH&lt;20&gt; . Although we could compute GLYPH&lt;27&gt; GLYPH&lt;20&gt; using maximum entropy RL [12], doing so would be extremely expensive, and we would have to repeat the process each time we update the reward parameters GLYPH&lt;20&gt; .

Importance sampling Instead of learning and sampling from GLYPH&lt;27&gt; GLYPH&lt;20&gt; , we can estimate the ME IRL gradient using importance sampling. Importance sampling is a widely applicable technique for working with distributions that are difficult to sample from or evaluate densities for. Imagine that we would like to evaluate an expectation E ? 5 ' -' , where -is a random variable with density ? and

5 : X ! R is some function on the sample space X . Say that the distribution ? is intractable to sample from, and we only know how to evaluate an unnormalised density ¡ ? ' G ' = GLYPH&lt;13&gt; ? ' G ' with unknown scaling factor GLYPH&lt;13&gt; . However, imagine that it is still possible to draw samples from another distribution @ for which we can evaluate the density @ ' G ' at any point G 2 X . In this case, we can rewrite our expectation as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we can evaluate GLYPH&lt;13&gt; using an expectation with respect to @ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we let F ' G ' = ¡ ? ' G 'GLYPH&lt;157&gt; @ ' G ' , then we have

<!-- formula-not-decoded -->

Both the numerator and the denominator of Eq. (100) can be approximated by averaging F ' -' 5 ' -' and F ' -' on samples drawn from @ . This removes the need to sample from ? or evaluate its density directly.

Applying importance sampling to IRL Now consider how we could use importance sampling to estimate Term (90b). In this case, we wish to sample from the distribution induced by GLYPH&lt;27&gt; GLYPH&lt;20&gt; , which has discounted likelihood ? GLYPH&lt;20&gt; GLYPH&lt;150&gt; GLYPH&lt;15&gt; ' GLYPH&lt;31&gt; ' = 1 / exp A GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' , for / = fl exp A GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' 3 GLYPH&lt;31&gt; integrated over all feasible trajectories GLYPH&lt;31&gt; . 4 Again, computing / is intractable-doing so is equivalent to optimally solving an entropy-regularised RL problem. Instead of evaluating / directly, we can use importance sampling: if there is some easy-to-compute trajectory distribution @ ' GLYPH&lt;31&gt; ' that we can draw samples from, then we can rewrite the expectation in Term (90b) as 5

<!-- formula-not-decoded -->

4 Note that if GLYPH&lt;15&gt; = 1, then ? GLYPH&lt;20&gt; GLYPH&lt;150&gt; GLYPH&lt;15&gt; will be undefined over infinite horizons for some MDPs, since A GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' will diverge. On the other hand, if GLYPH&lt;15&gt; 5 1, then ? GLYPH&lt;20&gt; GLYPH&lt;150&gt; GLYPH&lt;15&gt; will not be a normalised density unless we incorporate an additional GLYPH&lt;15&gt; -dependent factor into the / ' GLYPH&lt;20&gt; ' we derived in Lemma 3.4. We will ignore these subtleties for the remainder of the section by implicitly assuming that the horizon is finite, but that the policy is still stationary.

5 To avoid confusion with the horizon ) , we are treating lowercase GLYPH&lt;31&gt; as a random variable in these expectations.

where F GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' = GLYPH&lt;0&gt; exp A GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' GLYPH&lt;1&gt; GLYPH&lt;157&gt; @ ' GLYPH&lt;31&gt; ' denotes the importance weighting function.

ThemaininsightofGuidedCostLearning(GCL)isthattheproposaldistribution @ ' GLYPH&lt;31&gt; ' canbeiteratively updated to bring it close to the optimal proposal distribution ? GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' without having to compute ? GLYPH&lt;20&gt; ' GLYPH&lt;31&gt; ' exactly [6]. Specifically, after each cost function update, the proposal distribution @ ' GLYPH&lt;31&gt; ' is updated by using reinforcement learning to maximise the entropy-regularised return,

<!-- formula-not-decoded -->

Since GCL uses importance sampling to evaluate the ME IRL gradient, it does not need to train @ to optimality against 'GLYPH&lt;29&gt; ' @ ' . Instead, doing a few RL steps after each cost function update can suffice to produce a good proposal distribution @ .

In addition to the core insight that an adaptive proposal distribution can improve importance-sampled gradient estimates in ME IRL, the GCL paper also proposes a number of tricks that are necessary to make the method work in practice. Most significantly, the actual proposal distribution @ ' GLYPH&lt;31&gt; ' used in the GCL paper is not just a policy trained with RL, but a mixture between an RL-trained policy and an approximation of the expert trajectory distribution. Moreover, the learnt reward function uses an architecture and pair of regularisers that make it particularly well-suited to the goal-reaching tasks that GCL was evaluated on. In the remainder of this primer, we will introduce the Adversarial IRL algorithm [8], which does not rely on importance sampling and does not require as many special tricks to work well in practice.

## 4.3 An interlude on generative adversarial networks

In the next sub-section, we will see that ME IRL can be recast as a way of learning a speciallystructured kind of Generative Adversarial Network (GAN). This sub-section will briefly review the basic principles behind GANs. Readers unfamiliar with GANs may wish to consult the original GAN paper [11].

GANs include a generator function G = GLYPH&lt;28&gt; ' I ' that maps from a noise vector I to a generated sample G 2 X , for some output space X . Given some fixed noise density ?= ' I ' (e.g. from a unit Gaussian distribution), we define ? 6 ' G ' to represent the density over X induced by sampling / GLYPH&lt;24&gt; ?= and then computing -= GLYPH&lt;28&gt; ' / ' . The overarching objective of GAN training is to ensure that ? 6 = ? data, i.e. to have -match the true data distribution.

In GANtraining, we alternate between training a discriminator GLYPH&lt;25&gt; : X ! » 0 GLYPH&lt;150&gt; 1 … to distinguish between ? 6 and ? data, and training the generator GLYPH&lt;28&gt; ' I ' to produce samples that appear 'real' to GLYPH&lt;25&gt; ' G ' . Specifically, GLYPH&lt;25&gt; ' G ' is treated as a classifier that predicts the probability that G is a sample from ? data instead of ? 6 . GLYPH&lt;25&gt; ' G ' is trained to minimise the cross-entropy loss

<!-- formula-not-decoded -->

In tandem, GLYPH&lt;28&gt; is trained to maximise !GLYPH&lt;25&gt; -that is, we want to solve max GLYPH&lt;28&gt; min GLYPH&lt;25&gt; !GLYPH&lt;25&gt; . If GLYPH&lt;25&gt; can be an arbitrary function, then min GLYPH&lt;25&gt; !GLYPH&lt;25&gt; is attained at [11, Proposition 1]:

<!-- formula-not-decoded -->

It is known that GLYPH&lt;28&gt; 2 arg max GLYPH&lt;28&gt; min GLYPH&lt;25&gt; !GLYPH&lt;25&gt; if and only if ? 6 = ? data [11, Theorem 1]. As a proof sketch, if ? 6 = ? data then the loss is !GLYPH&lt;25&gt; = log 4. Moreover, it can be shown that min GLYPH&lt;25&gt; !GLYPH&lt;25&gt; is equal to log 4 GLYPH&lt;0&gt; 2 GLYPH&lt;1&gt; JSD ' ? data k ? 6 ' , where JSD ' ? k @ ' denotes the Jensen-Shannon divergence between distributions ? and @ . Since the Jensen-Shannon divergence is always non-negative and becomes zero if and only if ? data = ? 6 , it follows that log 4 is the optimal value of max GLYPH&lt;28&gt; min GLYPH&lt;25&gt; !GLYPH&lt;25&gt; , and that ? 6 = ? data is the only choice of GLYPH&lt;28&gt; that attains this optimum.

## 4.4 Adversarial IRL: A state-centric, GAN-based approximation

GANs can be used to produce an IRL method similar to GCL, called GAN-GCL [5]. However, GANGCL obtains poor performance in practice [8, Table 2]. In particular, performing discrimination over entire trajectories GLYPH&lt;31&gt; can complicate training.

Adversarial IRL (AIRL) [8] instead discriminates between state-action pairs. Concretely, it learns a (stationary) stochastic policy GLYPH&lt;27&gt; ' 0 j B ' and reward function 5 GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' . The AIRL discriminator is defined by:

<!-- formula-not-decoded -->

The generator GLYPH&lt;27&gt; is trained with the discriminator confusion:

<!-- formula-not-decoded -->

The discriminator is trained with cross-entropy loss, updating only the parameters GLYPH&lt;20&gt; corresponding to the reward function 5 GLYPH&lt;20&gt; :

<!-- formula-not-decoded -->

Recall we are working in the setting of infinite horizon MDPs, since AIRL (and GCL) assume stationary policies. Yet the summation in Eq. (107) is undiscounted, and would usually diverge if ) = 1 . For AIRL, we must therefore additionally assume that the transition dynamics are proper , such that the MDP will almost surely enter an absorbing state in finite time. In Eq. (107), the trajectory fragments GLYPH&lt;31&gt; consist of the finite-length component prior to entering an absorbing state, ensuring the loss is well-defined. Unfortunately, omitting the absorbing state from GLYPH&lt;31&gt; also means the discriminator-and therefore reward function-cannot learn an absorbing-state reward.

In the following sections, we summarise the key theoretical results of AIRL. To make it clear where assumptions are being used, we state intermediate lemmas with minimal assumptions. However, all theorems require deterministic dynamics, and the theorems in Section 4.4.2 additionally rely on the dynamics being 'decomposable'.

## 4.4.1 Policy objective is entropy-regularised 5 GLYPH&lt;20&gt;

The reward GLYPH&lt;160&gt; A GLYPH&lt;20&gt; used to train the generator GLYPH&lt;27&gt; ' 0 j B ' is a sum of two widely used GAN objectives: the minimax objective GLYPH&lt;0&gt; log ' 1 GLYPH&lt;0&gt; GLYPH&lt;25&gt; GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 '' and the more widely used log GLYPH&lt;25&gt; GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' objective. Combining the two objectives is reasonable since both objectives have the same fixed point. Moreover, the combined objective has a pleasing interpretation in terms of 5 GLYPH&lt;20&gt; :

<!-- formula-not-decoded -->

Summing over entire trajectories, we obtain the entropy-regularised policy objective:

<!-- formula-not-decoded -->

It is known that the policy GLYPH&lt;27&gt; that maximises this objective is [29, 12]:

<!-- formula-not-decoded -->

## 4.4.2 5 GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' recovers the optimal advantage

Recall that for a fixed generator GLYPH&lt;28&gt; , the optimal discriminator GLYPH&lt;25&gt; is

<!-- formula-not-decoded -->

where ? data ' G ' is the true data distribution (in our case, expert trajectories), and ? 6 ' G ' is the distribution induced by the generator (in our case, policy GLYPH&lt;27&gt; ).

Normally in GAN training, it is computationally efficient to sample from the generator distribution ? 6 but expensive to evaluate the density ? 6 ' G ' of a given sample. Fortunately, in AIRL (like GAN-GCL), density evaluation is cheap since the generator ? 6 ' G ' is defined by a stochastic policy GLYPH&lt;27&gt; ' 0 j B ' , which explicitly defines a distribution.

Suppose the generator is always pitted against the optimal discriminator in Eq. (104). It is known that the generator maximizes the loss in Eq. (107) of the optimal discriminator if and only if ? 6 ' G ' = ? data ' G ' . In our case this is attained when the generator GLYPH&lt;27&gt; ' 0 j B ' is equal to the expert policy GLYPH&lt;27&gt; GLYPH&lt;26&gt; ' 0 j B ' .

The optimal discriminator for an optimal generator will always output 1 2 : i.e. 5 GLYPH&lt;3&gt; GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' = log GLYPH&lt;27&gt; GLYPH&lt;26&gt; ' 0 j B ' . Moreover, the expert policy GLYPH&lt;27&gt; GLYPH&lt;26&gt; is the optimal maximum causal entropy policy for the MDP's reward function A , so GLYPH&lt;27&gt; GLYPH&lt;26&gt; ' 0 j B ' = exp GLYPH&lt;22&gt; soft A ' BGLYPH&lt;150&gt; 0 ' . So, if the GAN converges, then at optimality 5 GLYPH&lt;3&gt; GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' = GLYPH&lt;22&gt; soft A ' BGLYPH&lt;150&gt; 0 ' and GLYPH&lt;27&gt; ' 0 j B ' = GLYPH&lt;27&gt; GLYPH&lt;26&gt; ' 0 j B ' .

## 4.4.3 Reward shaping in MCE RL

In this section, we will introduce a classical result of (hard) optimal policy equivalence under potential shaping due to Ng et al. [18], and then generalise it to the case of maximum entropy (soft) optimal policies.

Definition 4.1. Let A and A 0 be two reward functions. We say A and A 0 induce the same hard optimal policy under transition dynamics T if, for all states B 2 S :

<!-- formula-not-decoded -->

Theorem 4.1. Let A and A 0 be two reward functions. A and A 0 induce the same hard optimal policy under all transition dynamics T if:

<!-- formula-not-decoded -->

for some GLYPH&lt;23&gt; 7 0 and potential-shaping function ) : S ! R .

<!-- formula-not-decoded -->

Definition 4.2. Let A and A 0 be two reward functions. We say they induce the same soft optimal policy under transition dynamics T if, for all states B 2 S and actions 0 2 A :

<!-- formula-not-decoded -->

Theorem 4.2. Let A and A 0 be two reward functions. A and A 0 induce the same soft optimal policy under all transition dynamics T if A 0 ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' = A ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' , GLYPH&lt;15&gt; ) ' B 0 ' GLYPH&lt;0&gt; ) ' B ' for some potential-shaping function ) : S ! R .

Proof. We have:

<!-- formula-not-decoded -->

So:

<!-- formula-not-decoded -->

Thus &amp; soft A 0 GLYPH&lt;150&gt; T ' BGLYPH&lt;150&gt; 0 ' , ) ' B ' satisfies the soft Bellman backup for A , so:

<!-- formula-not-decoded -->

It follows that the optimal advantage is invariant to shaping:

<!-- formula-not-decoded -->

GLYPH&lt;3&gt;

Remark 4.1. Note that rescaling A by GLYPH&lt;23&gt; &lt; 1 changes the soft optimal advantage function and, therefore, the soft-optimal policy. In particular, it approximately rescales the soft optimal advantage function (this is not exact as log-sum-exp is non-linear). Rescaling will therefore tend to have the effect of making the soft-optimal policy less ( GLYPH&lt;23&gt; 7 1 ' or more ( GLYPH&lt;23&gt; 5 1 ) stochastic.

## 4.4.4 Discriminator objective

In this section, we show that minimising the loss of the discriminator corresponds to ME IRL in deterministic dynamics when 5 GLYPH&lt;20&gt; is already an advantage for some reward function.

Theorem4.3. Consider an undiscounted, deterministic MDP. Suppose 5 GLYPH&lt;20&gt; and GLYPH&lt;27&gt; GLYPH&lt;20&gt; are the soft-optimal advantage function and policy for reward function A GLYPH&lt;20&gt; . Then minimising the cross-entropy loss of the discriminator under generator GLYPH&lt;27&gt; GLYPH&lt;20&gt; is equivalent to maximising the log-likelihood of observations under the Maximum Entropy (ME) IRL model.

Specifically, recall that the gradient of ME IRL (with GLYPH&lt;15&gt; = 1 ) log likelihood is

<!-- formula-not-decoded -->

We will show that the gradient of the discriminator objective is

<!-- formula-not-decoded -->

Proof. Note that ! ' GLYPH&lt;20&gt; ' is the discriminator loss, and so we wish to maximise the discriminator objective GLYPH&lt;0&gt; ! ' GLYPH&lt;20&gt; ' . This has the gradient:

<!-- formula-not-decoded -->

Observe that:

Thus:

Similarly:

So:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Recall we train the policy GLYPH&lt;27&gt; ' 0 C j B C ' to maximise Eq. (109). The optimal maximum entropy policy for a given 5 GLYPH&lt;20&gt; is GLYPH&lt;27&gt; GLYPH&lt;3&gt; 5 GLYPH&lt;20&gt; ' 0 C j B C ' = exp GLYPH&lt;22&gt; soft 5 GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' .

By assumption, 5 GLYPH&lt;20&gt; is the advantage for reward function A GLYPH&lt;20&gt; , so:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we write B 0 = T' BGLYPH&lt;150&gt; 0 ' for the deterministic next-state. Restricting ourselves only to feasible transitions ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' , we can alternatively write:

<!-- formula-not-decoded -->

That is, 5 GLYPH&lt;20&gt; is A GLYPH&lt;20&gt; shaped by potential function + soft A ' B ' . Applying Theorem 4.2, it follows that:

<!-- formula-not-decoded -->

but by assumption 5 GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' = GLYPH&lt;22&gt; soft A GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' , so we have that 5 GLYPH&lt;20&gt; ' BGLYPH&lt;150&gt; 0 ' is idempotent under the advantage operator:

<!-- formula-not-decoded -->

Thus the optimal policy is GLYPH&lt;27&gt; GLYPH&lt;3&gt; 5 GLYPH&lt;20&gt; ' 0 C j B C ' = exp 5 GLYPH&lt;20&gt; ' B C GLYPH&lt;150&gt; 0 C ' = GLYPH&lt;27&gt; GLYPH&lt;3&gt; A GLYPH&lt;20&gt; ' 0 C j B C ' . Substituting this expression into Eq. (125) gives

<!-- formula-not-decoded -->

and into Eq. (127) gives

<!-- formula-not-decoded -->

So:

<!-- formula-not-decoded -->

GLYPH&lt;3&gt;

Remark4.2. Section 4.4.2 showed the globally optimal 5 GLYPH&lt;20&gt; is the optimal soft advantage function. However, there is no guarantee that 5 GLYPH&lt;20&gt; is ever a soft advantage function during training. So this theorem does not demonstrate convergence, but does provide intuition for why AIRL often works well in practice.

## 4.4.5 Recovering rewards

In Section 4.4.3, we saw that if a reward function A 0 is a potential shaped version of A then A 0 induces the same soft &amp; -values as A up to a state-only function. In the case that both reward functions are state-only, i.e. A ' B ' and A 0 ' B ' , then potential shaping (when GLYPH&lt;15&gt; 5 1 ' reduces to the special case of A 0 ' B ' = A ' B ' , : for some constant : . Perhaps surprisingly, AIRL can determine state-only rewards up to a constant provided the (deterministic) transition dynamics T satisfies a strong requirement known as the decomposability condition .

Definition 4.3 (Decomposability Condition) . We define two states DGLYPH&lt;150&gt; E 2 S as being 1-step linked under a transition distribution T' B 0 j BGLYPH&lt;150&gt; 0 ' if there exists a state B 2 S and actions 0GLYPH&lt;150&gt; 1 2 A such that D and E are successor states to B : i.e. T' D j BGLYPH&lt;150&gt; 0 ' 7 0 and T' E j BGLYPH&lt;150&gt; 1 ' 7 0 .

We define two states DGLYPH&lt;150&gt; E 2 S as being = , 1 -step linked if they are = -step linked or if there is an intermediate state B 2 S such that D is = -step linked to B and B is 1 -step linked to E .

We define two states DGLYPH&lt;150&gt; E 2 S as being linked if there is some = 2 N for which they are = -step linked.

A transition distribution T is decomposable if all pairs of states in the MDP are linked.

The decomposability condition can be counterintuitive, so we consider some examples before using the definition further. Note that although the decomposability condition is stated in terms of transition probabilities, later applications of this condition will assume deterministic dynamics where probabilities are either 0 or 1.

A simple MDP that does not satisfy the condition is a two-state cyclic MDP, where it is only possible to transition from state GLYPH&lt;22&gt; to GLYPH&lt;23&gt; and vice-versa. There is no state that can reach both GLYPH&lt;22&gt; and GLYPH&lt;23&gt; , so they are not 1-step linked. They are therefore also not = -step linked for any = , since there are no possible intermediary states. However, the MDP would be decomposable if the dynamics were extended to allow self-transitions (from GLYPH&lt;22&gt; ! GLYPH&lt;22&gt; and GLYPH&lt;23&gt; ! GLYPH&lt;23&gt; ).

A similar pattern holds in gridworlds. Imagine a checkerboard pattern on the grid. If all actions move to an adjacent cell (left, right, up or down), then all the successors of white cells are black, and vice-versa. Consequently, cells are only ever 1-step linked to cells of the same colour. Taking the transitive closure, all cells of the same colour are linked together, but never to cells of a different colour. However, if ones adds a 'stay' action to the gridworld then all cells are linked.

We have not been able to determine whether the decomposability condition is satisfied in standard RL benchmarks, such as MuJoCo tasks or Atari games.

We will now show a key application of the decomposability condition: that equality of soft optimal policies implies equality of state-only rewards up to a constant.

Theorem 4.4. Let T be a deterministic dynamics model satisfying the decomposability condition, and let GLYPH&lt;15&gt; 7 0 . Let A ' B ' and A 0 ' B ' be two reward functions producing the same MCE policy in T . That is, for all states B 2 S and actions 0 2 A :

<!-- formula-not-decoded -->

Then A 0 ' B ' = A ' B ' , : , for some constant : .

Proof. Westart by considering the general case of a stochastic dynamics model T andrewardfunctions over ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' triples. We introduce the simplifying assumptions only when necessary, to highlight why we make these assumptions.

Substituting the definition for GLYPH&lt;22&gt; soft in Eq. (138):

<!-- formula-not-decoded -->

So:

<!-- formula-not-decoded -->

where 5 ' B ' = + soft AGLYPH&lt;150&gt; T ' B ' GLYPH&lt;0&gt; + soft A 0 GLYPH&lt;150&gt; T ' B ' . Now:

<!-- formula-not-decoded -->

Contrast this with the Bellman backup on A 0 :

<!-- formula-not-decoded -->

So, equating these two expressions for &amp; soft A 0 GLYPH&lt;150&gt; T ' BGLYPH&lt;150&gt; 0 ' :

<!-- formula-not-decoded -->

In the special case of deterministic dynamics, then if B 0 = T' BGLYPH&lt;150&gt; 0 ' , we have:

<!-- formula-not-decoded -->

This looks like A 0 being a potential-shaped version of A , but note this equality may not hold for transitions that are not feasible under this dynamics model T .

If we now constrain A 0 and A to be state-only, we get:

<!-- formula-not-decoded -->

In particular, since GLYPH&lt;15&gt; &lt; 0 this implies that for a given state B , all possible successor states B 0 (reached via different actions) must have the same value 5 ' B 0 ' . In other words, all 1-step linked states have the same 5 value. Moreover, as 2-step linked states are linked via a 1-step linked state and equality is transitive, they must also have the same 5 values. By induction, all linked states must have the same 5 value. Since by assumption T is decomposable, then 5 ' B ' = 2 for some constant, and so:

<!-- formula-not-decoded -->

GLYPH&lt;3&gt;

Now, we will see how the decomposability condition allows us to make inferences about equality between state-only functions. This will then allow us to prove that AIRL can recover state-only reward functions.

Lemma 4.1. Suppose the transition distribution T is decomposable. Let 0 ' B ' GLYPH&lt;150&gt; 1 ' B ' GLYPH&lt;150&gt; 2 ' B ' GLYPH&lt;150&gt; 3 ' B ' be functions of the state. Suppose that for all states B 2 S , actions 0 2 A and successor states B 0 2 S for which T' B 0 j BGLYPH&lt;150&gt; 0 ' 7 0 , we have

Then for all B 2 S , where : 2 R is a constant.

<!-- formula-not-decoded -->

Base case : Since 5 ' B ' = 6 ' B 0 ' for any successor state B 0 of B , it must be that 6 ' B 0 ' takes on the same value for all successor states B 0 for B . This shows that all 1-step linked states have the same 6 ' B 0 ' .

Inductive case : Moreover, this extends by transitivity. Suppose that all = -step linked states B 0 have the same 6 ' B 0 ' . Let D and E be ' = , 1 ' -step linked. So D is = -step linked to some intermediate B 2 S that is 1-step linked to E . But then 6 ' D ' = 6 ' B ' = 6 ' E ' . So in fact all ' = , 1 ' -step linked states B 0 have the same 6 ' B 0 ' .

By induction, it follows that all linked states B 0 have the same 6 ' B 0 ' . In a decomposable MDP, all states are linked, so 6 ' B ' is equal to some constant : 2 R for all B 2 S . 6 Moreover, in any (infinite-horizon) MDP any state B 2 S must have at least one successor state B 0 2 S (possibly itself). By assumption, 5 ' B ' = 6 ' B 0 ' , so 5 ' B ' = : for all B 2 S . GLYPH&lt;3&gt;

Finally, we can use the preceding result to show when AIRL can recover a state-only reward (up to a constant). Note that even if the ground-truth reward A ' B ' is state-only, the reward network 5 in AIRL must be a function of states and actions (or next states) in order to be able to represent the global minimum: the soft advantage function GLYPH&lt;22&gt; soft ' BGLYPH&lt;150&gt; 0 ' . The key idea in the following theorem is to decompose 5 into a state-only reward 6 GLYPH&lt;20&gt; ' B ' and a potential shaping term. This gives 5 capacity to represent GLYPH&lt;22&gt; soft ' BGLYPH&lt;150&gt; 0 ' , while the structure ensures 6 GLYPH&lt;20&gt; ' B ' equals the ground-truth reward up to a constant.

Theorem 4.5. Suppose the reward network is parameterized by

<!-- formula-not-decoded -->

Suppose the ground-truth reward is state-only, A ' B ' . Suppose moreover that the MDP has deterministic dynamics T satisfying the decomposability condition, and that GLYPH&lt;15&gt; 7 0 . Then if 5 GLYPH&lt;20&gt; GLYPH&lt;3&gt; GLYPH&lt;150&gt; ) GLYPH&lt;3&gt; is the reward network for a global optimum ' GLYPH&lt;25&gt; GLYPH&lt;20&gt; GLYPH&lt;3&gt; GLYPH&lt;150&gt; ) GLYPH&lt;3&gt; GLYPH&lt;150&gt; GLYPH&lt;27&gt; ' of the AIRL problem, that is 5 GLYPH&lt;20&gt; GLYPH&lt;3&gt; GLYPH&lt;150&gt; ) GLYPH&lt;3&gt; ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' = 5 GLYPH&lt;3&gt; ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' , we have:

<!-- formula-not-decoded -->

where : 1 GLYPH&lt;150&gt; : 2 2 R are constants and B 2 S is a state.

Proof. We know the global minimum 5 GLYPH&lt;3&gt; ' BGLYPH&lt;150&gt; 0GLYPH&lt;150&gt; B 0 ' = GLYPH&lt;22&gt; soft ' BGLYPH&lt;150&gt; 0 ' , from Section 4.4.2. Now:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

6 Note that in a decomposable MDP all states are the successor of at least one state.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

So for all states B 2 S , actions 0 2 A and resulting deterministic successor states B 0 = T' BGLYPH&lt;150&gt; 0 ' we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now applying Lemma 4.1 with 0 ' B ' = 6 GLYPH&lt;20&gt; GLYPH&lt;3&gt; ' B ' GLYPH&lt;0&gt; GLYPH&lt;17&gt; ) GLYPH&lt;3&gt; ' B ' , 2 ' B ' = A ' B ' GLYPH&lt;0&gt; + soft A ' B ' , 1 ' B 0 ' = GLYPH&lt;15&gt; + soft A ' B 0 ' and 3 ' B 0 ' = GLYPH&lt;15&gt; GLYPH&lt;17&gt; ) GLYPH&lt;3&gt; ' B 0 ' gives:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where : 0 2 R is a constant. Rearranging and using GLYPH&lt;15&gt; &lt; 0 we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bythedecomposability condition, all states B 2 S are the successor of some state (possibly themselves). So we can apply Eq. (161) to all B 2 S :

<!-- formula-not-decoded -->

Finally, applying Eq. (162) to Eq. (160) yields:

<!-- formula-not-decoded -->

GLYPH&lt;3&gt;

as required.

Remark 4.3. Note this theorem makes several strong assumptions. In particular, it requires that 5 attains the global minimum, but AIRL is not in general guaranteed to converge to a global optimum. Additionally, many environments have stochastic dynamics or are not 1-step linked.

Note that in stochastic dynamics there may not exist any function 5 ' BGLYPH&lt;150&gt; B 0 ' that is always equal to GLYPH&lt;22&gt; soft ' BGLYPH&lt;150&gt; 0 ' . This is because there may exist 0GLYPH&lt;150&gt; 0 0 such that T'GLYPH&lt;1&gt; j BGLYPH&lt;150&gt; 0 ' and T'GLYPH&lt;1&gt; j BGLYPH&lt;150&gt; 0 0 ' differ but both have support for B 0 while GLYPH&lt;22&gt; soft ' BGLYPH&lt;150&gt; 0 ' &lt; GLYPH&lt;22&gt; soft ' BGLYPH&lt;150&gt; 0 0 ' .

## 5 Conclusion

We have described three Inverse Reinforcement Learning (IRL) algorithms: the tabular method Maximum Causal Entropy (MCE) IRL; and deep-learning based, dynamics-free algorithms Guided Cost Learning (GCL) and Adversarial IRL (AIRL). We have shown that MCE IRL can be derived from maximising entropy under a feature expectation matching constraint. Furthermore, we have shown how this is equivalent to maximising the likelihood of the data. Finally, we have explained how GCL and AIRL can both be viewed as extensions of this maximum likelihood solution to settings with unknown dynamics and potentially continuous state and action spaces.

While contemporary methods such as GCL and AIRL have many points in common with MCE IRL, the connection is far from exact. For example, the discriminator objective of AIRL only aligns with that of MCE IRL in undiscounted MDPs, yet it is routinely applied to discounted MDPs. One promising direction for future work would be to derive a dynamics-free algorithm using function approximation directly from the original MCE IRL approach. We consider it probable that such an algorithm would provide more stable performance than existing heuristic approximations to MCE IRL.

## Acknowledgements

We would like to thank Alyssa Li Dayan, Michael Dennis, Yawen Duan, Daniel Filan, Erik Jenner, Niklas Lauffer and Cody Wild for feedback on earlier versions of this manuscript.

## References

- [1] Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In ICML , 2004.
- [2] Kareem Amin, Nan Jiang, and Satinder Singh. Repeated inverse reinforcement learning. In NIPS , 2017.
- [3] Stephen Boyd and Lieven Vandenberghe. Convex optimization . Cambridge University Press, 2004.
- [4] Haoyang Cao, Samuel N. Cohen, and Lukasz Szpruch. Identifiability in inverse reinforcement learning. arXiv: 2106.03498v2 [cs.LG], 2021.
- [5] Chelsea Finn, Paul Christiano, Pieter Abbeel, and Sergey Levine. A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models. arXiv: 1611.03852v3 [cs.LG], 2016.
- [6] Chelsea Finn, Sergey Levine, and Pieter Abbeel. Guided cost learning: Deep inverse optimal control via policy optimization. In ICML , 2016.
- [7] Roy Fox, Ari Pakman, and Naftali Tishby. Taming the noise in reinforcement learning via soft updates. In UAI , 2016.
- [8] Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning. In ICLR , 2018.
- [9] Adam Gleave, Pedro Freire, Steven Wang, and Sam Toyer. seals: Suite of environments for algorithms that learn specifications. https://github.com/HumanCompatibleAI/seals , 2020.
- [10] AdamGleave, MichaelDennis, Shane Legg, Stuart Russell, and Jan Leike. Quantifying differences in reward functions. In ICLR , 2021.
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NIPS , 2014.
- [12] Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, and Sergey Levine. Reinforcement learning with deep energy-based policies. In ICML , 2017.
- [13] Hong Jun Jeon, Smitha Milli, and Anca Dragan. Reward-rational (implicit) choice: A unifying formalism for reward learning. In NeurIPS , 2020.
- [14] Andrej Karpathy. Keynote talk. CVPR Workshop on Scalability in Autonomous Driving, 2020.
- [15] Kuno Kim, Shivam Garg, Kirankumar Shiragur, and Stefano Ermon. Reward identification in inverse reinforcement learning. In ICML , 2021.
- [16] Ilya Kostrikov, Kumar Krishna Agrawal, Debidatta Dwibedi, Sergey Levine, and Jonathan Tompson. Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning. In ICLR , 2019.
- [17] Andrew Y Ng and Stuart J Russell. Algorithms for inverse reinforcement learning. In ICML , 2000.
- [18] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In ICML , 1999.

- [19] Deepak Ramachandran and Eyal Amir. Bayesian inverse reinforcement learning. In ĲCAI , 2007.
- [20] Dorsa Sadigh, Anca D. Dragan, S. Shankar Sastry, and Sanjit A. Seshia. Active preference-based learning of reward functions. In RSS , 2017.
- [21] Rohin Shah, Dmitrii Krasheninnikov, Jordan Alexander, Pieter Abbeel, and Anca Dragan. Preferences implicit in the state of the world. In ICLR , 2019.
- [22] Joar Skalse, Matthew Farrugia-Roberts, Stuart Russell, Alessandro Abate, and Adam Gleave. Invariance in policy optimisation and partial identifiability in reward learning. arXiv:2203.07475v1 [cs.LG], 2022.
- [23] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction . MITPress, 2018.
- [24] Tesla. Upgrading Autopilot: Seeing the world in radar. https://www.tesla.com/blog/ upgrading-autopilot-seeing-world-radar , 2016. Accessed: 2020-07-27.
- [25] Flemming Topsøe. Information-theoretical optimization techniques. Kybernetika , 1979.
- [26] Aaron Tucker, Adam Gleave, and Stuart Russell. Inverse reinforcement learning for video games. arXiv: arXiv:1810.10593v1 [cs.LG], 2018.
- [27] Mel Vecerik, Oleg Sushkov, David Barker, Thomas Rothörl, Todd Hester, and Jon Scholz. A practical approach to insertion with variable socket position using deep reinforcement learning. In ICRA , 2019.
- [28] Markus Wulfmeier, Peter Ondruska, and Ingmar Posner. Maximum entropy deep inverse reinforcement learning. arXiv: 1507.04888v3 [cs.LG], 2015.
- [29] Brian D Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy . PhD thesis, CMU, 2010.
- [30] Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, and Anind K Dey. Maximum entropy inverse reinforcement learning. In AAAI , 2008.
- [31] Brian D. Ziebart, J. Andrew Bagnell, and Anind K. Dey. Modeling interaction via the principle of maximum causal entropy. In ICML , 2010.