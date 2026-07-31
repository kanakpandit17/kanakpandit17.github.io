---
title: "Squeezing a Million Mesh Points into 512 Tokens using Universal Physics Transformers"
date: 2026-07-25
permalink: /posts/2026/07/universal-physics-transformers/
excerpt: "Neural operators are meant to replace slow numerical PDE solvers, but most of them slow down as the mesh gets finer because every layer processes every mesh point. Universal Physics Transformers compress the whole simulation into a fixed number of tokens first, then run the dynamics there. A walkthrough of the architecture, the experiments and where the method falls short."
tags:
  - neural-operators
  - transformers
  - computational-fluid-dynamics
  - scientific-machine-learning
  - PDE
mathjax: true
---

*Based On **Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators**, by Alkin, Fürst, Schmid, Gruber, Holzleitner and Brandstetter, NeurIPS 2024 [[1]](#ref-1). Written for the Deep Learning for Sciences seminar.*

---

## Abstract
{:.no_toc}



Neural operators were designed to speed up numerical PDE solvers but most of them still hit the same wall. As the number of mesh points grows and hence does the computational cost, which actually makes large scientific simulations hard to handle. The Universal Physics Transformer (UPT) takes a different route. An encoder compresses a whole simulation snapshot into a fixed number of latent tokens (64 for a car surface, 512 for a pipe flow), a transformer predicts how the system evolves inside that compact space and a query-based decoder reconstructs the physical field wherever you ask for it. Two auxiliary objectives, inverse encoding and inverse decoding, keep the latent state anchored to the physics, so the model can roll out for a long time without decoding back into the physical domain at every step. On a transient pipe-flow benchmark UPT reaches 400× the throughput of OpenFOAM. On ShapeNet-Car it matches a GINO running on a $$64^3$$ latent grid using only 64 latent tokens, with GPU memory down from 19.8 GB to 0.6 GB. This post works through the architecture, the two inverse losses, the three experiments and where I think the approach still falls short.

## Contents
{:.no_toc}

* placeholder
{:toc}

---

## 1. Why we are still waiting on simulations

Designing a faster car or predicting tomorrow's weather sounds like two completely different problems. Surprisingly, they're powered by the same mathematics. The frustrating part is that we already know the physics. The Navier-Stokes equations have described fluid motion for nearly two centuries. 
What still slows us down isn't understanding the equations... it's solving them numerically. For complex simulations, that's the step that can take days or demand an entire room filled with high-performance computers.

For incompressible flow, the velocity field $$\mathbf{u}(\mathbf{x}, t)$$ and pressure $$p(\mathbf{x}, t)$$ satisfy

$$
\frac{\partial \mathbf{u}}{\partial t} = -\underbrace{(\mathbf{u} \cdot \nabla)\mathbf{u}}_{\text{convection}} + \underbrace{\mu \nabla^2 \mathbf{u}}_{\text{viscosity}} - \underbrace{\nabla p}_{\text{pressure}} + \underbrace{\mathbf{f}}_{\text{body forces}}
$$

together with the incompressibility constraint

$$
\nabla \cdot \mathbf{u} = 0 .
$$

Three things about this system make it expensive, and they compound.

The convection term $$(\mathbf{u} \cdot \nabla)\mathbf{u}$$ is nonlinear, so a tiny disturbance in the flow can grow and spread through the whole domain instead of staying local. Pressure is worse. It has no time evolution equation of its own, so it cannot be marched forward from one timestep to the next. It has to be solved for at every single step so that the velocity field stays divergence-free, and because pressure at one location depends on the entire flow field, that solve is global. Turbulence then piles on top. At the Reynolds numbers of most engineering problems the fluid fills with vortices and eddies across a wide range of sizes, and resolving them needs either a very fine mesh or a turbulence model. Both cost.

Solvers such as OpenFOAM handle this by splitting the domain into hundreds of thousands of small cells and marching the equations forward with a scheme like PISO [[3]](#ref-3), in steps small enough to stay stable. Here is what that costs on the dataset this paper uses. One hundred seconds of simulated 2D pipe flow takes between **2,000 and 200,000 internal solver steps** and about **120 seconds of wall clock on 16 CPUs**, and that is a two-dimensional pipe with a few circles in it. A realistic 3D case such as airflow over a vehicle runs into days.

<figure>
  <img src="/images/timestep-comparison.png" alt="Classical solver taking many small steps versus a neural surrogate taking a few large ones">
  <figcaption><b>Figure 1.</b> The classical solver is actually stuck taking small and stable steps \(\Delta t\). A learned surrogate trains on these snapshots \(K\Delta t\) apart and jumps straight between them.</figcaption>
</figure>

The saving in Figure 1 is not that each step gets cheaper. It is that almost every step disappears.

A neural surrogate makes a very different trade-off. Instead of solving the governing equations step by step it learns to look at one snapshot and predict the state $$K\Delta t$$ into the future, skipping every intermediate computation. Whether that trade-off pays off depends almost entirely on how well the model scales, and that's the central question this paper tries to answer.

---

## 2. Background

### 2.1 Why ordinary supervised learning doesn't fit

Standard supervised learning fits a function $$f_\theta : \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$$ between finite dimensional vectors, treating every training example as an independent sample.

Physics doesn't work that way. What we want to predict is not a vector but a *field*, an element of an infinite-dimensional function space, and neighbouring points are not independent. Change the pressure at one location and the surrounding region responds, because every point is tied to every other one through the governing equations.

Computers cannot store a continuous function, so we discretize. We sample the domain $$\Omega$$ at $$M$$ locations $$\{\mathbf{x}_1, \dots, \mathbf{x}_M\}$$ and keep the field values only there. Sampling more densely brings the finite representation closer to the true field, but every extra point costs memory and compute.

<figure>
  <img src="/images/discretization.png" alt="A continuous function reduced to a finite set of samples">
  <figcaption><b>Figure 2.</b> Discretization. The continuous field (Eg. Temperature) on the left holds infinitely-many values. The solver never sees the original continuous function. It only works with this finite collection of sampled points on the right.</figcaption>
</figure>

Worse, the model you train ends up quietly tied to whichever sampling you happened to choose. The physics does not care whether you use a structured grid, an unstructured mesh or particles, but the network does. One trained on a $$64 \times 64$$ grid usually struggles when handed a $$128 \times 128$$ grid, a tetrahedral mesh or an SPH particle cloud, which means retraining every time the discretization changes.


### 2.2 Operator learning

Neural operators get around this by learning **relationships between functions** instead of between fixed-size vectors. Rather than memorizing solutions for one particular mesh, they learn the physical operator itself, which is what lets a single model generalize across discretizations [[4]](#ref-4).

Let $$\mathcal{A}$$ be the space of all possible **input functions** and $$\mathcal{U}$$ the space of all possible **output functions**. Both are **Banach spaces**, meaning spaces whose elements are entire functions rather than numbers or vectors. A neural operator learns a map between them:

$$
\mathcal{G} : \mathcal{A} \to \mathcal{U}, \qquad a \mapsto u
$$

Here $$a$$ is an input function such as an initial velocity field, a pressure field or a boundary condition, $$u$$ is the corresponding solution once the governing equations have been solved and $$\mathcal{G}$$ is the operator that turns one into the other. The true $$\mathcal{G}$$ is unknown, so we approximate it with a network $$\mathcal{G}_\theta$$ and fit its parameters $$\theta$$ by minimising the expected squared error over input functions drawn from the training distribution $$\nu$$:

$$
\min_{\theta} \; \mathbb{E}_{a \sim \nu} \big[ \| \mathcal{G}(a) - \mathcal{G}_\theta(a) \|_{\mathcal{U}}^2 \big]
$$

The norm here is taken over the whole output *function*, not over a fixed vector of samples, and $$\mathcal{G}(a)$$ is the ground truth from a numerical solver. For time-dependent systems the operator maps the current state and a timestep to the next state,

$$
\mathcal{G} : \big(u(\cdot, t), \Delta t\big) \mapsto u(\cdot, t + \Delta t),
$$

where the dot in $$u(\cdot, t)$$ stands for **all spatial locations** in $$\Omega$$ rather than a single point. Instead of stepping through the governing equations thousands of times, the operator learns this evolution straight from data.

<figure>
  <img src="/images/neural-operator.png" alt="A neural operator mapping an input function to an output function">
  <figcaption><b>Figure 3.</b> A neural operator learns a mapping from one function to another rather than from one vector to another. This allows the same model to operate across different discretizations of the same physical system.</figcaption>
</figure>

Since $$\mathcal{G}_\theta$$ is defined on functions, it should not care how you sampled them. The Fourier Neural Operator [[5]](#ref-5) builds this out of a spectral convolution,

$$
(\mathcal{K}v)(\mathbf{x}) = \mathcal{F}^{-1}\big( R_\phi \cdot \mathcal{F}(v) \big)(\mathbf{x}),
$$

learning a complex multiplier $$R_\phi$$ on the lowest Fourier modes. Graph-based solvers take the other route and handle irregular meshes by passing messages along edges [[6]](#ref-6), [[7]](#ref-7).

Both approaches work. Both share the same problem: cost grows with the number of mesh points, so both eventually run into the same wall.

---

## 3. The memory wall

This is the part that motivates everything else, so it is worth writing the scaling down properly instead of gesturing at it. Let $$M$$ be the number of mesh points or particles.

**Table 1.** Per-layer cost of the main neural operator families, with $$M$$ the number of mesh points or particles. Every row grows with the discretization, which is the wall UPT is built to get around. Complexities are those reported in the respective architecture papers, message passing [[6]](#ref-6), [[7]](#ref-7), attention [[11]](#ref-11), FNO [[5]](#ref-5) and GINO [[8]](#ref-8), collected here alongside the scaling analysis in Alkin et al. [[1]](#ref-1).

| Model family | Cost of one layer | What breaks |
| :--- | :--- | :--- |
| GNN message passing | $$\mathcal{O}(\|E\|) = \mathcal{O}(M \bar{d})$$ | Edge list for millions of nodes blows past GPU memory |
| Vanilla transformer | $$\mathcal{O}(M^2 h)$$ | Quadratic attention over every mesh point |
| CNN / U-Net on a 3D grid | $$\mathcal{O}(N^3)$$ at resolution $$N$$ | Cubic growth, and you have to interpolate the mesh onto a grid first |
| FNO on a 3D grid | $$\mathcal{O}(G \log G)$$, $$G = N^3$$ | Same cubic $$G$$, same interpolation |
| GINO | $$\mathcal{O}(M \bar{d} + G \log G)$$ | Latent grid dominates, and $$G \gg M$$ in practice |

In the table, $$M$$ counts mesh vertices or SPH particles and $$\bar{d}$$ is the average number of neighbours each one exchanges messages with, so a graph has roughly $$\|E\| \approx M \bar{d}$$ edges and message passing costs that much per layer. For the grid-based rows, $$N$$ is the resolution along one spatial axis and $$G = N^3$$ is the total number of cells in a structured 3D grid.

That last row deserves a closer look, because GINO [[8]](#ref-8) was built for precisely this situation. It uses a graph neural operator to lift an irregular mesh onto a regular latent grid, runs an FNO there, then maps back. Reasonable plan. The latent grid is where it falls apart. At a fairly modest $$N = 64$$ you get

$$
G = 64^3 = 262{,}144 \quad \text{latent points},
$$

while a ShapeNet-Car surface mesh has roughly $$3{,}586$$ points. So the "compressed" representation is **73× bigger than the input**. Most of those cells are empty air around a car.

This isn't a limitation of just one architecture. The same memory bottleneck shows up across modern deep learning whenever the input grows too large [[9]](#ref-9). In scientific computing it arrives in the shape of mesh points.

<figure>
  <img src="/images/memory-wall.png" alt="GPU memory versus problem scale for GNN, transformer, GINO-2D, GINO-3D and UPT" style="width:55%; margin:0 auto;">
  <figcaption><b>Figure 4.</b> Peak GPU memory against problem scale, where scale 1 is 32K input points and scale 128 is roughly 4.2M. Figure from Alkin et al. [<a href="#ref-1">1</a>].</figcaption>
</figure>

As Figure 4 shows, GNNs and plain transformers exhaust an 80GB A100 by scale 2 and even GINO runs out well before the largest problem sizes. UPT is the only curve still on the chart at scale 128, which is roughly 4.2 million points.



The pattern is the same everywhere. Whichever architecture you pick, its most expensive part still processes a token count that grows with the discretization, $$n_{\text{tokens}} \propto M$$. The idea behind UPT is to break that proportionality.

---

## 4. Method: Universal Physics Transformers

### 4.1 Setting up the problem

Take a trajectory indexed by $$i$$. At time $$t$$ we observe the field at $$k$$ locations,

$$
\mathbf{u}^t_{i,k} \in \mathbb{R}^{k \times d}, \qquad \mathbf{x}^1_i, \dots, \mathbf{x}^k_i \in \Omega \subset \mathbb{R}^n .
$$

We want the field at time $$t'$$ evaluated at $$k'$$ *query* locations $$\mathbf{y}^1_i, \dots, \mathbf{y}^{k'}_i$$. Note that the query set is allowed to be completely unrelated to the input set. That freedom matters more than it looks.

UPT splits the solution operator into three learned pieces:

$$
\mathcal{G}_\theta = \mathcal{D} \circ \mathcal{A} \circ \mathcal{E}
$$

$$
\underbrace{\mathcal{E} : \mathbf{u}^t_{i,k} \mapsto \mathbf{z}^t_i}_{\text{encoder}} \qquad
\underbrace{\mathcal{A} : \mathbf{z}^t_i \mapsto \mathbf{z}^{t'}_i}_{\text{approximator}} \qquad
\underbrace{\mathcal{D} : (\mathbf{z}^{t'}_i, \mathbf{y}) \mapsto \hat{\mathbf{u}}^{t'}_{i,k'}}_{\text{decoder}}
$$

with the latent state living in

$$
\mathbf{z}^t_i \in \mathbb{R}^{n_{\text{latent}} \times h}, \qquad n_{\text{latent}} \; \text{fixed, independent of } k .
$$



<figure>
  <img src="/images/upt-schematic.png" alt="Schematic of the UPT learning paradigm">
  <figcaption><b>Figure 5.</b> UPT uses the same encoder for both meshes and particle clouds, evolves the entire system inside a fixed-size latent space, and reconstructs the physical field by querying the decoder at any desired spatial location. Figure from Alkin et al. [<a href="#ref-1">1</a>].</figcaption>
</figure>

Figure 5 is worth holding in mind for the rest of this section.

### 4.2 Encoder

The encoder has the hardest job in the entire architecture. It must compress anything from a few thousand to millions of input points into just a few hundred latent tokens, all without performing expensive quadratic operations on the full input. It achieves this in four stages.

<figure>
  <img src="/images/encoder-pipeline.png" alt="Encoder pipeline: subsample, select supernodes, message passing, aggregate, compress" style="width:75%; margin:0 auto;">
  <figcaption><b>Figure 6.</b> The UPT encoder compresses millions of input points into a fixed number of latent tokens through four stages. Random subsampling provides data augmentation, supernodes aggregate local information via message passing, a transformer captures global interactions and perceiver pooling compresses the representation into a fixed latent budget.</figcaption>
</figure>

Figure 6 traces the compression from left to right. The token count shrinks at every stage while the expensive operations only ever run on the small end of that chain.

**Stage 1, embedding.** Every one of the $$k$$ input points carries two pieces of information: the value of the physical field there, and where "there" actually is. Both matter, so both go in. A linear layer lifts the field value up to the model's working width $$h$$ and a sinusoidal encoding of the point's coordinates is added on top of it. Each point leaves this stage as a single vector that knows what the physics is doing and where it's happening.

**Stage 2, supernode message passing.** Next, choose a subset of those points, $$n_s$$ of them, and call them supernodes. They aren't picked at random. They're chosen so the character of the mesh survives the cut, which means densely meshed regions keep more supernodes and sparse regions keep fewer. Each supernode then gathers information from every input point within a fixed radius of it, through a message passing layer [[10]](#ref-10).

One detail does the heavy lifting here. Messages travel only *towards* the supernodes and never back out, so the model never computes anything at the remaining $$k - n_s$$ points. This is the only stage in the whole architecture that touches all $$k$$ inputs, and because of that one-way flow its cost grows linearly with them rather than quadratically.

**Stage 3, global mixing.** The simulation is now down to $$n_s$$ supernodes, which go through a stack of standard pre-normalization transformer blocks [[11]](#ref-11). Message passing only moved information within local neighbourhoods, whereas self-attention lets it cross the whole domain, so the model can capture long-range effects such as the wake behind one obstacle changing the flow around another. Self-attention is normally the bottleneck because its cost is quadratic in tokens, but here it runs on the $$n_s$$ supernodes rather than the $$k$$ input points, giving $$\mathcal{O}(n_s^2)$$ instead of $$\mathcal{O}(k^2)$$. With $$n_s$$ fixed at 2,048, this stage costs the same whether the simulation had 30,000 mesh points or several million.

**Stage 4, perceiver pooling.** One last squeeze. A small set of $$n_{\text{latent}}$$ learned query vectors cross-attends into the supernode tokens [[12]](#ref-12) and pulls everything down into the final latent representation. The important part is that these queries are parameters of the model rather than anything derived from the input, which is precisely why the output has a fixed size no matter what went in. In the transient experiment the full chain runs from roughly 59,000 mesh points to 2,048 supernodes to 512 latent tokens, about 115× compression.

Adding the four stages together, the encoder costs

$$
\mathcal{O}\big( \underbrace{k \bar{d}}_{\text{message passing}} + \underbrace{n_s^2}_{\text{transformer}} + \underbrace{n_s \, n_{\text{latent}}}_{\text{pooling}} \big)
$$

where $$\bar{d}$$ is the average neighbourhood size. Linear in mesh size. Quadratic only in numbers you chose yourself.

### 4.3 Approximator

The approximator is a transformer mapping one latent state to the next,

$$
\mathcal{A} : \mathbb{R}^{n_{\text{latent}} \times h} \to \mathbb{R}^{n_{\text{latent}} \times h}, \qquad \mathbf{z}^{t + \Delta t} = \mathcal{A}(\mathbf{z}^t),
$$

and you can apply it over and over:

$$
\mathbf{z}^{t + K\Delta t} = \mathcal{A}^{K}(\mathbf{z}^t).
$$

Cost per step is $$\mathcal{O}(n_{\text{latent}}^2 h)$$. Look at that expression and notice what's absent: there's no $$k$$ in it anywhere.

Boundary conditions and the timestep get injected through DiT-style adaptive layer norm [[13]](#ref-13). A conditioning vector $$\mathbf{c} = \text{emb}(\Delta t) + \text{emb}(\text{bc})$$ produces per-block scale, shift and gate parameters:

$$
\text{modulate}(\mathbf{h}; \mathbf{c}) = \alpha(\mathbf{c}) \odot \Big[ \big(1 + \beta(\mathbf{c})\big) \odot \text{LN}(\mathbf{h}) + \delta(\mathbf{c}) \Big].
$$

That's how one trained model covers inflow velocities from 0.01 to 0.06 m/s instead of needing a network per setting.

### 4.4 Decoder

The decoder inverts the pooling. Query positions become attention queries, latent tokens supply keys and values:

$$
\hat{\mathbf{u}}(\mathbf{y}) = \text{MLP}\!\left( \text{softmax}\!\left( \frac{\big(\gamma(\mathbf{y}) W_Q\big)\big(\mathbf{z}^{t'} W_K\big)^\top}{\sqrt{d_k}} \right) \mathbf{z}^{t'} W_V \right)
$$

Each query attends to the latent tokens independently, so the cost is

$$
\mathcal{O}(k' \, n_{\text{latent}}),
$$

linear in the number of output locations $$k'$$. Queries never attend to each other, which is what keeps this cheap when a lot of predictions are wanted.

Most of UPT's flexibility comes from here. You can reconstruct on the original 59,000 mesh points, on a uniform grid the model never saw during training, or at a single point behind one obstacle. The decoder only needs coordinates. This is also what makes UPT a **neural operator** rather than a mesh-to-mesh regression model: the latent state stands for a continuous field that can be sampled anywhere, not a fixed set of output slots.

<figure>
  <img src="/images/encoder-decoder-block.png" alt="Block diagram of encoder, approximator and decoder with data and learnable components marked">
  <figcaption><b>Figure 7.</b> Input positions and query positions are provided to the model as separate inputs. This separation is what makes UPT discretization agnostic. The encoder can process one discretization while the decoder reconstructs the solution on an entirely different one, allowing the model to generalize across different meshes, grids and particle clouds without retraining. Figure from Alkin et al. [<a href="#ref-1">1</a>].</figcaption>
</figure>

### 4.5 The two inverse losses

A straightforward version of this architecture runs into an important problem.

If the model is only ever trained to predict the next state, nothing forces the latent representation $$\mathbf{z}$$ to capture the physics. It only has to carry enough information for the decoder to get a single timestep right. During a long autoregressive rollout the approximator keeps transforming $$\mathbf{z}$$, small errors accumulate and the latent state drifts into regions the decoder was never trained to interpret. Accuracy then falls off quickly and the rollout goes unstable.

UPT pins the latent space down from both directions instead.

**Inverse encoding.** After the encoder produces a latent state, the decoder has to reconstruct the original input at the same spatial locations it came from. That forces the encoder to keep whatever is needed to recover the original physical state rather than discarding it during compression.

**Inverse decoding.** The predicted output field is passed back through the encoder, and the latent state that comes out has to match the one the approximator produced. Decoding and immediately re-encoding should be close to a no-op, which is what keeps the latent trajectory alongside the physical trajectory instead of drifting away from it over a long horizon.

Both run alongside the ordinary prediction loss on the decoded field, and the final objective is the sum of the three.

<figure>
  <img src="/images/inverse-losses.png" alt="Diagram showing the three loss terms connecting data, latent and model components">
  <figcaption><b>Figure 8.</b> Training strategy used by UPT to enable stable latent-space rollouts through inverse encoding and inverse decoding losses. Figure from Alkin et al. [<a href="#ref-1">1</a>].</figcaption>
</figure>

### 4.6 Latent rollout

With the latent space stabilized, inference can be reorganised. A conventional autoregressive rollout repeats the same cycle at every timestep: encode the current state, step it forward, decode back into the physical domain, then encode again before the next step.

$$
\hat{\mathbf{u}}^{t + K\Delta t} = \big( \mathcal{D} \circ \mathcal{A} \circ \mathcal{E} \big)^{K} (\mathbf{u}^t), \qquad
\text{cost} = K\big(C_{\mathcal{E}} + C_{\mathcal{A}} + C_{\mathcal{D}}\big).
$$

**Latent rollout** encodes the input once at the start, advances the latent state through time entirely in the latent space and calls the decoder only when a prediction is actually wanted.

$$
\hat{\mathbf{u}}^{t + K\Delta t} = \mathcal{D}\big( \mathcal{A}^{K}(\mathcal{E}(\mathbf{u}^t)), \, \mathbf{y} \big), \qquad
\text{cost} = C_{\mathcal{E}} + K \, C_{\mathcal{A}} + C_{\mathcal{D}} .
$$

Both $$C_{\mathcal{E}}$$ and $$C_{\mathcal{D}}$$ grow with the size of the mesh while $$C_{\mathcal{A}}$$ depends only on the fixed latent size, so the saving grows with both the rollout length $$K$$ and the mesh resolution. On the paper's 100-step rollouts over roughly 59,000 mesh points it comes close to an order of magnitude.

Latent rollout also fixes a problem specific to particles. The encoder needs particle positions, but during inference the future positions are exactly what has not been predicted yet, which makes repeated encode-decode cycles awkward. Latent rollout only needs the positions at the first timestep.

---

## 5. Experiments

The experiments focus on three questions: Is the latent representation sufficient? Does the model scale to large simulations? And can it generalize beyond Eulerian meshes to Lagrangian particle simulations?

### 5.1 Steady state flows: is 64 tokens really enough?

**Setup.** ShapeNet-Car [[14]](#ref-14): 889 car geometries, roughly 3.6K surface mesh points each, 700 for training and 189 for testing. The dataset is small, so everything overfits past a point. Model sizes were picked as the largest that still generalize, which puts UPT and GINO around 300M parameters and FNO and U-Net between 15M and 100M.

**Results.**

**Table 2.** Steady-state results on ShapeNet-Car. Best result in each block is bold. The first block is the one to read closely: UPT matches GINO's accuracy using 64 latent tokens against roughly 262,000, and a thirty-third of the memory. All values reported by Alkin et al. [[1]](#ref-1).

| Model | SDF resolution | Latent tokens | MSE ($$\times 10^{-2}$$) $$\downarrow$$ | Memory (GB) $$\downarrow$$ |
| :--- | :---: | :---: | :---: | :---: |
| U-Net | 0 | $$64^3$$ | 6.13 | 1.3 |
| FNO | 0 | $$64^3$$ | 4.04 | 3.8 |
| GINO | 0 | $$64^3$$ | 2.34 | 19.8 |
| **UPT** | **0** | **64** | **2.31** | **0.6** |
| U-Net | 32 | $$32^3$$ | 3.66 | 0.2 |
| FNO | 32 | $$32^3$$ | 3.31 | 0.5 |
| GINO | 32 | $$32^3$$ | 2.90 | 2.1 |
| **UPT** | **32** | $$8^3 + 1024$$ | **2.35** | 2.7 |
| U-Net | 64 | $$64^3$$ | 2.83 | 1.3 |
| FNO | 64 | $$64^3$$ | 3.26 | 3.8 |
| **GINO** | 64 | $$64^3$$ | **2.14** | 19.8 |
| UPT | 64 | $$8^3 + 1024$$ | 2.24 | 2.7 |

The first block is the most striking result in the paper. GINO needs a $$64^3 = 262{,}144$$ point latent grid and 19.8 GB of GPU memory to reach 2.34, while UPT gets a slightly better **2.31** out of **64 latent tokens** and **0.6 GB**. That is roughly **4,000× fewer latent elements** and **33× less memory** for the same accuracy.

The same gap shows up in training time. A GINO epoch takes around **900 seconds** against **4 seconds** for a comparable UPT, a factor of about **225**, which is the difference between testing a handful of configurations and testing a hundred.

There is a trade-off, though. Given a high-resolution $$64^3$$ signed distance function, GINO takes the benchmark outright with **2.14** against UPT's **2.24**, so a dense grid representation does still buy a little accuracy. The question is whether that last 0.10 is worth 19.8 GB against 2.7 GB, and for anything sitting inside a design loop I don't think it is.

### 5.2 Transient flows: does it scale?

**Setup.** The authors generated **10,000 transient CFD simulations** with OpenFOAM's `pisoFoam` solver [[2]](#ref-2), split into 8,000 training, 1,000 validation and 1,000 test. Each one is flow through a pipe containing between one and four circular obstacles of random size and position, with inlet velocity sampled between **0.01 and 0.06 m/s**. Every simulation runs 100 seconds and stores **100 snapshots** of the pressure field and the two velocity components. Adaptive meshing leaves each simulation with between **29,000 and 59,000 mesh points**.

UPT runs its full hierarchy here: local information into **2,048 supernodes**, then down to **512 latent tokens**, with the decoder trained on **16,000 randomly sampled query positions**. The baselines are **GINO**, **FNO** and **U-Net**, and because FNO and U-Net need structured grids, the irregular CFD mesh has to be interpolated onto one first.

**Evaluation metrics.** Two of them. **Mean squared error** measures pointwise prediction error. **Correlation time** measures how long an autoregressive rollout stays physically meaningful. If the Pearson correlation between the predicted and ground-truth solution at timestep $$t$$ is

$$
\rho(t) = \frac{\operatorname{Cov}\big(\hat{\mathbf{u}}^t, \mathbf{u}^t\big)}{\sigma_{\hat{\mathbf{u}}^t} \, \sigma_{\mathbf{u}^t}}, \qquad
T_{\text{corr}} = \min \{\, t : \rho(t) < 0.8 \,\},
$$

then the correlation time is the first timestep where it drops below **0.8**. A model can have an excellent one-step MSE and still accumulate enough error to diverge from the true solution, which is exactly what this second metric catches.

**Results.** UPT beats every baseline at every model size tested, on both metrics. The largest UPT, at **68 million parameters**, takes roughly **450 A100 GPU-hours** for 100 epochs, comparable to GINO at the same parameter count, and the smaller models still come out ahead on **150 to 200 GPU-hours**.

The inference table is where the argument gets made:

**Table 3.** Time to roll out one full transient pipe-flow trajectory of 100 timesteps. The 400× headline compares a GPU model against a 16-CPU solver, so read the middle column for the hardware-matched picture. All timings reported by Alkin et al. [[1]](#ref-1).

| Method | 16 CPUs | 1 A100 | Speedup |
| :--- | :---: | :---: | :---: |
| pisoFoam (numerical solver) | 120 s | n/a | 1× |
| GINO-68M (autoregressive) | 48 s | 1.2 s | 100× |
| UPT-68M (autoregressive) | 46 s | 2.0 s | 60× |
| **UPT-68M (latent rollout)** | **3 s** | **0.3 s** | **400×** |

One result worth pausing on: **autoregressive UPT is slower than autoregressive GINO**, 2.0 s against 1.2 s. The paper is upfront about it. UPT's encoder is heavier, and that cost gets paid on every single encode-decode cycle.

**Latent rollout** is what turns this around. Encoding once and decoding once drops inference from **2.0 s to 0.3 s**, taking the speedup over OpenFOAM from roughly **60× to 400×** at almost the same accuracy as the autoregressive version.

**Discretization generalization.** The authors also vary the discretization at inference time. Although the model trains on 2,048 supernodes and 16,000 query positions, it holds up across different numbers of input points, supernodes and query locations without retraining. Handing it **more input points** than it saw during training even improves accuracy slightly rather than degrading it.

### 5.3 Lagrangian dynamics: a different kind of discretization entirely

**Setup.** The final experiment leaves Eulerian meshes behind for **particle-based fluid dynamics**. The authors use the **Taylor-Green Vortex (TGV)** datasets from **LagrangeBench** [[15]](#ref-15) in both two and three dimensions, generated with **Smoothed Particle Hydrodynamics (SPH)**. TGV is a standard benchmark because it has an exact analytical solution to the incompressible Navier-Stokes equations, so learned predictions can be checked against a known ground truth. The baselines are **Graph Network-based Simulators (GNS)** [[6]](#ref-6) and **Steerable E(3)-Equivariant Graph Neural Networks (SEGNN)** [[16]](#ref-16), both built specifically for particle simulations.

**Fields instead of accelerations.** This is one of the most distinctive ideas in the paper. GNS and SEGNN predict a per-particle acceleration and then integrate it numerically to update velocity and position:

$$
\hat{\mathbf{a}}^t \;\Rightarrow\; \mathbf{v}^{t + \Delta t} = \mathbf{v}^t + \hat{\mathbf{a}}^t \Delta t, \qquad
\mathbf{x}^{t + \Delta t} = \mathbf{x}^t + \mathbf{v}^{t + \Delta t} \Delta t .
$$

That comes with a catch. Because positions come out of numerical integration, $$\Delta t$$ has to stay small to remain stable. The solver got replaced by a network, but the small-timestep restriction that made the solver expensive came along with it.

UPT instead learns the underlying **velocity field** and evaluates it wherever it is needed, taking particle motion directly from the predicted velocities:

$$
\hat{\mathbf{v}}(\mathbf{y}, t') = \mathcal{D}\big( \mathcal{A}^{K}(\mathcal{E}(\cdot)), \, \mathbf{y} \big), \qquad
\mathbf{x}^{t'} = \mathbf{x}^t + \hat{\mathbf{v}} \, \Delta T, \qquad \Delta T = 10 \Delta t .
$$

Because the decoder stands for a continuous velocity field rather than a set of particle trajectories, it can be queried at **any spatial location**, whether or not a particle currently sits there. The model predicts the flow itself instead of tracking individual particles, and one model call advances the simulation by **ten timesteps**.

<figure>
  <img src="/images/lagrangian-rollout.png" alt="Comparison of acceleration-based particle integration against velocity-field prediction">
  <figcaption><b>Figure 9.</b> Comparison of two approaches for particle dynamics. Traditional graph-based simulators predict particle accelerations and recover future positions through numerical integration. UPT instead predicts the underlying velocity field and moves particles directly through that field. Figure adapted from Alkin et al. [<a href="#ref-1">1</a>].</figcaption>
</figure>

**Training.** The model sees particle positions and velocities at timesteps $$t-1$$ and $$t$$ and predicts the velocities at $$t'-1$$ and $$t'$$ under an MSE loss. Each iteration randomly samples between **50% and 100%** of the available particles, which doubles as data augmentation.

**Results.**

<figure class="half">
  <img src="/images/tgv2d-error.png" alt="Velocity error against timestep for GNS, SEGNN and UPT on TGV2D">
  <img src="/images/tgv3d-error.png" alt="Velocity error against timestep for GNS, SEGNN and UPT on TGV3D">
  <figcaption><b>Figure 10.</b> Rollout error against timestep, lower is better. <b>Left:</b> TGV2D, where UPT begins as the worst of the three at step 10 and crosses below both baselines by roughly step 20, after which all three decay towards a similar floor by step 120. <b>Right:</b> TGV3D, the harder case, where the same crossing happens by about step 18 but the gap keeps widening instead of closing, leaving UPT roughly 40% below GNS and SEGNN at step 50. Note in both panels that the baselines get <i>worse</i> before they get better, peaking around step 14 to 25. UPT has no such hump. Figure from Alkin et al. [<a href="#ref-1">1</a>].</figcaption>
</figure>

The shape of the curves in Figure 10 says more than the final numbers do. UPT starts as the worst of the three around timestep 10 and passes both baselines by roughly timestep 20, after which the gap keeps widening. The baselines get worse before they get better, peaking around timestep 14. That hump is integration error piling up faster than the Taylor-Green vortex naturally decays, and UPT has nothing equivalent because there is no integration chain for error to accumulate in.

Inference cost on TGV3D:

**Table 4.** Inference cost on TGV3D. SEGNN's equivariant message passing is expensive enough that it barely beats the numerical solver, while UPT is roughly 11× faster than GNS. All timings reported by Alkin et al. [[1]](#ref-1).

| Solver | Rollout time | Speedup vs SPH |
| :--- | :---: | :---: |
| SPH (numerical) | 4.90 s | 1× |
| SEGNN | 2.76 s | 1.8× |
| GNS | 0.56 s | 8.8× |
| **UPT** | **0.05 s** | **98×** |

Equivariant message passing is expensive enough that SEGNN buys almost nothing over just running the solver.

---

## 6. Strengths and limitations

On the strengths: keeping the latent representation fixed breaks the link between cost and mesh size, so UPT handles much larger simulations on much less memory than existing neural operators. Latent rollout drops the repeated encode-decode cycle, which costs training complexity but makes inference fast enough to be interesting for interactive use. And the same architecture covers structured grids, irregular meshes and particles with no changes and no per-discretization retraining.

The limitations are worth taking seriously too. Representing tens of thousands of mesh points in a few hundred latent tokens works on these benchmarks, but nothing here shows that the same compression survives highly turbulent or strongly discontinuous flows, and those are the cases people actually want a fast surrogate for. Constraints such as incompressibility are learned from data rather than built in, so there is no guarantee the output respects them, which rules out applications that need hard physical guarantees. The paper also reports diminishing returns past 68 million parameters, which suggests the training setup, not the architecture, is what runs out first.

---

## 7. Conclusion

At its core UPT is one idea: the cost of a neural operator should not grow just because the mesh has more points in it. Compress the simulation into a fixed-size latent state, evolve the dynamics entirely in there and reconstruct the field only where an answer is actually needed.

Three pieces make that work. A hierarchical encoder that compresses large meshes through supernodes and message passing, a query-based decoder that can be evaluated on discretizations the model never trained on, and the two inverse losses that keep the latent state stable enough for long latent rollouts. Without touching the network, the same framework covers steady-state surface flows, transient CFD and Lagrangian particle dynamics, at up to **400×** the throughput of the reference solver.

The open problems from the previous section are real, and conservation in particular seems like the thing to fix next. Even so, the paper makes a convincing case for its central claim: computational cost does not have to scale with the size of the discretization.

### Key takeaways
{:.no_toc}

- The number of latent tokens is independent of the number of mesh points, so cost stays roughly flat as simulations get larger.
- On ShapeNet-Car, **64 latent tokens** match GINO's accuracy on **33× less memory** and train about 225× faster per epoch.
- Encoding once and running the dynamics entirely in latent space is what buys the inference speedup, and it is what makes particle simulations tractable at all.
- The two inverse losses are the price of that: extra training compute in exchange for a latent state stable enough to roll out a long way.

---

## References
{:.no_toc}

<a name="ref-1"></a>[1] B. Alkin, A. Fürst, S. Schmid, L. Gruber, M. Holzleitner and J. Brandstetter. **Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators.** *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. [arXiv:2402.12365](https://arxiv.org/abs/2402.12365)

<a name="ref-2"></a>[2] H. G. Weller, G. Tabor, H. Jasak and C. Fureby. **A tensorial approach to computational continuum mechanics using object-oriented techniques.** *Computers in Physics*, 12(6):620-631, 1998.

<a name="ref-3"></a>[3] R. I. Issa. **Solution of the implicitly discretised fluid flow equations by operator-splitting.** *Journal of Computational Physics*, 62(1):40-65, 1986.

<a name="ref-4"></a>[4] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart and A. Anandkumar. **Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs.** *Journal of Machine Learning Research*, 24(89):1-97, 2023.

<a name="ref-5"></a>[5] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart and A. Anandkumar. **Fourier Neural Operator for Parametric Partial Differential Equations.** *International Conference on Learning Representations (ICLR)*, 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

<a name="ref-6"></a>[6] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec and P. Battaglia. **Learning to Simulate Complex Physics with Graph Networks.** *International Conference on Machine Learning (ICML)*, 2020.

<a name="ref-7"></a>[7] J. Brandstetter, D. Worrall and M. Welling. **Message Passing Neural PDE Solvers.** *International Conference on Learning Representations (ICLR)*, 2022.

<a name="ref-8"></a>[8] Z. Li, N. Kovachki, C. Choy, B. Li, J. Kossaifi, S. Otta, M. A. Nabian, M. Stadler, C. Hundt, K. Azizzadenesheli and A. Anandkumar. **Geometry-Informed Neural Operator for Large-Scale 3D PDEs.** *Advances in Neural Information Processing Systems (NeurIPS)*, 2023. [arXiv:2309.00583](https://arxiv.org/abs/2309.00583)

<a name="ref-9"></a>[9] A. Gholami, Z. Yao, S. Kim, C. Hooper, M. W. Mahoney and K. Keutzer. **AI and Memory Wall.** *IEEE Micro*, 44(3):33-39, 2024. [arXiv:2403.14123](https://arxiv.org/abs/2403.14123)

<a name="ref-10"></a>[10] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals and G. E. Dahl. **Neural Message Passing for Quantum Chemistry.** *International Conference on Machine Learning (ICML)*, 2017.

<a name="ref-11"></a>[11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser and I. Polosukhin. **Attention Is All You Need.** *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

<a name="ref-12"></a>[12] A. Jaegle, F. Gimeno, A. Brock, A. Zisserman, O. Vinyals and J. Carreira. **Perceiver: General Perception with Iterative Attention.** *International Conference on Machine Learning (ICML)*, 2021.

<a name="ref-13"></a>[13] W. Peebles and S. Xie. **Scalable Diffusion Models with Transformers.** *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023.

<a name="ref-14"></a>[14] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su, J. Xiao, L. Yi and F. Yu. **ShapeNet: An Information-Rich 3D Model Repository.** *CoRR*, abs/1512.03012, 2015. [arXiv:1512.03012](https://arxiv.org/abs/1512.03012)

<a name="ref-15"></a>[15] A. Toshev, G. Galletti, F. Fritz, S. Adami and N. Adams. **LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite.** *NeurIPS Datasets and Benchmarks Track*, 2023.

<a name="ref-16"></a>[16] J. Brandstetter, R. Hesselink, E. van der Pol, E. J. Bekkers and M. Welling. **Geometric and Physical Quantities Improve E(3) Equivariant Message Passing.** *International Conference on Learning Representations (ICLR)*, 2022.

