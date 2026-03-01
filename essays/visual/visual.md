---
title: Geometry Before Abstraction
---

![Banner](banner.jpg)

#Geometry Before Abstraction 
##Multi-Scale Visual–Spatial Co-Learning

We argue that embodied intelligence is fundamentally a geometric epistemology. An agent does not first learn symbols and then attach them to the world; it first acquires a representation whose internal topology is aligned with the topology of physical interaction. From this alignment, higher-level abstractions become compressions of stable geometric relations rather than free-floating codes. This position yields a disciplined consequence: representation learning for embodied agents should preserve spatial topology across scales, treat temporal change primarily as transport of structure, and allow abstraction only through topology-preserving coarse-graining. We formalize a minimal set of physical assumptions under which geometry-aligned learning is well-posed and illustrate constructively how approximate metric cognition, such as estimating “ten steps,” can emerge from geometry-preserving visual fields via calibration to reference units, without requiring explicit symbolic three-dimensional reconstruction. Our claim is not that geometry is sufficient for all cognition, but that it is a necessary substrate. Abstraction that precedes geometric grounding tends to be brittle, data-hungry, and unstable for control.


## Position: Embodied Learning as Geometric Epistemology

The dominant temptation in modern machine learning is to treat representation as an unconstrained coding problem. Any latent that supports prediction is considered acceptable. In embodied settings, this temptation is costly, because the world is not an arbitrary data distribution. It is a physical system governed by invariants: continuity in space and time, locality of interaction, monotonic projection cues under approach, and structured coarse-to-fine organization in both perception and action.

These invariants are not statistical accidents. They are the structural preconditions that make control possible. A representation that ignores them can still interpolate within a dataset, but it cannot reliably generalize across physically plausible perturbations. It cannot become a stable model of interaction.

Our position is therefore both philosophical and technical. Philosophically, an embodied agent “knows” space when its internal representation admits the same adjacency and ordering relations that the world enforces through contact and motion. Technically, this is best understood not as explicit coordinate reconstruction but as topology alignment. The geometry of representation should be a structured deformation of the geometry of the world. When this alignment holds, learning becomes alignment with existing structure rather than rediscovery of structure, and abstraction becomes compression rather than invention.

This is not an argument against abstraction. It is an argument about order. Geometry must precede symmetry, and symmetry must precede abstraction. Abstraction introduced before geometric grounding tends to entangle spatial relations into global codes, increasing sample complexity and destabilizing control. Abstraction introduced after geometry has stabilized can be powerful precisely because it compresses a representation that is already physically meaningful.

## Why Visual–Spatial Co-Learning Is the Correct Framing

It is common to divide perception into separate subproblems: visual features for appearance and spatial features for location and motion. In embodied systems, this division is misleading. Appearance is not separable from space because motion transports appearance, identity is defined by persistence over time, and objecthood emerges from coherent regions whose boundaries are visually determined but whose relations are spatial. Conversely, spatial reasoning cannot proceed without appearance, because regions and correspondences must be anchored in discriminable structure.

Visual learning and spatial learning therefore form a coupled constraint system acting on a single representation. The learner must discover channels that are appearance-sensitive and at the same time stable under transport. It must discover spatial structure through appearance-defined regions embedded in a geometry-preserving lattice. This coupling is multi-scale because the world is multi-scale. Fine control requires high-frequency geometric precision, while robust identity and layout require coarser invariances. The central principle is that invariance is not permission to destroy geometry. It must arise through topology-preserving coarse-graining.

## Minimal Physical Assumptions

To state the position rigorously, we begin with minimal assumptions that make geometry learnable rather than arbitrary.

Assume there exists a low-dimensional task-relevant physical state s_t \in \mathcal{M}, and observations satisfy

I_t = \Pi(s_t) + \eta_t,

where \Pi is a projection map and \eta_t is bounded noise. Furthermore, sufficiently nearby states induce images related by bounded, locally coherent deformations. Small changes in the world do not produce combinatorially unrelated images.

Assume further that physical dynamics are piecewise smooth, so that

s_{t+1} = g(s_t, a_t) + \epsilon_t,

where g is locally Lipschitz away from event boundaries and \epsilon_t is bounded. Most temporal variation is therefore well modeled as transport of structure.

These assumptions justify the central representational demand: continuity and transport in the world should be mirrored as continuity and transport in representation. Without these assumptions, there is no principled reason to prefer geometry-preserving structure over arbitrary global codes.

## Formal Core: Geometry-Preserving Fields

Representation is modeled as a hierarchy of retinotopic feature fields

F_t^{(\ell)} \in \mathbb{R}^{H_\ell \times W_\ell \times C_\ell},

where lattice indices correspond to localized image regions. Each channel is a spatial field. Spatial relations are encoded not as symbolic coordinates but as the geometry of activation patterns across the lattice.

Temporal evolution is modeled through a transport operator that enforces locality-biased correspondence across time. Multi-step rollout consistency requires that transported estimates remain close to actual features for horizons greater than one. When this condition holds across diverse trajectories, appearance cannot be encoded in arbitrary frame-specific codes. It must be encoded in a form that remains stable under local transport. Thus transport constraints couple visual and spatial learning.

Across scales, abstraction is enforced through local aggregation:

F_t^{(\ell+1)} \approx A^{(\ell)}(F_t^{(\ell)}).

Cross-scale alignment constrains higher-level invariances to remain consistent with local aggregation of fine-scale geometry. Abstraction becomes disciplined coarse-graining rather than unconstrained global mixing.

The result is a representation whose topology mirrors the topology of physical interaction, while still permitting increasing invariance across scales.

## Constructive Derivation: From Fields to Metric Cognition

A philosophical position must demonstrate constructive power. We therefore illustrate how approximate metric spatial judgments can arise without positing explicit symbolic three-dimensional reconstruction.

Consider two spheres of known radius R. Regionization over retinotopic fields yields centroids (u_1, v_1) and (u_2, v_2), as well as apparent radii r_1, r_2, derived from region extents.

Under a pinhole camera model with focal length f,

r \approx f \frac{R}{Z},

which implies

Z \approx \frac{fR}{r}.

Depth proxies therefore emerge from apparent size, grounded in appearance and geometry.

Angular separation satisfies

\theta \approx \frac{\sqrt{(u_2 - u_1)^2 + (v_2 - v_1)^2}}{f}.

An approximate center distance follows:

\widehat{D}^2 \approx \widehat{Z}_1^2 + \widehat{Z}_2^2 - 2 \widehat{Z}_1 \widehat{Z}_2 \cos\theta.

The point is not analytic precision but constructibility. Once geometry-preserving fields yield region statistics, approximate metric quantities are derivable when combined with a reference unit.

Metric cognition then emerges through calibration. Let \rho be a reference unit, such as step length or calibrated impulse increment. The agent learns a mapping

\widetilde{D} = \mathcal{C}(\|\Delta\|, r_1, r_2; \theta_C),

where \Delta is centroid displacement and \mathcal{C} is learned through interaction. Symbolic reports arise through quantization of calibrated quantities. In this sense, symbolic spatial reasoning is a compression of calibrated geometric relations, not an independent perceptual substrate.


## Falsifiability

A serious position must make predictions. The geometry-first view predicts that strong embodied systems will exhibit scales at which low-complexity readouts recover spatial variables, that rollout error will degrade gradually with horizon under physically plausible perturbations, and that coarse-scale invariances will remain traceable to local aggregation of fine-scale fields. Systems that collapse spatial structure early may achieve dataset-level accuracy yet fail under perturbations that preserve physics, because their internal geometry is not aligned with the world’s geometry.


## Outlook

Geometry is not the ceiling of intelligence. It is the substrate. Memory, object-centric latents, predictive world models, and planning become stable only when operating on geometry-preserving representations. Without such stabilization, long-horizon reasoning amplifies error and decision-making becomes brittle. The order remains invariant: geometry first, symmetry next, abstraction last, and planning on top of stabilized geometry.

