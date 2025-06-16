# Mixture-of-Experts (MoE) Architecture

The Mixture-of-Experts (MoE) architecture is a powerful approach for scaling large language models 
by dividing the model into multiple specialized subnetworks, known as experts.

In a typical MoE layer, a router or gating network determines, for each input token, 
which subset of experts is most suitable to process the information. 

For example, in DeepSeek-R1, out of 256 available experts per layer, 
only the top-8 experts with the highest relevance scores are activated for each token. 

This dynamic selection ensures that only a small fraction of the experts are active at any time, 
significantly reducing computational costs while maintaining high model capacity.

The outputs of the selected experts are then weighted and aggregated before being passed to the next layer. 
This mechanism allows the model to flexibly adapt to different tasks and contexts, 
as the selection of experts can vary for each token and input.

Schematic  MoE architecture and expert selection :


+-------------------+
|   Input Embedding |
+-------------------+
          |
+-------------------+
| Transformer Block |
+-------------------+
          |
+--------------------------+
|      MoE Layer           |
|  +--------------------+  |
|  |  Router (Gating)   |  |
|  +--------------------+  | 
|      /     |     \       |
|   +-----+ +-----+ +-----+|
|   |Exp 1| |Exp 2| |Exp N||
|   +-----+ +-----+ +-----+|
|      \     |     /       |
|   (Only Top-8 active)    |
+--------------------------+
          |
+-------------------+
|   Output Head(s)  |
+-------------------+
          |
   Model Output     |
 (e.g. Reasoning,   |
  Language, Refusal)|
          |
-----------------------------------------------
  ^         ^         ^                ^
  |         |         |                |
  |         |         |                |
  |         |         |                +---> Expert N: e.g., fine-tuned for language switching (e.g., Chinese)
  |         |         +--------------------> Expert 3: fine-tuned for domain knowledge
  |         +------------------------------> Expert 2: deactivated to reduce refusal responses
  +----------------------------------------> Expert 1: enhanced for specific reasoning behavior


────────────────────────────────────────────────────────────
|                  MoE Layer (DeepSeek)                    |
────────────────────────────────────────────────────────────
|                                                          |
|   Token Representation (u_t)                             |
|           |                                              |
|           v                                              |
|   +-------------------+                                  |
|   |   Router/Gating   |                                  |
|   +-------------------+                                  |
|           |                                              |
|           v                                              |
|   Score calculation for all experts (1...256)            |
|           |                                              |
|           v                                              |
|   Select top-8 experts with highest scores               |
|           |                                              |
|           v                                              |
|   +-----------------------------------------------+      |
|   |   Activated experts:                          |      |
|   |   [Exp 3, Exp 7, Exp 14, Exp 21, ...]         |      |
|   +-----------------------------------------------+      |
|           |                                              |
|           v                                              |
|   Check for negatively classified experts                |
|   (e.g., Exp 14 = "Refusal Expert")                      |
|           |                                              |
|           v                                              |
|   If negative: set this expert's contribution to 0       |
|   (no replacement, reweight the others)                  |
|           |                                              |
|           v                                              |
|   Weighted aggregation of remaining experts              |
|           |                                              |
|           v                                              |
|   Pass to next layer / output                            |
────────────────────────────────────────────────────────────
