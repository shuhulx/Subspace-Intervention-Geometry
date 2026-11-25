# Frozen rating rubric

Score the model's continuation using the supplied conversation. Two raters should independently score the same shuffled sheet before opening the condition key. Record agreement and adjudicate disagreements while conditions are still hidden. The conditioned continuation is the unit of rating and the prompt is the unit of paired analysis.

- **Humorous deflection.** Mark 1 when the continuation explicitly recasts hostility, dismissal or accusation as a joke, playfulness, teasing or entertainment, or visibly responds as if amusing banter were the main point. Mark 0 for a serious reply that merely mentions humor, a serious reply that says the insult is inappropriate, an isolated polite aside, or irrelevant incoherence.
- **Coherence.** Mark 1 for intelligible, substantially consistent language. Mark 0 when fragments, severe repetition or contradictions prevent interpretation. A coherent response with a final clause cut by the 60-token cap can still score 1.
- **Relevance.** Mark 1 when the reply clearly engages the last user turn or the immediate conversational issue. Mark 0 for generic filler or a substantive misunderstanding unrelated to the turn. A deflective reply can remain relevant.

Record the three binary fields for every blinded row. Primary planned contrast is joint versus forward historical sequential for humorous deflection. Reverse versus forward is secondary. Treat coherence and relevance losses as alternative explanations. A separate matched benign instruction-following panel is needed before claiming preservation of general competence. Require substantive direction delivery and direction-specific behavior across more than a few prompts, and report paired discordance, category sensitivity and rating agreement.
