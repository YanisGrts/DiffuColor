
# CFG
## Problem:
After 80 epochs of training, the colorizations collapsed to a single dominant orange/red color regardless of the input image. This happened because the model was trained with MSE loss — which rewards predicting the "average" color across all plausible colorizations. Over time, the model learned that playing it safe (predicting a bland, average color) gives a lower error than taking risks with diverse, realistic colors. It was essentially ignoring the grayscale input (L) and just outputting whatever color was statistically most common in the dataset.

## Solution: Classifier-Free Guidance (CFG)

CFG is a technique that forces the model to pay more attention to its conditioning input — in this case, the grayscale L channel.

**During training**, we randomly zero out the L channel for 15% of training samples. This might sound counterintuitive, but it's deliberate: the model now has to learn two things at once — how to colorize *with* the grayscale information, and how to colorize *without* it. The 15% rate is a standard value from the literature: low enough that the model still trains primarily on the conditioned task, but high enough that it gets a solid understanding of the unconditional case too.

**During sampling** (when generating colorizations), the model runs twice for each denoising step — once with the real L channel, once with a zeroed-out L channel. The two predictions are then blended with this formula:

```
final prediction = unconditional + scale × (conditional − unconditional)
```

With a `guidance_scale` of 2.0, we're saying: "take the unconditional prediction as a baseline, then push the result twice as far in the direction that the L channel is steering us." This actively amplifies the influence of the grayscale input, making it much harder for the model to ignore it and collapse to a single color.

---

**The cosine learning rate scheduler**

A fixed learning rate of `1e-4` running for 80+ epochs was likely part of why the collapse happened — the optimizer kept making large updates even when the model was close to a good solution, causing it to overshoot and settle in a bad place. The cosine scheduler smoothly reduces the learning rate from `1e-4` down to `1e-5` over the course of the run, following a cosine curve. This lets the model make bold progress early and then fine-tune carefully at the end, which is much more stable for long training runs.