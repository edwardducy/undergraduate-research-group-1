#import "figure-style.typ": *
#show: figure-style

#ipo-diagram(
  height: 3.3in,
  widths: (1.6in, 2.5in, 1.7in),
  [
    - Taglish posts on X about tropical cyclones, January to November 2026
    - HumAID categories and HUMAID-NER entity types
    - Pretrained transformer encoder \
      (SEA-LION-ModernBERT-300M)
    - 13 multi-task optimizers
    - One laptop for training
  ],
  [
    + *Design and build the dataset and the multi-task model* \
      Collect, clean, annotate, and split the posts, and share one encoder between both tasks (RQ2)
    + *Train single-task models* \
      Measure baseline F1-scores (RQ1)
    + *Benchmark the 13 multi-task optimizers* \
      Compare negative transfer, training time, and peak memory, and select the optimizer based on negative transfer (RQ3)
    + *Test the final multi-task model* \
      Compare it with the single-task models in F1-score, training time, and peak memory (RQ4)
  ],
  [
    - A dataset of Taglish typhoon posts annotated for both tasks
    - A multi-task model for humanitarian classification and NER, trained with the selected optimizer
    - A comparison of the 13 multi-task optimizers in negative transfer, training time, and peak memory
  ],
)
