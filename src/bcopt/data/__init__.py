"""Data preparation entry points.

  prepare_fineweb_edu  — download FineWeb-Edu, tokenize with the Qwen
                         tokenizer, and pack into fixed-length sequences.
  make_noisy_packed    — build a span-replacement-corrupted variant of an
                         existing packed dataset, used for the mixed-quality
                         pretraining diagnostic in the paper.
"""
