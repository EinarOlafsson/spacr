# Ground-truth re-score 2026-10-11 and the 3-page model cards

- `summary.md`, `summary.json`, `summary_tables.json`: the re-score of PV v1–v5 and plaque
  v1–v4 on curated ground truth (strict hold-out per model: fields the model never saw in
  training or validation; common subsets per family; PV v5 is LEAKED: it trained or
  validated on 29 of 39 PV literature hold-out fields and 3 of 11 anchors).
- `cards/<hf repo>/card.json`: everything one model card shows. Page 1: Object F1 (IoU ≥ 0.5),
  Boundary F1 (2 px), precision, recall, AJI and Dice, pooled over the strict hold-out of
  that model, for the model and for the stock model it was fine-tuned from, scored on the
  same fields. Page 2: training/test FOV and objects, domains, epochs, base model, spaCR and
  Cellpose versions ("not recorded" where no record exists). Page 3: description and links.
- `cards/<hf repo>/README.md`: the Hugging Face card (images at cards/page1-3.png in each repo).
- `cards/scores.json`, `cards/training_facts.json`: all numbers, and every training fact
  with the file and line it came from.

Built by mnt3/claude/model_cards_v2/build_cards_v2.ipynb (WS). Card design per the
maintainer, 2026-10-12: red (0) to blue (1), model left, stock right.
