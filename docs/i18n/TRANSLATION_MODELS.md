# Translation model attribution

spaCR's English source text is authoritative. The external locale catalogs
were drafted with permissively licensed OPUS-MT and M2M100 models, then checked by spaCR's
placeholder, source-freshness, terminology, and contextual-review tooling.
They are not copied into Python function docstrings.

| spaCR locale | Model | Upstream license |
|---|---|---|
| Swedish | [`Helsinki-NLP/opus-mt-en-sv`](https://huggingface.co/Helsinki-NLP/opus-mt-en-sv) | Apache-2.0 |
| German | [`Helsinki-NLP/opus-mt-en-de`](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) | CC-BY-4.0 |
| Spanish | [`Helsinki-NLP/opus-mt-en-es`](https://huggingface.co/Helsinki-NLP/opus-mt-en-es) | Apache-2.0 |
| Simplified Chinese | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M), normalized with [OpenCC `t2s`](https://github.com/BYVoid/OpenCC) | MIT; Apache-2.0 |
| Portuguese | [`Helsinki-NLP/opus-mt-tc-big-en-pt`](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt) | CC-BY-4.0 |
| Hindi | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) | MIT |
| Korean | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) | MIT |
| Icelandic | [`facebook/m2m100_418M`](https://huggingface.co/facebook/m2m100_418M) | MIT |
| French | [`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) | Apache-2.0 |

CC-BY-4.0 model attribution is retained both here and in the generated locale
metadata.

Simplified-Chinese model output is normalized with OpenCC's conservative
Traditional-to-Simplified (`t2s`) configuration after protected code, RST,
URLs, identifiers, and product names have been isolated. The catalog metadata
records this normalizer separately from the M2M100 generator. Generation and
audit require OpenCC 1.1 plus its `t2s.json` data and verify that a second pass
is a no-op; they never use the locale-expanding `tw2sp` configuration.

The research-only NLLB checkpoint used by the separate tutorial-production
workspace is deliberately not used for the shipped application, installer,
README, or API catalogs.
