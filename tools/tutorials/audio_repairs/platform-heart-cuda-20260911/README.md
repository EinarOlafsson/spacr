# Platform Installers: first CUDA, English Heart

Published and reverified at media-host commit
`003f4c8e88e6050dab5aad7750fb2f990974b636`. Only the two Heart files changed.
`hosted-browser-check.json` verifies the final served hash and the actual
requested seek positions, not merely whichever caption happens to be active.

Only the first CUDA sentence of the **hosted** Heart track was synthesized
again, using the same pinned Kokoro model and voice with `/kˈudə/` instead
of `/kˈuːdᵊ/`: a full final schwa instead of the reduced vowel. No other
voice, language, pronunciation rule or shared video was changed.

The replacement occupies scene 5, sentence 2, 54.375–61.880 seconds. A
0.996669 tempo ratio fits the existing window. All decoded PCM samples
outside that window were preserved before AAC re-encoding, including the
second CUDA mention. AAC re-encoding changes compressed bytes throughout;
this is not a claim of identical encoded audio outside the edit. Caption
text, sentence boundaries and total duration remain unchanged.

The older authoring-directory track has a different script and was not used.
The source pair is backed up privately and recoverable through media-host
history. The receipt records its exact revision and hashes.

Verification: finite samples and equal decoded lengths; intended synthesis
phonemes; matching audio/metadata hashes; actual browser playback and native
captions at 27, 37, 55 and 76 seconds; three native-track reloads; 70 focused
tests. FFmpeg measured -2.51 dBFS sample peak, **-1.65 dBTP** true peak,
**-18.17 LUFS** and **2.0 LU LRA**. No whole-track loudness filter was applied.
These are technical checks, not a claimed human listening review.

The repair script runs with two CPU threads under the 100-GiB total-RAM
guard. Its explicit `--publish` uploads only the two Heart files in one
media-host commit, guarded by current source hashes and parent revision.
It never invokes the bulk publisher. See `repair-receipt.json` for actual
publication state.

The separate caption-player fix still requires a successful Pages deployment.
Publishing this audio does not deploy the website or the private refreshed
tutorial collection.
