Source-only pending-visual-seek followup for lesson04.
Apply this exact third patch after K4 b929b7dc53465132322c3ca0b866f567de773b9f and K5 253f5814bc2add67d01032948271b8aa1b06aa00.
K6 2427d57f17ad78d97cf81f18720c1cd9e0c7cc02 is a direct child of K5 and changes only playerJS + its focused test.
Native capture, rendered media, timing and retained narration remain K4-derived.
The preserve-only guard pauses narration and retains the explicit requested audio offset
until the visual seek settles, preventing automatic scene-start rewinds.
Actual owning final cohort:226PASS75.25seconds/terminal0; previous224 and225 cohorts
are historical evidence and must not be summed. Independent source peer PASS included.
Actual K6 production73-case matrix/publication remain root-owned and pending.
No new capture, reencode, synthesis, tests, browser or publication performed by this packet.
Original two closed source handoffs remain untouched.
