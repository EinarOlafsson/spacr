# Issue 147: source triage of the missing tool-approval interface

The open issue reports macOS spaCR 1.5.1.4 showing an approval requirement
without a prompt. The inspected chat path starts plain CLI providers with a
line-based output pipe and inherited stdin; it exposes no structured approval
request/response channel or chat pseudo-terminal. Login uses a separate path.
Worker/UI events cannot present an approval request, and the streaming stage
precedes provider output. Default prompts do not describe this GUI limitation.

The exact permission-rule reminder filter is source-supported, but the reported
phrase does not match its prefixes. It is not established as the cause. A model
claim that approval is needed is not proof of a real vendor approval event.
Selected function ASTs match the released reference; whole files and effective
provider configuration are not asserted identical. No native reproduction or
provider version/configuration was measured.

Six source findings and eight future acceptance families distinguish an
explicit manual fallback from a reviewed protocol bridge. No proposed fix
bypasses approval. Cancellation, process cleanup, privacy and existing CLI
authentication must be preserved. Root rechecked 11 source/test bindings.
No provider, authentication, tools, tests, GUI, issue comment or app edit ran.
Home ownership and the release freeze still gate implementation.
