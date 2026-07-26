export const meta = {
  name: 'econirl-testing-loop',
  description: 'econirl self-study loop: triage Joe clean-room findings real-vs-false against HEAD source + papers + held-out truth, research the hard cases against the literature (with a download-and-read proof gate), then synthesize a surgical workplan for the one human gate.',
  phases: [
    { title: 'Load', detail: "read the round's findings.json" },
    { title: 'Validate', detail: 'one refute-by-default skeptic per finding' },
    { title: 'Research', detail: 'hard cases: search, source, read, and PROVE the read before resolving' },
    { title: 'Synthesize', detail: 'write the validated ledger + the workplan for the human gate' },
  ],
}

// ---- inputs (args optional; defaults target the nfxp crank) -------------------
const A = typeof args === 'string' ? (args.trim() ? JSON.parse(args) : {}) : (args || {})
const estimator = A.estimator || 'nfxp'
const round = A.round || 1
const repo = A.repo || '/Users/pranjal/Code/econirl'
const rr = String(round).padStart(2, '0')
const outDir = A.outDir || `${repo}/acceptance/loop/${estimator}/round-${rr}`
const truthPath = A.truthPath || `${repo}/acceptance/problems/${estimator}/truth.json`
const findingsPath = A.findingsPath || `${repo}/acceptance/reports/${estimator}_findings.json`

// ---- contracts ----------------------------------------------------------------
const FINDINGS_SCHEMA = {
  type: 'object',
  additionalProperties: true,
  required: ['findings'],
  properties: {
    findings: { type: 'array', items: { type: 'object', additionalProperties: true } },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['finding_ref', 'verdict', 'severity', 'reproduced', 'evidence', 'root_cause', 'proposed_fix', 'needs_research', 'research_question', 'confidence'],
  properties: {
    finding_ref: { type: 'string', description: "the finding's `where` field, verbatim" },
    verdict: { type: 'string', enum: ['real_bug', 'real_ux', 'already_fixed', 'false'] },
    severity: { type: 'string', enum: ['high', 'med', 'low'] },
    reproduced: { type: 'boolean', description: 'does the static HEAD evidence confirm the claim still holds' },
    evidence: { type: 'string', description: 'HEAD file:line citations that confirm or refute, with a one-line reading of each' },
    root_cause: { type: 'string' },
    proposed_fix: { type: 'string', description: 'the smallest surgical change; empty if false or already_fixed' },
    needs_research: { type: 'boolean', description: 'true if resolving this needs a domain or literature judgment (the correct formula, an identification assumption, a modeling default, what a paper proves), not a mechanical fix' },
    research_question: { type: 'string', description: 'the precise question to answer from the literature; empty when needs_research is false' },
    confidence: { type: 'number', description: '0..1' },
  },
}

// The research result must NAME an on-disk artifact and QUOTE from it. Those two
// fields are grep-verified by a fresh agent before "resolved" is believed.
const RESEARCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['finding_ref', 'resolved', 'answer', 'citations', 'artifact_path', 'supporting_quote', 'recommended_fix', 'defer_reason', 'confidence'],
  properties: {
    finding_ref: { type: 'string' },
    resolved: { type: 'boolean', description: 'true ONLY if you downloaded and read the source and can quote it; false if ambiguous, a maintainer judgment, or you could not get the document' },
    answer: { type: 'string', description: 'what the literature says, in two or three sentences' },
    citations: { type: 'array', items: { type: 'string' }, description: 'paper + section/page; never fabricate' },
    artifact_path: { type: 'string', description: 'absolute path to the on-disk .md/.tex/.pdf you actually read; required when resolved is true' },
    supporting_quote: { type: 'string', description: 'a verbatim span (12+ words) copied from artifact_path that supports the answer; it WILL be grep-checked against the file' },
    recommended_fix: { type: 'string', description: 'the fix the literature implies; empty if unresolved' },
    defer_reason: { type: 'string', description: 'why this goes to the human; empty if resolved' },
    confidence: { type: 'number' },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['verified', 'detail'],
  properties: {
    verified: { type: 'boolean', description: 'true only if the artifact exists, is a whole document, and the quote greps out of it' },
    detail: { type: 'string', description: 'what passed or failed, with the command output' },
  },
}

function shortWhere(s) {
  return String(s || 'finding').replace(/\s+/g, ' ').slice(0, 30)
}

function validatePrompt(f) {
  return [
    'You are an adversarial validator on the econirl self-study loop. A clean-room tester ("Joe") installed the PUBLISHED package econirl==0.0.10 from PyPI as a blind stranger and logged the finding below. Decide, against the CURRENT repo source (HEAD), whether it is REAL.',
    '',
    'Default to REFUTING. Confirm only if the HEAD source forces it. Joe may be wrong, may have misused the API, or may have hit a bug that HEAD already fixes.',
    '',
    'THE FINDING (Joe tested 0.0.10):',
    `  kind: ${f.kind}   severity: ${f.severity}`,
    `  where: ${f.where}`,
    `  what: ${f.what}`,
    `  repro: ${f.repro}`,
    '',
    `READ in ${repo} (cheap reads + reasoning ONLY -- do NOT run pytest, do NOT fit an estimator):`,
    '  - src/econirl/estimators/nfxp.py        (the sklearn wrapper Joe used)',
    '  - src/econirl/estimation/nfxp.py        (the lower-level estimator + optimizer)',
    '  - docs/research/internal_docs/estimators/nfxp/index.md  (maintainer ground truth)',
    `  - the held-out truth at ${truthPath} if it exists`,
    '',
    'CLASSIFY into exactly one verdict: real_bug, real_ux, already_fixed (cite the fixing file:line), or false.',
    '',
    'THEN flag the hard cases. Set needs_research = true when resolving the finding needs a domain or literature judgment (the correct standard-error formula, an identification assumption, what a paper proves), not a mechanical change. When true, write the precise research_question. When false, leave it empty.',
    '',
    'Cite HEAD file:line in `evidence`. Mechanical real items get a surgical `proposed_fix`; needs_research items leave it to the research phase.',
  ].join('\n')
}

function researchPrompt(v) {
  return [
    'You research ONE hard finding on the econirl self-study loop. Resolve it from the literature, or defer it. The bar is high on purpose: a downstream agent will grep your quote out of the file you name, so you cannot fake having read it.',
    '',
    `FINDING: ${v.finding_ref}`,
    `QUESTION: ${v.research_question}`,
    `VALIDATOR EVIDENCE: ${v.evidence}`,
    '',
    'STEPS (follow /source and /read):',
    `1. Check the repo first: ${repo}/docs/research/papers/ and ${repo}/docs/research/internal_docs/estimators/${estimator}/. If the source is already there, use it.`,
    '2. If not, find the authoritative source, DOWNLOAD the actual document (arxiv e-print, or the PDF), and convert it to markdown with docling (one docling at a time, never more than two). Save it under docs/research/papers/. A search snippet or an abstract is NOT a source.',
    '3. Read the segments that bear on the question, in the converted text.',
    '4. To set resolved = true you MUST fill artifact_path (the absolute path to the file you read) and supporting_quote (a verbatim span of 12 or more words copied from that file that backs your answer). The quote will be grep-checked against the file. If you did not download and read a real document, you CANNOT set resolved = true.',
    '5. If the document is paywalled or you could not get it, or the literature leaves it to a maintainer judgment, set resolved = false with a clear defer_reason. Never answer from memory, an abstract, or a search snippet.',
    '',
    'A clear textbook result (for example a standard-error identity) with a real quote counts as resolved. Anything you could not actually read defers.',
  ].join('\n')
}

function verifyPrompt(r) {
  return [
    'You verify that a research agent actually downloaded and read its source. Run these and report, do not reason around them.',
    '',
    `ARTIFACT: ${r.artifact_path}`,
    `QUOTE: ${r.supporting_quote}`,
    '',
    'Run, in order:',
    `1. test -f "${r.artifact_path}" && echo EXISTS || echo MISSING`,
    `2. ~/.claude/skills/read/scripts/check-fulltext.sh "${r.artifact_path}"   (it must be a whole document, not a stub)`,
    `3. grep -F -- "${r.supporting_quote}" "${r.artifact_path}" >/dev/null && echo QUOTE_FOUND || echo QUOTE_MISSING`,
    '',
    'Set verified = true ONLY if the file exists, check-fulltext does not say INCOMPLETE, and the quote is QUOTE_FOUND. Otherwise verified = false. Put the actual command output in detail. Do not run anything else, do not fetch anything, do not be generous.',
  ].join('\n')
}

function synthPrompt(verdicts) {
  return [
    `You are the synthesizer on the econirl self-study loop, crank ${round}, estimator ${estimator}. Below are the per-finding verdicts as JSON; some carry a nested research result with a verified flag. Write TWO files.`,
    '',
    `Run: mkdir -p ${outDir}`,
    '',
    `File 1 -> ${outDir}/validated.md (the full ledger): a one-line count, a summary table (finding, verdict, severity, confidence, researched?), then one short section per finding with HEAD file:line evidence, root cause, proposed fix, and the research answer + citation when present.`,
    '',
    `File 2 -> ${outDir}/workplan.md (the HUMAN GATE), four sections:`,
    '  (a) "Auto-resolved from the literature (ratify)" -- real items where research.resolved is true AND research.verified is true. Give the title, file:line, the answer with its citation and the verified quote, the recommended fix, and "- [ ] RATIFY   - [ ] OVERRIDE".',
    '  (b) "Defer to human (judgment)" -- needs_research items that are unresolved OR whose read could not be verified. Give the title, the open question, the defer_reason (note explicitly when it deferred because the source could not be downloaded/verified), and "- [ ] decide: ____".',
    '  (c) "Mechanical fixes (approve/cut)" -- real items with needs_research false. "- [ ] APPROVE   - [ ] CUT", title, file:line, surgical fix, one-line acceptance criterion.',
    '  Then "Already fixed in HEAD (needs a release)" and a one-line "Cleared (false)" list.',
    '  End with: "Edit this file, then the apply phase consumes the RATIFY + APPROVE items and skips the rest."',
    '',
    'Tight and scannable. No em-dashes, no colon/semicolon splices. Then return a 3-line summary: counts, auto-resolved vs deferred (and how many deferred for an unverifiable read), and the highest-priority item.',
    '',
    'VERDICTS:',
    JSON.stringify(verdicts, null, 2),
  ].join('\n')
}

// ---- run ----------------------------------------------------------------------
phase('Load')
const loaded = await agent(
  `Read the JSON file ${findingsPath} and return the array under its top-level "findings" key, verbatim, as {"findings": [...]}. Each item has kind, severity, where, what, repro. Do not summarize, reword, drop, or reorder any item.`,
  { label: 'load-findings', phase: 'Load', schema: FINDINGS_SCHEMA }
)
const findings = (loaded && loaded.findings) || []
log(`crank ${round} on ${estimator}: ${findings.length} findings to triage`)

phase('Validate')
const verdicts = await parallel(
  findings.map((f) => () =>
    agent(validatePrompt(f), { label: `validate:${shortWhere(f.where)}`, phase: 'Validate', schema: VERDICT_SCHEMA })
  )
)
const clean = verdicts.filter(Boolean)
log(`validated ${clean.length}/${findings.length}`)

// Research the hard cases, sequential (keeps docling at most one at a time).
// Every "resolved" is then PROVED by a fresh verifier that greps the quote out
// of the named artifact. A lazy non-read produces no artifact, fails the grep,
// and is forced to defer to the human rather than become a fake answer.
phase('Research')
const hard = clean.filter((v) => v.needs_research && v.verdict !== 'false')
log(`${hard.length} hard case(s) to research`)
const byRef = {}
for (const v of hard) {
  const r = await agent(researchPrompt(v), {
    label: `research:${shortWhere(v.finding_ref)}`, phase: 'Research', schema: RESEARCH_SCHEMA,
  })
  if (r && r.resolved) {
    const check = await agent(verifyPrompt(r), {
      label: `verify-read:${shortWhere(v.finding_ref)}`, phase: 'Research', schema: VERIFY_SCHEMA, effort: 'low',
    })
    r.verified = !!(check && check.verified)
    if (!r.verified) {
      r.resolved = false
      r.defer_reason = `Auto-deferred: the cited source could not be verified as downloaded and read. ${check ? check.detail : 'verifier failed'}`
    }
  } else if (r) {
    r.verified = false
  }
  if (r) byRef[r.finding_ref] = r
}
const enriched = clean.map((v) => (v.needs_research ? { ...v, research: byRef[v.finding_ref] || null } : v))

phase('Synthesize')
const summary = await agent(synthPrompt(enriched), { label: 'synthesize', phase: 'Synthesize' })

return { verdicts: enriched, summary }
