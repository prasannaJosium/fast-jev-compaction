/**
 * Pulls the high-value tokens out of a tool result that is about to be
 * dropped. A dropped result keeps only a short head, and for a listing, a
 * test run or a build log the head is preamble: the paths, identifiers and
 * failures live further down. Scanning the whole text for those tokens keeps
 * the parts that cannot be reconstructed from memory.
 */

/**
 * Ordered alternation, scanned once so results come back in the order they
 * appear. Two orderings matter and are load-bearing:
 *
 * - An assignment comes first and consumes `NAME=VALUE` whole. Only the name
 *   is ever kept, and because the match swallows the value, no later
 *   alternative can pick the value up. Salvage must never turn a leaked
 *   secret in a tool result into a secret in the retained context.
 * - A URL comes before the path pattern, so the slashes inside it are not
 *   mistaken for a path.
 *
 * A SHA must carry a digit: without that, ordinary words spelled from a-f
 * ("defaced") read as hashes.
 */
const SCAN =
  /\b([A-Za-z_][A-Za-z0-9_]{2,})=\S+|https?:\/\/[^\s)'"<>\]]+|[A-Za-z]:\\[^\s:*?"<>|]+|(?:\.{0,2}\/)?(?:[\w.@-]+\/)+[\w.@-]+|\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b/g;

/**
 * `fail` needs its own word-bounded alternative: test runners emit a bare
 * "FAIL", which neither "failed" nor "failure" matches.
 */
const ERROR_LINE =
  /error|exception|\bfail(ed|ure|ing|s)?\b|fatal|traceback|panic|refused/i;
const EXIT_CODE = /\bexit (?:code|status)\s+(\d+)/i;

/** The same assignment shape `SCAN` consumes, for redacting kept lines. */
const ASSIGNMENT = /\b([A-Za-z_][A-Za-z0-9_]{2,})=\S+/g;

/**
 * Strips assigned values out of a line that is kept whole. A failing line is
 * often the one carrying a credential ("auth failed for TOKEN=..."), and the
 * name is the useful part; the value must not survive into the context.
 */
function redact(line: string): string {
  return line.replace(ASSIGNMENT, '$1=<redacted>');
}

/**
 * Every identifier in `text`, in the order it first appears, without repeats.
 * For an assignment only the name is returned, never the value.
 */
export function salvageIdentifiers(text: string): string[] {
  const seen = new Set<string>();
  for (const match of text.matchAll(SCAN)) {
    seen.add(match[1] ?? match[0]);
  }
  return [...seen];
}

/**
 * The lines of `text` that report a failure, in order, without repeats. An
 * exit line counts only when the status is non-zero.
 */
export function salvageErrorLines(text: string): string[] {
  const seen = new Set<string>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length === 0) continue;
    const exit = EXIT_CODE.exec(line);
    if (exit) {
      if (exit[1] !== '0') seen.add(redact(line));
      continue;
    }
    if (ERROR_LINE.test(line)) seen.add(redact(line));
  }
  return [...seen];
}

export interface PeekOptions {
  /** Characters sampled from the start of the result. */
  headChars: number;
  /** Characters sampled from the end of the result. */
  tailChars: number;
}

/**
 * A bounded sample of a tool result, for the question that asks whether the
 * result is worth keeping verbatim. Without it that question is answered from
 * the tool name and a byte count alone.
 *
 * A failed result is sampled tail-first: the reason a thing failed is printed
 * last, while the head is the command echo.
 *
 * The sample is redacted, and that matters more here than anywhere else in
 * this file: it is the only place where the content of a tool result leaves
 * the machine for the model's API. Everything else sent as state is a stub.
 */
export function resultPeek(
  text: string,
  isError: boolean,
  options: PeekOptions,
): string {
  const head = isError ? Math.floor(options.headChars / 2) : options.headChars;
  const tail = isError
    ? options.tailChars + Math.ceil(options.headChars / 2)
    : options.tailChars;
  if (text.length <= head + tail + 40) return redact(text);
  const omitted = text.length - head - tail;
  return `${redact(text.slice(0, head))}\n[… ${omitted} chars …]\n${redact(
    text.slice(-tail),
  )}`;
}

export interface SalvageOptions {
  /** Characters of the original head kept before the salvaged block. */
  headChars: number;
  /** Ceiling on the salvaged block itself. */
  maxChars: number;
}

/** As many items as fit `budget`, in order. */
function capped(items: readonly string[], budget: number): string[] {
  const out: string[] = [];
  let used = 0;
  for (const item of items) {
    const cost = item.length + 2;
    if (used + cost > budget) break;
    out.push(item);
    used += cost;
  }
  return out;
}

/**
 * The replacement for a tool result that was not kept: a bounded head, a note
 * saying what went, and the identifiers and failures recovered from the whole
 * text. Upstream keeps only the head, which for a listing or a build log is
 * preamble — the parts worth keeping are further down.
 *
 * Salvage is an improvement to a path that already worked, so it must never
 * be the reason compaction fails: any error falls back to the plain head.
 */
export function salvagedResultText(
  text: string,
  isError: boolean,
  options: SalvageOptions,
): string {
  if (text.length <= options.headChars + 120) return text;
  const head = options.headChars > 0 ? `${text.slice(0, options.headChars)}\n` : '';
  const dropped = text.length - options.headChars;

  let block = '';
  try {
    const errors = capped(salvageErrorLines(text), options.maxChars);
    const spent = errors.reduce((sum, line) => sum + line.length + 2, 0);
    const refs = capped(salvageIdentifiers(text), options.maxChars - spent);
    block = [
      errors.length > 0 ? `errors: ${errors.join(' | ')}` : '',
      refs.length > 0 ? `refs: ${refs.join(' ')}` : '',
    ]
      .filter((line) => line.length > 0)
      .join('\n');
  } catch {
    block = '';
  }

  const note = `[fast-jev-compaction dropped ${dropped} chars of this tool result${
    isError ? ' (error)' : ''
  }; ${
    block.length > 0
      ? 'salvaged below, not verbatim — re-run the tool for the full output]'
      : 're-run the tool if needed]'
  }`;

  return block.length > 0 ? `${head}${note}\n${block}` : `${head}${note}`;
}
