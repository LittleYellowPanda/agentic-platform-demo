// Server-Sent Events, browser side: turn network chunks into events.
//
// The network delivers CHUNKS, not events. A chunk can hold half an event, exactly
// one, or several. An event ends with a blank line ("\n\n"); everything after the
// last blank line is an unfinished event: keep it, and glue the next chunk to it.
//
//   chunk 1: 'event: step\ndata: {"pos'          -> no event yet, rest = the whole text
//   chunk 2: 'ition": 0}\n\nevent: done\ndata: {}\n\n'
//                                                -> 2 events, rest = ""

export type SseEvent = {
  event: string; // the "event:" line; "message" when there is none (SSE default)
  data: string; // the "data:" lines joined with "\n" (we send one line of JSON)
};

export type ParseResult = {
  events: SseEvent[];
  rest: string; // the unfinished end, to pass back with the next chunk
};

export function parseSSE(rest: string, chunk: string): ParseResult {
  // Step 1: glue the new chunk to what was left over, normalize "\r\n" to "\n"
  const text = (rest + chunk).replace(/\r\n/g, "\n");

  // Step 2: cut on blank lines. The LAST piece is unfinished: it becomes the new rest
  const blocks = text.split("\n\n");
  const unfinished = blocks.pop() ?? "";

  const events: SseEvent[] = [];
  for (const block of blocks) {
    // Step 3: read the lines of one block:
    //   ":..."        a comment (our keep-alive): ignore the line
    //   "event: x"    the event name
    //   "data: y"     one line of data (several lines are joined with "\n")
    //   only the FIRST ":" separates field and value (JSON contains ":" too),
    //   and one space after it is optional
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line === "" || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    // Step 4: a block with no data line (e.g. only a keep-alive) is not an event
    if (data.length > 0) events.push({ event, data: data.join("\n") });
  }
  return { events, rest: unfinished };
}
