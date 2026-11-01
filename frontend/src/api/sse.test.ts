// Specification of parseSSE. Run from frontend/:  node --test src/api/sse.test.ts
// (Node reads TypeScript directly since v22.18: no test library to install.)
import assert from "node:assert/strict";
import { test } from "node:test";

import { parseSSE, type SseEvent } from "./sse.ts";

// Feed chunks one after the other, like the network does, and collect all events
function feed(...chunks: string[]): { events: SseEvent[]; rest: string } {
  let rest = "";
  const events: SseEvent[] = [];
  for (const chunk of chunks) {
    const result = parseSSE(rest, chunk);
    events.push(...result.events);
    rest = result.rest;
  }
  return { events, rest };
}

test("one complete event", () => {
  assert.deepEqual(feed('event: step\ndata: {"position": 0}\n\n'), {
    events: [{ event: "step", data: '{"position": 0}' }],
    rest: "",
  });
});

test("two events in one chunk", () => {
  const { events } = feed("event: a\ndata: 1\n\nevent: b\ndata: 2\n\n");
  assert.deepEqual(events, [
    { event: "a", data: "1" },
    { event: "b", data: "2" },
  ]);
});

test("step 2: an event cut in two chunks waits for its end", () => {
  const first = parseSSE("", 'event: step\ndata: {"pos');
  assert.deepEqual(first.events, []);
  assert.equal(first.rest, 'event: step\ndata: {"pos');

  const second = parseSSE(first.rest, 'ition": 0}\n\n');
  assert.deepEqual(second.events, [{ event: "step", data: '{"position": 0}' }]);
  assert.equal(second.rest, "");
});

test("step 2: the blank line itself can be cut between two chunks", () => {
  const { events, rest } = feed("event: done\ndata: {}\n", "\n");
  assert.deepEqual(events, [{ event: "done", data: "{}" }]);
  assert.equal(rest, "");
});

test("step 2: an event cut in the middle of a field name still arrives whole", () => {
  const { events, rest } = feed("event: a\ndata: 1\n\nevent: b\nda", "ta: 2\n\n");
  assert.deepEqual(
    events.map((e) => e.event),
    ["a", "b"],
  );
  assert.equal(rest, "");
});

test("step 3: keep-alive comments are ignored", () => {
  const { events } = feed(": keep-alive\n\n", "event: done\ndata: {}\n\n", ": keep-alive\n\n");
  assert.deepEqual(events, [{ event: "done", data: "{}" }]);
});

test("step 3: only the first colon separates field and value", () => {
  const { events } = feed('event: step\ndata: {"content": "time: 10:30"}\n\n');
  assert.equal(events[0].data, '{"content": "time: 10:30"}');
});

test("step 3: the space after the colon is optional", () => {
  assert.deepEqual(feed("event:step\ndata:{}\n\n").events, [{ event: "step", data: "{}" }]);
});

test("step 3: several data lines are joined with a newline", () => {
  assert.equal(feed("data: line 1\ndata: line 2\n\n").events[0].data, "line 1\nline 2");
});

test("step 3: without an event line, the name is 'message'", () => {
  assert.equal(feed("data: hello\n\n").events[0].event, "message");
});

test("step 1: Windows line endings work too", () => {
  assert.deepEqual(feed("event: done\r\ndata: {}\r\n\r\n").events, [
    { event: "done", data: "{}" },
  ]);
});

test("step 4: a block without data is not an event", () => {
  assert.deepEqual(feed("event: lonely\n\n").events, []);
});

test("the real stream: every byte in its own chunk still gives the same events", () => {
  const stream =
    'event: run_started\ndata: {"run_id": 7}\n\n: keep-alive\n\n' +
    'event: step\ndata: {"kind": "llm_call"}\n\nevent: done\ndata: {"status": "completed"}\n\n';
  const { events, rest } = feed(...stream.split(""));
  assert.deepEqual(
    events.map((e) => e.event),
    ["run_started", "step", "done"],
  );
  assert.equal(JSON.parse(events[2].data).status, "completed");
  assert.equal(rest, "");
});
