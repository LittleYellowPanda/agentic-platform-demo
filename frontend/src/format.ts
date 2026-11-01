// Small display helpers shared by several components

// "intfloat/multilingual-e5-small" → "multilingual-e5-small"
export function shortModelName(name: string): string {
  return name.split("/").pop() ?? name;
}

export function pluralize(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

// 850 -> "850 ms", 14495 -> "14.5 s"
export function formatDuration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "–";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

// "2026-10-30T14:02:11Z" -> "14:02" today, "30 Oct, 14:02" otherwise
export function formatTime(iso: string): string {
  const date = new Date(iso);
  const time = date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  if (date.toDateString() === new Date().toDateString()) return time;
  return `${date.toLocaleDateString(undefined, { day: "numeric", month: "short" })}, ${time}`;
}
