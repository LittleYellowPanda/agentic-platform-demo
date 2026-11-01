import type { ReactNode } from "react";

// Models write a little Markdown (**bold**, `code`). A full Markdown library would be
// overkill: this handles those two, and line breaks come from CSS (pre-wrap).
// Safe by construction: React escapes the text, nothing is injected as HTML.
export default function AnswerText({ text }: { text: string }) {
  const parts: ReactNode[] = text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    return part;
  });
  return <div className="answer-text">{parts}</div>;
}
