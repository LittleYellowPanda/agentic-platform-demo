import { useCallback, useEffect, useState } from "react";

import { errorMessage } from "../api/client";

export type AsyncData<T> = {
  data: T | undefined;
  error: string | null;
  isLoading: boolean; // true only for the very first load
  reload: () => void;
};

/**
 * Load data when the component appears, and again each time `reload()` is called.
 *
 * - Previous data stays on screen while reloading: polling every few seconds
 *   doesn't make the page flicker.
 * - `load` is read once per (re)load. If it depends on a prop (e.g. a KB id),
 *   the parent must remount the component when that prop changes, with
 *   `key={kb.id}`, so the data of one KB never shows up under another.
 */
export function useAsyncData<T>(load: () => Promise<T>): AsyncData<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    // If the component disappears (or reloads) before the response arrives,
    // ignore that late response instead of writing it into the state.
    let ignore = false;

    load().then(
      (result) => {
        if (!ignore) {
          setData(result);
          setError(null);
        }
      },
      (err: unknown) => {
        if (!ignore) setError(errorMessage(err));
      },
    );

    return () => {
      ignore = true;
    };
    // `load` is a new function at every render: listing it here would reload in
    // a loop. Reloads are triggered explicitly, through `version`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version]);

  // useCallback keeps the same function across renders, so it can safely be
  // used in another effect's dependency list (the polling interval).
  const reload = useCallback(() => setVersion((v) => v + 1), []);

  return { data, error, isLoading: data === undefined && error === null, reload };
}
