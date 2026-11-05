import { useEffect, useState } from "react";
export function useNarrow(px = 640) {
  const q = () => window.matchMedia(`(max-width: ${px}px)`).matches;
  const [n, setN] = useState(q);
  useEffect(() => {
    const m = window.matchMedia(`(max-width: ${px}px)`);
    const h = () => setN(m.matches);
    m.addEventListener("change", h);
    return () => m.removeEventListener("change", h);
  }, [px]);
  return n;
}
