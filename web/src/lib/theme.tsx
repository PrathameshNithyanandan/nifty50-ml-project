import { createContext, ReactNode, useContext, useEffect, useState } from "react";

type Mode = "light" | "dark";
const Ctx = createContext<{ mode: Mode; toggle: () => void }>({ mode: "light", toggle: () => {} });
function initial(): Mode {
  try { const s = localStorage.getItem("theme"); if (s === "light" || s === "dark") return s; } catch { /* storage may be blocked */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
export function ThemeProvider({ children }: { children: ReactNode }) {
  const forced = new URLSearchParams(window.location.search).get("theme");
  const [mode, setMode] = useState<Mode>(forced === "dark" || forced === "light" ? forced : initial);
  useEffect(() => {
    document.documentElement.dataset.theme = mode;
    try { localStorage.setItem("theme", mode); } catch { /* storage may be blocked */ }
  }, [mode]);
  return <Ctx.Provider value={{ mode, toggle: () => setMode((m) => (m === "light" ? "dark" : "light")) }}>{children}</Ctx.Provider>;
}
export const useTheme = () => useContext(Ctx);
