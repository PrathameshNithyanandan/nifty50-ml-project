const SUP: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
const none = "—";
const bad = (x: any) => x === null || x === undefined || Number.isNaN(x);
export const f = {
  /** fixed decimals */
  n: (x: any, d = 3) => (bad(x) ? none : Number(x).toFixed(d)),
  /** signed fixed decimals */
  s: (x: any, d = 3) => (bad(x) ? none : (x >= 0 ? "+" : "") + Number(x).toFixed(d)),
  /** integer with separators */
  i: (x: any) => (bad(x) ? none : Math.round(Number(x)).toLocaleString("en-US")),
  /** percentage from a share */
  pct: (x: any, d = 1) => (bad(x) ? none : (100 * Number(x)).toFixed(d) + "%"),
  /** scientific notation */
  sci: (x: any, d = 1) => {
    if (bad(x)) return none;
    if (x === 0) return "0";
    const [m, e] = Number(x).toExponential(d).split("e");
    return `${m}×10${e.replace("+", "").split("").map((c) => SUP[c]).join("")}`;
  },
  /** p-value */
  p: (x: any) => {
    if (bad(x)) return none;
    if (x === 0) return "≈ 0";
    return x < 0.001 ? f.sci(x, 1) : Number(x).toFixed(3);
  },
  /** short number for hyperparameters */
  g: (x: any) => {
    if (bad(x)) return none;
    const v = Number(x);
    return Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? f.sci(v, 0) : String(Number(v.toPrecision(4)));
  },
};
