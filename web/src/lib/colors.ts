import { useTheme } from "./theme";

// Categorical slots: the same eight hues, stepped for each surface.
export const SLOTS = {
  light: { blue: "#2a78d6", orange: "#eb6834", aqua: "#1baf7a", yellow: "#eda100", magenta: "#e87ba4", green: "#008300", violet: "#6250d6", red: "#e34948", grey: "#7c7b75", grey2: "#a8a7a0" },
  dark: { blue: "#3987e5", orange: "#d95926", aqua: "#199e70", yellow: "#c98500", magenta: "#d55181", green: "#1f9d1f", violet: "#9085e9", red: "#e66767", grey: "#a3a29a", grey2: "#77766f" },
};
type Slot = keyof typeof SLOTS.light;

// A model keeps its hue on every chart. Baselines are always neutral grey.
const MODEL: Record<string, Slot> = {
  persistence: "grey", majority: "grey", same_as_today: "grey2",
  ols_a: "blue", ols_b_raw: "blue", ols_b_eng: "blue",
  huber_a: "orange", huber_b_eng: "orange",
  ridge_a: "aqua", ridge_b_raw: "aqua", ridge_b_eng: "aqua",
  blr_a: "violet", blr_b_eng: "violet",
  lda: "aqua", gnb: "yellow", logistic: "blue", bayes_logistic: "violet", linear_svm: "orange", rbf_svm: "magenta",
  cart: "yellow", bagging: "aqua", random_forest: "green", adaboost: "orange", gradient_boosting: "blue", stacking: "violet", gam: "magenta", bma: "red",
  hmm: "magenta", kalman_ll: "yellow", kalman_llt: "red",
};
export const REGIME: Record<string, Slot> = { bear: "red", sideways: "grey", bull: "green" };
export const CLUSTER_ORDER: Slot[] = ["blue", "orange", "aqua", "violet", "yellow", "green", "magenta", "red"];
export const SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "triangle-down", "pentagon", "star"];

export function usePalette() {
  const { mode } = useTheme();
  const s = SLOTS[mode];
  return {
    mode, s,
    model: (key: string) => s[MODEL[key] ?? "blue"],
    regime: (name: string) => s[REGIME[name] ?? "grey"],
    cluster: (i: number) => s[CLUSTER_ORDER[i % CLUSTER_ORDER.length]],
    ink: mode === "dark" ? "#f0efec" : "#171716",
    ink2: mode === "dark" ? "#c9c8be" : "#4a4946",
    muted: mode === "dark" ? "#a3a29a" : "#6b6a65",
    grid: mode === "dark" ? "#2c2c2a" : "#e8e7e1",
    axis: mode === "dark" ? "#4a4a47" : "#c3c2b7",
    surface: mode === "dark" ? "#1a1a19" : "#fdfdfc",
    /** one-hue sequential ramp that recedes to the surface near its low end */
    seq: mode === "dark"
      ? [[0, "#1a1a19"], [0.25, "#184f95"], [0.5, "#2a78d6"], [0.75, "#6da7ec"], [1, "#cde2fb"]]
      : [[0, "#f4f8fd"], [0.25, "#b7d3f6"], [0.5, "#5598e7"], [0.75, "#1c5cab"], [1, "#0d366b"]],
  };
}
