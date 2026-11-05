import { lazy, Suspense, useMemo } from "react";
import { usePalette } from "../lib/colors";
import { useNarrow } from "../lib/useNarrow";

const Plot: any = lazy(async () => {
  const [p, fac]: any[] = await Promise.all([import("plotly.js-dist-min"), import("react-plotly.js/factory")]);
  const Plotly = p.default ?? p;
  const factory = fac.default?.default ?? fac.default;
  return { default: factory(Plotly) };
});

export type ChartProps = {
  data: any[];
  layout?: any;
  height?: number;
  heightNarrow?: number;
  x?: string;
  y?: string;
  hover?: "closest" | "x" | "x unified";
  legend?: boolean;
  ariaLabel: string;
  note?: string;
};

function merge(a: any, b: any) {
  const out = { ...a };
  for (const k of Object.keys(b ?? {})) out[k] = b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a[k] ? merge(a[k], b[k]) : b[k];
  return out;
}

export function Chart({ data, layout, height = 380, heightNarrow, x, y, hover = "closest", legend = true, ariaLabel, note }: ChartProps) {
  const pal = usePalette();
  const narrow = useNarrow();
  const h = narrow ? heightNarrow ?? Math.max(300, height - 60) : height;
  const full = useMemo(() => {
    const axis = (title?: string) => ({
      title: title ? { text: title, font: { size: narrow ? 12 : 13, color: pal.ink2 }, standoff: 8 } : undefined,
      gridcolor: pal.grid, linecolor: pal.axis, zerolinecolor: pal.axis, tickfont: { size: narrow ? 11 : 12, color: pal.muted }, automargin: true, showline: true,
    });
    const base = {
      autosize: true, height: h,
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter Variable, system-ui, sans-serif", color: pal.ink2, size: narrow ? 12 : 13 },
      margin: { l: narrow ? 46 : 60, r: narrow ? 12 : 20, t: 12, b: 48 },
      hovermode: hover,
      hoverlabel: { bgcolor: pal.surface, bordercolor: pal.axis, font: { color: pal.ink, size: 13, family: "Inter Variable, system-ui, sans-serif" } },
      showlegend: legend,
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: narrow ? 11 : 12, color: pal.ink2 }, bgcolor: "rgba(0,0,0,0)" },
      xaxis: axis(x), yaxis: axis(y),
      dragmode: narrow ? false : "zoom",
    };
    return merge(base, layout ?? {});
  }, [layout, h, x, y, hover, legend, narrow, pal]);
  return (
    <figure className="m-0 min-w-0" aria-label={ariaLabel} role="img">
      <Suspense fallback={<div className="skeleton" style={{ height: h }} />}>
        <Plot data={data} layout={full} config={{ displaylogo: false, displayModeBar: false, responsive: true, doubleClick: "reset" }} useResizeHandler style={{ width: "100%", height: h }} />
      </Suspense>
      {note && <figcaption className="text-xs text-ink2 mt-1">{note}</figcaption>}
    </figure>
  );
}
export const Skeleton = ({ h = 320 }: { h?: number }) => <div className="skeleton" style={{ height: h }} aria-busy="true" />;
