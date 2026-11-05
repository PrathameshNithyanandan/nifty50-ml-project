// Fails if a number in the site source looks like a result instead of coming from web/public/data.
// It walks the TypeScript AST of src/, so class names, import paths, data keys and comments are not counted.
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const SRC = join(ROOT, "src");

// Method constants and labels: fixed by the design of the analysis, not measured from the data.
const ALLOWED_TEXT_NUMBERS = new Set([
  "0", "1", "2", "3", "4", "5", "6", // small counts, unit and practice numbers, formula constants
  "0.5", // the coin-flip reference and the decision threshold
  "0.05", // the significance level and the illustrative learning rate
  "10", "20", "50", "60", // rolling-window lengths in feature descriptions
  "80", "95", "99", "100", // named rules, quantile levels and the 100·ln(Close) scale
  "252", // trading days per year in σ√252
  "2000", "2008", "2020", "2021", // named periods and the dataset's title
  "98", // the live demo's 1st to 99th percentile training range
]);
const IGNORED_PHRASES = [/rgba?\([^)]*\)/g, /\b(?:19|20)\d\d-\d\d-\d\d\b/g, /πσ²\/8/g, /NIFTY-50/g, /\bt\+1\b/g, /\bt−1\b/g, /\b1st\b/g, /\b99th\b/g];
const SKIP_ATTRS = new Set(["className", "style", "id", "key", "to", "href", "k", "d", "type", "step", "min", "max", "viewBox", "role", "tabIndex", "height", "heightNarrow", "x", "y", "width", "rx", "fontSize", "fontWeight", "strokeWidth", "textAnchor", "transform", "initialSort", "maxHeight", "aria-label", "ariaLabel"]);
const LAYOUT_KEYS = new Set(["size", "width", "height", "l", "r", "t", "b", "standoff", "thickness", "dtick", "tickangle", "opacity", "xgap", "ygap", "range", "y", "x", "x0", "x1", "y0", "y1", "zmin", "zmax", "margin"]);

function files(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? files(p) : /\.(ts|tsx)$/.test(n) && !n.endsWith(".d.ts") ? [p] : [];
  });
}

// Every value in results.json with at least three significant digits, to catch a literal copied from it.
const results = JSON.parse(readFileSync(join(ROOT, "public/data/results.json"), "utf8"));
const resultValues = new Set();
(function walk(o) {
  if (typeof o === "number") {
    const s = Math.abs(o).toPrecision(12).replace(/\.?0+$/, "").replace(".", "").replace(/^0+/, "");
    if (s.length >= 3) for (const d of [2, 3, 4, 5, 6]) resultValues.add(Number(Math.abs(o).toFixed(d)));
  } else if (o && typeof o === "object") Object.values(o).forEach(walk);
})(results);

const problems = [];
function report(file, node, sf, text, why) {
  const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
  problems.push(`${relative(ROOT, file)}:${line + 1}  ${why}: ${JSON.stringify(text.trim().slice(0, 90))}`);
}

function checkText(file, node, sf, text) {
  let t = text;
  for (const re of IGNORED_PHRASES) t = t.replace(re, " ");
  // a number glued to a letter or underscore is an identifier such as PC1, r_lag0 or u1
  for (const m of t.matchAll(/(?<![A-Za-z_\d.])\d+(?:[.,]\d+)*(?![A-Za-z_\d])/g)) {
    const token = m[0].replace(/,/g, "");
    if (!ALLOWED_TEXT_NUMBERS.has(token)) report(file, node, sf, text, `number ${m[0]} in text`);
  }
}

function inSkippedContext(node) {
  for (let p = node.parent; p; p = p.parent) {
    if (ts.isJsxAttribute(p) && SKIP_ATTRS.has(p.name.getText())) return true;
    if (ts.isImportDeclaration(p)) return true;
    if (ts.isPropertyAssignment(p) && ["hovertemplate", "symbol", "color", "dash", "fillcolor", "family", "bgcolor", "fill", "mode", "bordercolor"].includes(p.name.getText())) return true;
    if (ts.isCallExpression(p) && /^(useChart|parseInt|getElementById|startsWith|replace|matchMedia)$/.test(p.expression.getText().split(".").pop())) return true;
    if (ts.isElementAccessExpression(p)) return true;
    if (ts.isBinaryExpression(p) && [ts.SyntaxKind.EqualsEqualsEqualsToken, ts.SyntaxKind.ExclamationEqualsEqualsToken].includes(p.operatorToken.kind)) return true;
  }
  return false;
}

for (const file of files(SRC)) {
  const rel = relative(SRC, file);
  if (rel === "lib/colors.ts") continue; // only colour definitions
  const sf = ts.createSourceFile(file, readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true, file.endsWith("x") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  (function visit(node) {
    if (ts.isJsxText(node)) checkText(file, node, sf, node.getText(sf));
    else if ((ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) && !inSkippedContext(node)) {
      // a property key or a CSS-like token is not reader-facing text
      const parent = node.parent;
      const isKey = ts.isPropertyAssignment(parent) && parent.name === node;
      if (!isKey) checkText(file, node, sf, node.text);
    } else if ((ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) && !inSkippedContext(node)) checkText(file, node, sf, node.text);
    else if (ts.isNumericLiteral(node)) {
      const v = Number(node.text);
      const digits = node.text.replace(/^0\.?0*|\.|e.*$/g, "").replace(/0+$/, "");
      const parent = node.parent;
      const key = ts.isPropertyAssignment(parent) ? parent.name.getText() : "";
      const elementOfLayout = ts.isArrayLiteralExpression(parent) && ts.isPropertyAssignment(parent.parent) && LAYOUT_KEYS.has(parent.parent.name.getText());
      const isLayout = LAYOUT_KEYS.has(key) || elementOfLayout;
      const attr = ts.isJsxExpression(parent) && ts.isJsxAttribute(parent.parent) ? parent.parent.name.getText() : "";
      if (!SKIP_ATTRS.has(attr) && !isLayout && digits.length >= 3 && v !== 100 && v !== 1000) report(file, node, sf, node.text, "numeric literal looks like a result");
      if (!SKIP_ATTRS.has(attr) && !isLayout && digits.length >= 3 && resultValues.has(Math.abs(v))) report(file, node, sf, node.text, "numeric literal equals a value in results.json");
    }
    ts.forEachChild(node, visit);
  })(sf);
}

if (problems.length) {
  console.error(`FAILED: ${problems.length} possible hard-coded number(s)\n` + problems.join("\n"));
  process.exit(1);
}
console.log(`OK: no hard-coded result numbers in ${files(SRC).length} source files.`);
