import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1680, height: 1050 } });

const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(`PAGEERROR: ${e.message}`));

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.waitForTimeout(2500);

const report = {};

// --- Command view ---
report.command = {
  heroes: await page.locator(".pnl-hero").allTextContents(),
  topbarEquity: await page.locator("header .num").first().textContent(),
  breakerRows: await page.locator(".meter").count(),
  streamLines: await page.locator(".stream-line").count(),
  pills: await page.locator(".pill").allTextContents(),
  fontDisplay: await page
    .locator("header span")
    .first()
    .evaluate((el) => getComputedStyle(el).fontFamily),
  groundColor: await page.evaluate(
    () => getComputedStyle(document.body).backgroundColor,
  ),
  heroColor: await page
    .locator(".pnl-hero")
    .nth(1)
    .evaluate((el) => getComputedStyle(el).color),
  numericFont: await page
    .locator(".num")
    .first()
    .evaluate((el) => getComputedStyle(el).fontFamily),
  tabularNums: await page
    .locator(".num")
    .first()
    .evaluate((el) => getComputedStyle(el).fontVariantNumeric),
};

// --- Ranking view ---
await page.getByRole("tab", { name: "Ranking" }).click();
await page.waitForTimeout(1200);
const rows = page.locator("tbody tr.row-interactive");
report.ranking = {
  visibleRows: await rows.count(),
  headerCount: await page.locator("thead th").count(),
  sparklines: await page.locator("tbody svg polyline").count(),
  firstRowCells: await rows.first().locator("td").allTextContents(),
};

await rows.first().click();
await page.waitForTimeout(700);
report.ranking.inspectorOpen = await page.locator("aside.drawer").isVisible();
report.ranking.inspectorHeadings = await page
  .locator("aside.drawer .label")
  .allTextContents();
report.ranking.tableStillMounted = (await rows.count()) > 0;

// --- Attribution ---
await page.getByRole("tab", { name: "Attribution" }).click();
await page.waitForTimeout(1000);
report.attribution = {
  panels: await page.locator("section.panel h2").allTextContents(),
  verdicts: await page.locator("p.border-l-2").allTextContents(),
  sectorRows: await page.locator("tbody tr").count(),
};

// --- Evaluation ---
await page.getByRole("tab", { name: "Evaluation" }).click();
await page.waitForTimeout(1200);
report.evaluation = {
  panels: await page.locator("section.panel h2").allTextContents(),
  gatePills: await page.locator(".pill").allTextContents(),
  histogramBars: await page.locator("svg rect").count(),
  histogramMarker: await page.locator("svg line").count(),
  caption: await page.locator("svg + div span").allTextContents(),
};

// --- Journal ---
await page.getByRole("tab", { name: "Journal" }).click();
await page.waitForTimeout(1000);
report.journal = {
  rows: await page.locator("tbody tr").count(),
  vetoCauses: await page.locator("li .meter").count(),
  eventLabels: [
    ...new Set(await page.locator("tbody .pill").allTextContents()),
  ].slice(0, 8),
};

report.consoleErrors = errors;
console.log(JSON.stringify(report, null, 1));
await browser.close();
