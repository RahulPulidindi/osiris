// Preview the main dashboard by stubbing /api/connection as fully set up.
import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 960 },
  deviceScaleFactor: 2,
});

const errors = [];
page.on("pageerror", (e) => errors.push(`PAGEERROR: ${e.message}`));

await page.route("**/api/connection", (route) =>
  route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({
      robinhood_linked: true,
      broker: "robinhood-mcp",
      mode: "live",
      armed: true,
      risk_acknowledged: true,
      restart_required: false,
      connect_command: "python -m osiris.connect",
    }),
  }),
);

await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.waitForTimeout(2500);
await page.screenshot({ path: "shots/dashboard-main.png", fullPage: true });
console.log("captured dashboard-main");
console.log(errors.length ? `ERRORS:\n${errors.join("\n")}` : "no page errors");
await browser.close();
