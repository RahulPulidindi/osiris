import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Self-hosted fonts: identical rendering offline, no third-party request.
import "@fontsource-variable/instrument-sans";
import "@fontsource-variable/spline-sans-mono";
import "@fontsource/instrument-serif";

import "./styles/theme.css";
import { App } from "./App";

const root = document.getElementById("root");
if (!root) throw new Error("#root not found");

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
