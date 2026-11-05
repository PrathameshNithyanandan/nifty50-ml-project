import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";
import "@fontsource-variable/inter";
import "@fontsource-variable/source-serif-4";
import "./index.css";
import App from "./App";
import { DataProvider } from "./lib/data";
import { ThemeProvider } from "./lib/theme";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <DataProvider>
        <HashRouter>
          <App />
        </HashRouter>
      </DataProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
