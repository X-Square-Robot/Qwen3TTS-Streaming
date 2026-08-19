import {defineConfig, devices} from "@playwright/test";

const localChromium = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {baseURL: "http://127.0.0.1:4173", trace: "retain-on-failure"},
  webServer: {
    command: "node scripts/e2e-server.mjs",
    url: "http://127.0.0.1:4173/health",
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    {name: "chromium", use: {
      browserName: "chromium",
      launchOptions: {
        ...(localChromium ? {executablePath: localChromium} : {}),
        args: ["--autoplay-policy=no-user-gesture-required"],
      },
    }},
    {name: "firefox", use: {browserName: "firefox"}},
    {name: "webkit", use: {browserName: "webkit"}},
    {name: "mobile-chromium", use: {
      ...devices["Pixel 7"],
      launchOptions: localChromium ? {executablePath: localChromium} : {},
    }},
    {name: "mobile-webkit", use: {...devices["iPhone 15"]}},
  ],
});
