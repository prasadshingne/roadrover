import { ExtensionContext } from "@lichtblick/suite";
import { initPanel } from "./SessionPanel";

export function activate(extensionContext: ExtensionContext): void {
  extensionContext.registerPanel({
    name: "Roadrover Session Manager",
    initPanel,
  });
}
