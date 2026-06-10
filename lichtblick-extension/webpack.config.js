// @ts-check
const path = require("path");

/** @type {import("webpack").Configuration} */
module.exports = {
  entry: "./src/index.ts",
  output: {
    filename: "extension.js",
    path: path.join(__dirname, "dist"),
    library: { type: "commonjs2" },
  },
  // React and @lichtblick/suite are provided by the Lichtblick host — do not bundle them.
  externals: {
    "@lichtblick/suite": "@lichtblick/suite",
    react: "react",
    "react-dom": "react-dom",
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
  },
  // Extensions should not rely on Node built-ins
  target: "web",
};
