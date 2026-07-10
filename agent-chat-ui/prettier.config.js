/**
 * @see https://prettier.io/docs/configuration
 * @type {import("prettier").Config}
 */
const config = {
  endOfLine: "auto",
  printWidth: 100,
  singleAttributePerLine: false,
  plugins: ["prettier-plugin-tailwindcss"],
};

export default config;
