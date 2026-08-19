import tseslint from "typescript-eslint";

export default tseslint.config(
  {ignores: ["**/dist/**", "**/node_modules/**", "packages/demo/public/docs.json", "packages/demo/public/docs-assets/**"]},
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
    },
  },
);
