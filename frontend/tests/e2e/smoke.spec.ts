import { expect, test } from "@playwright/test";

test("home and graph pages render", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("PaperTrail")).toBeVisible();
  await expect(page.getByRole("button", { name: /刷新列表|Refresh list/ })).toBeVisible();

  await page.getByRole("link", { name: /图谱工作台|Graph Workspace/ }).click();
  await expect(page).toHaveURL(/\/graph/);
  await expect(page.getByText(/关系模式|Relation mode/)).toBeVisible();
  await expect(page.getByRole("tab", { name: /详情|Details/ })).toBeVisible();
  await expect(page.getByRole("tab", { name: /论文问答|Paper QA/ })).toBeVisible();
});
