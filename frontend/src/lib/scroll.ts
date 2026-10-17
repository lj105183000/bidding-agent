/**
 * 滚动跟随的判定（纯函数，便于单测）。
 *
 * ★ 修掉的行为
 * ----------
 * 原来无论用户在哪，每来一个 token 都执行 `scrollTop = scrollHeight`：
 * 用户想往上翻看上文，刚滚上去就被拽回底部（流式期间基本无法回看）。
 * 正确做法是只在**用户本来就在底部附近**时跟随。
 *
 * ★ 为什么用阈值而不是精确相等
 * ------------------------
 * ① 浏览器滚动位置是小数（`scrollTop` 可为 123.5），精确相等几乎不成立；
 * ② 触控板/滚轮的"到底"往往差几个像素；
 * ③ 留余量后轻微回滚不会立刻退出跟随（否则跟随会时灵时不灵）。
 */
export const FOLLOW_THRESHOLD_PX = 80

/** 判断滚动容器是否处于"底部附近"。传入普通对象而非元素，便于单测。 */
export function isNearBottom(
  metrics: { scrollHeight: number; scrollTop: number; clientHeight: number },
  threshold: number = FOLLOW_THRESHOLD_PX,
): boolean {
  const { scrollHeight, scrollTop, clientHeight } = metrics
  // 内容还没撑满容器（没有滚动条）时，也应视为"在底部"
  if (scrollHeight <= clientHeight) return true
  return scrollHeight - scrollTop - clientHeight < threshold
}
