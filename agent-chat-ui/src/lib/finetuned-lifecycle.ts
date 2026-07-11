import { getAppSettings, setAppSetting } from "@/lib/app-settings";

export const LIFECYCLE_KEY = "finetuned_model_lifecycle";

export type FinetunedLifecycle = { active: string | null; previous: string[] };

export async function getFinetunedLifecycle(): Promise<FinetunedLifecycle> {
  const value = (await getAppSettings([LIFECYCLE_KEY]))[LIFECYCLE_KEY];
  try {
    const state = JSON.parse(value) as FinetunedLifecycle;
    return state &&
      (typeof state.active === "string" || state.active === null) &&
      Array.isArray(state.previous) &&
      state.previous.every((model) => typeof model === "string")
      ? state
      : { active: null, previous: [] };
  } catch {
    return { active: null, previous: [] };
  }
}

export const promote = (state: FinetunedLifecycle, model: string): FinetunedLifecycle => ({
  active: model,
  previous: [state.active, ...state.previous]
    .filter(
      (value, index, values): value is string =>
        !!value && value !== model && values.indexOf(value) === index,
    )
    .slice(0, 20),
});

export const rollback = (state: FinetunedLifecycle): FinetunedLifecycle | null =>
  state.previous[0] ? { active: state.previous[0], previous: state.previous.slice(1) } : null;

export const remove = (state: FinetunedLifecycle, model: string): FinetunedLifecycle => {
  const previous = state.previous.filter((value) => value !== model);
  return state.active === model
    ? { active: previous[0] ?? null, previous: previous.slice(1) }
    : { active: state.active, previous };
};

export const saveFinetunedLifecycle = (state: FinetunedLifecycle) =>
  setAppSetting(LIFECYCLE_KEY, JSON.stringify(state));
