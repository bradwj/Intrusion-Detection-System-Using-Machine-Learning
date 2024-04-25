import { OverlayToaster, Position } from "@blueprintjs/core";

/** Singleton toaster instance. Create separate instances for different options. */
export const AppToaster = OverlayToaster.createAsync({
  position: Position.TOP,
});

export const showToast = async (props = {}) => {
  (await AppToaster).show(props);
};