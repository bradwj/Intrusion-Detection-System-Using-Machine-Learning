export const formatPythonVarName = (submodelName) => {
  return submodelName
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};