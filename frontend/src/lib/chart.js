// Shared chart colours (categorical slots 1 and 2 of the reference palette) and the
// assumption sliders of the decision panel.
export const BUY = "#2a78d6";
export const RENT = "#eb6834";

// Sliders: value shown in display units, stored / sent as fractions where unit is "%".
export const SLIDERS = [
    { key: "H", label: "Years you'd stay", min: 1, max: 30, step: 1, unit: "years" },
    { key: "d", label: "Down payment", min: 0, max: 100, step: 1, unit: "%" },
    { key: "r", label: "Mortgage rate", min: 0, max: 12, step: 0.05, unit: "%" },
    { key: "g", label: "House price growth / year", min: -5, max: 15, step: 0.1, unit: "%", forecast: "g" },
    { key: "q", label: "Rent growth / year", min: -3, max: 10, step: 0.1, unit: "%", forecast: "q" },
    { key: "k", label: "Discount rate", min: 0, max: 12, step: 0.25, unit: "%" },
];
