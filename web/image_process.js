import { app } from "/scripts/app.js";

const qualityWidgetsConfig = {
    "quality": ["jpg", "webp"],
    "compress_level": ["png"],
    "tiff_compression": ["tiff"],
};

function applyWidgetStyle(widget, disabled) {
    const dark_grey = "#444";
    const light_grey = "#666";

    const color = disabled ? dark_grey : undefined;
    const textColor = disabled ? light_grey : undefined;

    widget.disabled = disabled;
    if (widget.options) {
        widget.options.color = color;
        widget.options.text_color = textColor;
    }

    try { widget.color = color; } catch (e) { }
    try { widget.text_color = textColor; } catch (e) { }

    widget.label = `${widget.original_displayName} ${disabled ? "(Disabled)" : ""}`;
}

function toggleQualityWidgets(widgets, qualityType) {
    for (const name in qualityWidgetsConfig) {
        const widget = widgets[name];
        if (widget) {
            const enabledTypes = qualityWidgetsConfig[name];
            const isDisabled = !enabledTypes.includes(qualityType);
            applyWidgetStyle(widget, isDisabled);
        }
    }
}

function initOriginalName(widget) {
    if (widget) {
        widget.original_displayName = widget.name.replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase());
        widget.label = widget.original_displayName;
    }
}

app.registerExtension({
    name: "Wakaura.SaveImageAdvanced",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SaveImageAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const widgets = this.widgets.reduce((acc, w) => {
                    acc[w.name] = w;
                    return acc;
                }, {});

                const formatWidget = widgets["format"];

                if (formatWidget) {
                    [formatWidget, widgets["quality"], widgets["compress_level"], widgets["tiff_compression"]].forEach(initOriginalName);

                    const updateWidgets = () => {
                        const qualityType = formatWidget.value;
                        toggleQualityWidgets(widgets, qualityType);
                        this.setDirtyCanvas(true, true);
                    };

                    const originalCallback = formatWidget.callback;
                    formatWidget.callback = function () {
                        const result = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                        updateWidgets();
                        return result;
                    };

                    setTimeout(updateWidgets, 0);
                }
                return r;
            };
        }
    }
});
