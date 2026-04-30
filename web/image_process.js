import { app } from "/scripts/app.js";

const qualityWidgetsConfig = {
    "disable_metadata": ["png", "jpg", "webp", "tiff"],
    "join_alpha": ["png", "webp", "tiff", "bmp"],
    "invert_alpha": ["png", "webp", "tiff", "bmp"],
    "quality": ["jpg", "webp"],
    "compress_level": ["png"],
    "tiff_compression": ["tiff"],
};

const alphaExclusiveConfig = {
    "join_alpha": ["32bit"],
    "invert_alpha": ["32bit"]
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

function toggleQualityWidgets(widgets, value, config) {
    for (const name in config) {
        const widget = widgets[name];
        if (widget) {
            const enabledTypes = config[name];
            const isDisabled = !enabledTypes.includes(value);
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
                    [
                        formatWidget,
                        widgets["disable_metadata"],
                        widgets["join_alpha"],
                        widgets["invert_alpha"],
                        widgets["quality"],
                        widgets["compress_level"],
                        widgets["tiff_compression"]
                    ].forEach(initOriginalName);

                    const updateWidgets = () => {
                        const qualityType = formatWidget.value;
                        toggleQualityWidgets(widgets, qualityType, qualityWidgetsConfig);
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

app.registerExtension({
    name: "Wakaura.SaveImageAdvancedBMP",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SaveImageAdvancedBMP") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const widgets = this.widgets.reduce((acc, w) => {
                    acc[w.name] = w;
                    return acc;
                }, {});

                const bit_depth_widget = widgets["bit_depth"];

                if (bit_depth_widget) {
                    [bit_depth_widget,widgets["join_alpha"], widgets["invert_alpha"]].forEach(initOriginalName);
                    
                    const updateWidgets = () => {
                        toggleQualityWidgets(widgets, bit_depth_widget.value, alphaExclusiveConfig);
                        this.setDirtyCanvas(true, true);
                    };

                    const originalCallback = bit_depth_widget.callback;
                    bit_depth_widget.callback = function () {
                        const result = originalCallback ? originalCallback.apply(this, arguments) : undefined;
                        updateWidgets();
                        return result;
                    };

                    setTimeout(updateWidgets, 0);
                }

                return r;
            }
        }
    }
});