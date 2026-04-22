import { app } from "/scripts/app.js";

function toggleQualityWidgets(qualityWidget, compressWidget, isPNG) {
    const dark_grey = "#444";
    const light_grey = "#666";

    qualityWidget.disabled = isPNG;
    compressWidget.disabled = !isPNG;

    const applyStyle = (widget, disabled) => {
        const color = disabled ? dark_grey : undefined;
        const textColor = disabled ? light_grey : undefined;

        if (widget.options) {
            widget.options.color = color;
            widget.options.text_color = textColor;
        }

        try { widget.color = color; } catch (e) {}
        try { widget.text_color = textColor; } catch (e) {}

        widget.label = `${widget.original_displayName} ${disabled ? "(Disabled)" : ""}`;
    };

    applyStyle(qualityWidget, isPNG);
    applyStyle(compressWidget, !isPNG);
}

function initOriginalName (widget) {
    widget.original_displayName = widget.name.replace("_", " ").replace(/\b\w/g, char => char.toUpperCase());
    widget.label = widget.original_displayName;
}

app.registerExtension({
    name: "Wakaura.SaveImageAdvanced",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SaveImageAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const formatWidget = this.widgets.find((w) => w.name === "format");
                const qualityWidget = this.widgets.find((w) => w.name === "quality");
                const compressWidget = this.widgets.find((w) => w.name === "compress_level");

                if (formatWidget && qualityWidget && compressWidget) {
                    initOriginalName(formatWidget);
                    initOriginalName(qualityWidget);
                    initOriginalName(compressWidget);

                    const updateWidgets = () => {
                        const isPNG = formatWidget.value === "png";
                        toggleQualityWidgets(qualityWidget, compressWidget, isPNG);
                        this.setDirtyCanvas(true, true);
                    };

                    const callback = formatWidget.callback;
                    formatWidget.callback = function () {
                        const result = callback ? callback.apply(this, arguments) : undefined;
                        updateWidgets();
                        return result;
                    };

                    // Initial state check
                    setTimeout(() => updateWidgets(), 0);
                }

                return r;
            };
        }
    }
});
