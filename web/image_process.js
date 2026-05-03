import { app } from "/scripts/app.js";

const qualityWidgetsConfig = {
    "disable_metadata": ["png", "jpg", "webp", "tiff"],
    "join_alpha": ["png", "webp", "tiff", "bmp", "tga"],
    "invert_alpha": ["png", "webp", "tiff", "bmp", "tga"],
    "quality": ["jpg", "webp"],
    "compress_level": ["png"],
    "tiff_compression": ["tiff"],
    "tga_rle": ["tga"],
};

const alphaExclusiveConfig = {
    "join_alpha": ["32bit"],
    "invert_alpha": ["32bit"]
};

const colorPatchMergeConfig = {
    "neighborhood": ["Smooth"],
    "min_area": ["Unify"],
    "iterations": ["Unify"],
    "use_lab": ["Unify"]
};

class WidgetManager {
    constructor(node, config) {
        this.node = node;
        this.config = config;
        this.widgets = node.widgets.reduce((acc, w) => {
            acc[w.name] = w;
            return acc;
        }, {});
    }

    initWidget(name, prettify = true, customName = "") {
        const widget = this.widgets[name];
        if (widget) {
            if (customName !== "") {
                widget.original_displayName = customName;
            } else {
                if (prettify) {
                    widget.original_displayName = widget.name.replace(/_/g, " ").replace(/\b\w/g, char => char.toUpperCase());
                } else {
                    widget.original_displayName = widget.name;
                }
            }
            widget.label = widget.original_displayName;
        }
        return widget;
    }

    initWidgets(names) {
        names.forEach(name => this.initWidget(name));
    }

    applyStyle(widget, disabled) {
        const dark_grey = "#444";
        const light_grey = "#666";

        const color = disabled ? dark_grey : undefined;
        const textColor = disabled ? light_grey : undefined;

        widget.disabled = !!disabled;
        if (widget.options) {
            widget.options.color = color;
            widget.options.text_color = textColor;
        }

        try { widget.color = color; } catch (e) { }
        try { widget.text_color = textColor; } catch (e) { }

        widget.label = `${widget.original_displayName}${disabled ? " (Disabled)" : ""}`;
    }

    update(triggerValue) {
        for (const name in this.config) {
            const widget = this.widgets[name];
            if (widget) {
                const enabledTypes = this.config[name];
                const isDisabled = !enabledTypes.includes(triggerValue);
                this.applyStyle(widget, isDisabled);
            }
        }
        this.node.setDirtyCanvas(true, true);
    }

    bindTrigger(triggerName) {
        const triggerWidget = this.widgets[triggerName];
        if (!triggerWidget) return;

        const originalCallback = triggerWidget.callback;
        triggerWidget.callback = (...args) => {
            const result = originalCallback ? originalCallback.apply(triggerWidget, args) : undefined;
            this.update(triggerWidget.value);
            return result;
        };

        setTimeout(() => this.update(triggerWidget.value), 0);
    }
}

app.registerExtension({
    name: "Wakaura.SaveImageAdvanced",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SaveImageAdvanced") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const manager = new WidgetManager(this, qualityWidgetsConfig);
                manager.initWidgets([
                    "format",
                    "disable_metadata",
                    "join_alpha",
                    "invert_alpha",
                    "quality",
                    "compress_level"
                ]);
                manager.initWidget("tiff_compression", false, "TIFF Compression");
                manager.initWidget("tga_rle", false, "TGA RLE Compression");
                manager.bindTrigger("format");

                return r;
            };
        };
    }
});

app.registerExtension({
    name: "Wakaura.SaveImageAdvancedBMP",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "SaveImageAdvancedBMP") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const manager = new WidgetManager(this, alphaExclusiveConfig);
                manager.initWidgets(["bit_depth", "join_alpha", "invert_alpha"]);
                manager.bindTrigger("bit_depth");

                return r;
            };
        };
    }
});

app.registerExtension({
    name: "Wakaura.ColorPatchMerge",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ColorPatchMerge") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const manager = new WidgetManager(this, colorPatchMergeConfig);
                manager.initWidgets([
                    "merge_solution",
                    "neighborhood",
                    "min_area",
                    "iterations",
                    "use_lab",
                ]);
                manager.bindTrigger("merge_solution");

                return r;
            };
        };
    }
});