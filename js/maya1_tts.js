/**
 * Maya1 TTS - FINAL IMPLEMENTATION
 * Exact layout, inline editing with cursor positioning, all features working
 */

import { app } from "../../scripts/app.js";
import { tooltips, characterPresets, emotionTags } from "./config.js";

class Maya1TTSCanvas {
    constructor(node) {
        this.node = node;
        this.tooltips = tooltips;
        this.characterPresets = characterPresets;
        this.emotionTags = emotionTags;

        // Control positions
        this.controls = {};

        // UI state
        this.hoverElement = null;
        this.emotionsCollapsed = false;
        this.editingField = null;
        this.editingValue = "";
        this.cursorPos = 0;
        this.selectionStart = null;
        this.selectionEnd = null;
        this.clickedButtons = {}; // Track button click animations
        this.keydownHandler = null; // Store bound keydown handler
        this.isDragging = false; // Track if user is dragging to select text
        this.dragStartPos = null; // Starting position for drag selection
        this.scrollOffsets = {}; // Scroll offset for each text field
        this.wheelHandler = null; // Store bound wheel handler
        this.mouseUpHandler = null; // Store bound mouseup handler for drag end
        this.lightboxOpen = false; // Track if lightbox is open
        this.lightboxField = null; // Which field is being edited in lightbox
        this.lightboxScrollOffset = 0; // Scroll offset for lightbox text

        this.setupNode();
    }

    setupNode() {
        const node = this.node;
        const self = this;

        // Disable default LiteGraph tooltip on node header
        node.desc = null;

        // Store reference to canvas for keyboard capture
        node.onAdded = function(graph) {
            self.graph = graph;
        };

        // Get ALL widget references
        this.modelWidget = node.widgets?.find(w => w.name === 'model_name');
        this.dtypeWidget = node.widgets?.find(w => w.name === 'dtype');
        this.attentionWidget = node.widgets?.find(w => w.name === 'attention_mechanism');
        this.deviceWidget = node.widgets?.find(w => w.name === 'device');
        this.voiceWidget = node.widgets?.find(w => w.name === 'voice_description');
        this.textWidget = node.widgets?.find(w => w.name === 'text');
        this.keepVramWidget = node.widgets?.find(w => w.name === 'keep_model_in_vram');
        this.chunkLongformWidget = node.widgets?.find(w => w.name === 'chunk_longform');
        this.temperatureWidget = node.widgets?.find(w => w.name === 'temperature');
        this.topPWidget = node.widgets?.find(w => w.name === 'top_p');
        this.repPenaltyWidget = node.widgets?.find(w => w.name === 'repetition_penalty');
        this.maxTokensWidget = node.widgets?.find(w => w.name === 'max_tokens');
        this.seedWidget = node.widgets?.find(w => w.name === 'seed');
        this.controlAfterWidget = node.widgets?.find(w => w.name === 'control_after_generate');
        this.debugWidget = node.widgets?.find(w => w.name === 'debug_mode');

        // PROPERLY hide all widgets
        if (node.widgets) {
            for (const widget of node.widgets) {
                widget.type = "hidden";
                widget.computeSize = () => [0, -4];
                widget.hidden = true;
            }
        }

        node.size = [500, 750];

        node.onDrawForeground = function(ctx) {
            if (this.flags.collapsed) return;
            self.drawInterface(ctx);
        };

        node.onMouseDown = function(e, pos, canvas) {
            if (e.canvasY - this.pos[1] < 0) return false;
            return self.handleMouseDown(e, pos, canvas);
        };

        node.onMouseMove = function(e, pos, canvas) {
            // Safety check: if mouse buttons are not pressed but isDragging is true, clear it
            if (self.isDragging && e.buttons === 0) {
                self.isDragging = false;
                self.dragStartPos = null;
                node.setDirtyCanvas(true, true);
            }

            if (self.isDragging) {
                return self.handleMouseDrag(e, pos, canvas);
            }
            self.handleMouseHover(e, pos, canvas);
            return false;
        };

        node.onMouseUp = function(e, pos, canvas) {
            if (self.isDragging) {
                self.isDragging = false;
                self.dragStartPos = null;
                return true;
            }
            return false;
        };
    }

    startEditing(fieldLabel, initialValue, ctrl) {
        this.editingField = fieldLabel;
        this.editingValue = initialValue;
        this.selectionStart = null;
        this.selectionEnd = null;

        // Initialize scroll offset for this field if not exists
        if (!this.scrollOffsets[fieldLabel]) {
            this.scrollOffsets[fieldLabel] = 0;
        }

        // Create and attach document-level keyboard handler
        this.keydownHandler = (e) => this.handleKeyDown(e);
        document.addEventListener('keydown', this.keydownHandler, true); // Use capture phase

        // Create and attach wheel handler for scrolling - MUST use passive: false to allow preventDefault
        this.wheelHandler = (e) => this.handleWheel(e);
        document.addEventListener('wheel', this.wheelHandler, { passive: false, capture: true });

        this.node.setDirtyCanvas(true, true);
    }

    stopEditing(save = true) {
        if (!this.editingField) return;

        // Save the value if requested
        if (save) {
            // Find the field control directly by key (not by searching all controls)
            const ctrl = this.controls[`${this.editingField}Field`];
            if (ctrl && ctrl.widget) {
                if (ctrl.isNumber) {
                    const num = parseFloat(this.editingValue);
                    if (!isNaN(num)) {
                        ctrl.widget.value = num;
                    }
                } else {
                    ctrl.widget.value = this.editingValue;
                }
            }
        }

        // Remove document-level keyboard handler
        if (this.keydownHandler) {
            document.removeEventListener('keydown', this.keydownHandler, true);
            this.keydownHandler = null;
        }

        // Remove wheel handler with same options as when added
        if (this.wheelHandler) {
            document.removeEventListener('wheel', this.wheelHandler, { passive: false, capture: true });
            this.wheelHandler = null;
        }

        // Remove mouseup handler if it exists
        if (this.mouseUpHandler) {
            document.removeEventListener('mouseup', this.mouseUpHandler, true);
            this.mouseUpHandler = null;
        }

        this.editingField = null;
        this.selectionStart = null;
        this.selectionEnd = null;
        this.isDragging = false;
        this.dragStartPos = null;
        this.node.setDirtyCanvas(true, true);
    }

    drawInterface(ctx) {
        const node = this.node;
        const margin = 15;
        const spacing = 8;
        let currentY = LiteGraph.NODE_TITLE_HEIGHT + 10;

        this.controls = {};

        // EXACT LAYOUT AS SPECIFIED:
        // 1. Character Presets
        currentY = this.drawCharacterPresets(ctx, margin, currentY);
        currentY += spacing;

        // 2. Voice Description
        currentY = this.drawTextField(ctx, "Voice Description", this.voiceWidget, margin, currentY, 70);
        currentY += spacing;

        // 3. Text Input
        currentY = this.drawTextField(ctx, "Text", this.textWidget, margin, currentY, 80);
        currentY += spacing;

        // 4. Emotion Tags (collapsible)
        currentY = this.drawEmotionSection(ctx, margin, currentY);
        currentY += spacing * 2;

        // 5. Model + Dtype (2 columns)
        currentY = this.drawTwoColumnDropdowns(ctx, "Model", this.modelWidget, "Dtype", this.dtypeWidget, margin, currentY);
        currentY += spacing;

        // 6. Attention + Device (2 columns)
        currentY = this.drawTwoColumnDropdowns(ctx, "Attention", this.attentionWidget, "Device", this.deviceWidget, margin, currentY);
        currentY += spacing;

        // 7. Keep VRAM + Longform Chunking (2 columns toggles)
        currentY = this.drawTwoColumnToggles(ctx, "Keep in VRAM", this.keepVramWidget, "Longform Chunking", this.chunkLongformWidget, margin, currentY);
        currentY += spacing * 2;

        // 8. Number Grid 2x3 (inline editing)
        currentY = this.drawNumberGrid(ctx, margin, currentY);
        currentY += spacing;

        // 9. Control After Generate
        currentY = this.drawDropdownField(ctx, "Control After Generate", this.controlAfterWidget, margin, currentY);
        currentY += spacing;

        // Debug mode widget hidden but still functional (controlled by backend)

        node.size[1] = Math.max(currentY + 20, 700);

        // Draw tooltip last so it appears on top of everything
        this.drawTooltip(ctx);

        // Draw lightbox on top of everything if open
        if (this.lightboxOpen) {
            this.drawLightbox(ctx);
        }
    }

    drawCharacterPresets(ctx, x, y) {
        const buttonWidth = (this.node.size[0] - x * 2 - 16) / 5;
        const buttonHeight = 32;
        const padding = 4;

        ctx.fillStyle = "#aaa";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.fillText("Character Presets:", x, y + 10);

        y += 18;

        for (let i = 0; i < this.characterPresets.length; i++) {
            const preset = this.characterPresets[i];
            const btnX = x + i * (buttonWidth + padding);
            const isHover = this.hoverElement === `preset${i}Btn`;
            const isClicked = this.clickedButtons[`preset${i}Btn`];

            const gradient = ctx.createLinearGradient(btnX, y, btnX, y + buttonHeight);
            if (isClicked) {
                gradient.addColorStop(0, "#2a2a2a");
                gradient.addColorStop(1, "#1a1a1a");
            } else if (isHover) {
                gradient.addColorStop(0, "#4a4a4a");
                gradient.addColorStop(1, "#3a3a3a");
            } else {
                gradient.addColorStop(0, "#3a3a3a");
                gradient.addColorStop(1, "#2a2a2a");
            }
            ctx.fillStyle = gradient;

            ctx.shadowColor = "rgba(0, 0, 0, 0.4)";
            ctx.shadowBlur = 3;
            ctx.shadowOffsetY = 2;

            ctx.beginPath();
            ctx.roundRect(btnX, y, buttonWidth, buttonHeight, 7);
            ctx.fill();

            ctx.shadowColor = "transparent";
            ctx.shadowBlur = 0;
            ctx.shadowOffsetY = 0;

            ctx.strokeStyle = isHover ? "#7788ff" : "#667eea";
            ctx.lineWidth = isHover ? 2 : 1.5;
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 12px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`${preset.emoji} ${preset.name}`, btnX + buttonWidth / 2, y + buttonHeight / 2);

            this.controls[`preset${i}Btn`] = { x: btnX, y, w: buttonWidth, h: buttonHeight, preset };
        }

        ctx.textAlign = "left";
        return y + buttonHeight + 10;
    }

    drawTextField(ctx, label, widget, x, y, height) {
        const width = this.node.size[0] - x * 2;
        const isHover = this.hoverElement === `${label}Field`;
        const isEditing = this.editingField === label;

        ctx.fillStyle = "#aaa";
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "left";
        // Move label up more for Text field specifically
        const labelOffset = label === "Text" ? 4 : 8;
        ctx.fillText(label + ":", x, y + labelOffset);

        y += 16;

        ctx.fillStyle = isEditing ? "#333" : (isHover ? "#2a2a2a" : "#252525");
        ctx.strokeStyle = isEditing ? "#667eea" : (isHover ? "#667eea" : "#444");
        ctx.lineWidth = isEditing ? 2 : 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 5);
        ctx.fill();
        ctx.stroke();

        const value = isEditing ? this.editingValue : (widget?.value || "");
        ctx.fillStyle = "#e0e0e0";
        ctx.font = "12px 'Courier New', monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";

        const padding = 8;
        const maxWidth = width - padding * 2;
        const lines = this.wrapText(ctx, value, maxWidth);

        // Get scroll offset for this field
        const scrollOffset = isEditing ? (this.scrollOffsets[label] || 0) : 0;

        // Setup clipping region to prevent text from drawing outside the box
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.clip();

        let textY = y + padding - scrollOffset;

        // Draw selection highlight first (behind text)
        if (isEditing && this.selectionStart !== null && this.selectionEnd !== null) {
            const start = Math.min(this.selectionStart, this.selectionEnd);
            const end = Math.max(this.selectionStart, this.selectionEnd);

            let currentPos = 0;
            let lineIdx = 0;

            for (const line of lines.slice(0, Math.floor((height - padding * 2) / 16))) {
                const lineStart = currentPos;
                const lineEnd = currentPos + line.length;

                if (end > lineStart && start < lineEnd) {
                    // This line has selection
                    const selStart = Math.max(0, start - lineStart);
                    const selEnd = Math.min(line.length, end - lineStart);

                    const beforeSel = line.substring(0, selStart);
                    const selected = line.substring(selStart, selEnd);

                    const selX = x + padding + ctx.measureText(beforeSel).width;
                    const selWidth = ctx.measureText(selected).width;

                    ctx.fillStyle = "#667eea80"; // Semi-transparent blue
                    ctx.fillRect(selX, textY, selWidth, 16);
                }

                currentPos += line.length + 1; // +1 for space between words
                textY += 16;
                lineIdx++;
            }

            // Reset for text drawing
            textY = y + padding;
            ctx.fillStyle = "#e0e0e0";
        }

        // Draw text
        for (const line of lines.slice(0, Math.floor((height - padding * 2) / 16))) {
            ctx.fillText(line, x + padding, textY);
            textY += 16;
        }

        // Cursor
        if (isEditing) {
            // Find which line the cursor is on
            let currentPos = 0;
            let cursorLine = 0;
            let cursorX = x + padding;
            let cursorY = y + padding;

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const lineEnd = currentPos + line.length;

                if (this.cursorPos <= lineEnd) {
                    // Cursor is on this line
                    cursorLine = i;
                    const posInLine = this.cursorPos - currentPos;
                    const beforeCursor = line.substring(0, posInLine);
                    cursorX = x + padding + ctx.measureText(beforeCursor).width;
                    cursorY = y + padding + i * 16;
                    break;
                }

                currentPos += line.length + 1; // +1 for space
            }

            ctx.fillStyle = "#667eea";
            ctx.fillRect(cursorX, cursorY, 2, 16);
        }

        if (!value && !isEditing) {
            ctx.fillStyle = "#666";
            ctx.fillText("Click to edit...", x + padding, y + padding - scrollOffset);
        }

        // Restore clipping region
        ctx.restore();

        // Draw scrollbar indicator if content overflows
        if (isEditing && lines.length * 16 > height - 16) {
            const totalHeight = lines.length * 16;
            const visibleHeight = height - 16;
            const scrollbarHeight = Math.max(20, (visibleHeight / totalHeight) * visibleHeight);
            const scrollbarY = y + 8 + (scrollOffset / (totalHeight - visibleHeight)) * (visibleHeight - scrollbarHeight);

            // Scrollbar track
            ctx.fillStyle = "rgba(100, 100, 100, 0.2)";
            ctx.fillRect(x + width - 6, y + 8, 4, visibleHeight);

            // Scrollbar thumb
            ctx.fillStyle = "#667eea";
            ctx.fillRect(x + width - 6, scrollbarY, 4, scrollbarHeight);
        }

        // Draw expand button at bottom right (only for Text field)
        if (label === "Text") {
            const expandBtnSize = 20;
            const expandBtnX = x + width - expandBtnSize - 4;
            const expandBtnY = y + height - expandBtnSize - 4;
            const expandBtnHover = this.hoverElement === `${label}ExpandBtn`;

            // Rounded square with transparency
            ctx.fillStyle = expandBtnHover ? "rgba(102, 126, 234, 0.9)" : "rgba(100, 100, 100, 0.4)";
            ctx.beginPath();
            ctx.roundRect(expandBtnX, expandBtnY, expandBtnSize, expandBtnSize, 4);
            ctx.fill();

            ctx.fillStyle = "#fff";
            ctx.font = "16px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("⛶", expandBtnX + expandBtnSize / 2, expandBtnY + expandBtnSize / 2);

            this.controls[`${label}ExpandBtn`] = { x: expandBtnX, y: expandBtnY, w: expandBtnSize, h: expandBtnSize, label };
        }

        this.controls[`${label}Field`] = { x, y, w: width, h: height, widget, label, textX: x + padding, textY: y + padding };

        return y + height + 10;
    }

    drawTwoColumnDropdowns(ctx, label1, widget1, label2, widget2, x, y) {
        const width = this.node.size[0] - x * 2;
        const colWidth = (width - 8) / 2;
        const height = 32;

        // Left column
        this.drawDropdownInColumn(ctx, label1, widget1, x, y, colWidth, height);

        // Right column
        this.drawDropdownInColumn(ctx, label2, widget2, x + colWidth + 8, y, colWidth, height);

        return y + height + 10;
    }

    drawDropdownInColumn(ctx, label, widget, x, y, width, height) {
        const isHover = this.hoverElement === `${label}Dropdown`;

        ctx.fillStyle = isHover ? "#2a2a2a" : "#252525";
        ctx.strokeStyle = isHover ? "#667eea" : "#444";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 5);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = "#aaa";
        ctx.font = "10px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(label + ":", x + 6, y + 6);

        // Value
        ctx.fillStyle = "#c4b5fd";
        ctx.font = "bold 10px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        const val = String(widget?.value || "");
        ctx.fillText(val.length > 10 ? val.substring(0, 10) + "..." : val, x + width / 2, y + height - 6);

        // Arrow
        ctx.fillStyle = "#667eea";
        ctx.font = "8px Arial";
        ctx.textAlign = "right";
        ctx.fillText("▼", x + width - 6, y + height - 8);

        this.controls[`${label}Dropdown`] = { x, y, w: width, h: height, widget, label };
    }

    drawTwoColumnToggles(ctx, label1, widget1, label2, widget2, x, y) {
        const width = this.node.size[0] - x * 2;
        const colWidth = (width - 8) / 2;
        const height = 32;

        // Left column
        this.drawToggleInColumn(ctx, label1, widget1, x, y, colWidth, height);

        // Right column
        this.drawToggleInColumn(ctx, label2, widget2, x + colWidth + 8, y, colWidth, height);

        return y + height + 10;
    }

    drawToggleInColumn(ctx, label, widget, x, y, width, height) {
        const isHover = this.hoverElement === `${label}Toggle`;
        const isOn = widget?.value || false;

        ctx.fillStyle = isHover ? "#2a2a2a" : "#252525";
        ctx.strokeStyle = isHover ? "#667eea" : "#444";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 5);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#aaa";
        ctx.font = "10px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x + 6, y + height / 2);

        const switchWidth = 40;
        const switchHeight = 20;
        const switchX = x + width - switchWidth - 6;
        const switchY = y + (height - switchHeight) / 2;

        ctx.fillStyle = isOn ? "#4ade80" : "#555";
        ctx.beginPath();
        ctx.roundRect(switchX, switchY, switchWidth, switchHeight, 10);
        ctx.fill();

        ctx.fillStyle = "#fff";
        const knobX = isOn ? switchX + switchWidth - 18 : switchX + 2;
        ctx.beginPath();
        ctx.arc(knobX + 8, switchY + 10, 8, 0, Math.PI * 2);
        ctx.fill();

        this.controls[`${label}Toggle`] = { x, y, w: width, h: height, widget, label };
    }

    drawNumberGrid(ctx, x, y) {
        const width = this.node.size[0] - x * 2;
        const fieldWidth = (width - 8) / 2;
        const fieldHeight = 32;

        // 2x2 grid for main generation parameters
        const gridFields = [
            { label: "Max Tokens", widget: this.maxTokensWidget },
            { label: "Temperature", widget: this.temperatureWidget },
            { label: "Top P", widget: this.topPWidget },
            { label: "Rep Penalty", widget: this.repPenaltyWidget }
        ];

        let row = 0;
        let col = 0;

        for (const field of gridFields) {
            const fieldX = x + col * (fieldWidth + 8);
            const fieldY = y + row * (fieldHeight + 8);
            this.drawNumberField(ctx, field.label, field.widget, fieldX, fieldY, fieldWidth, fieldHeight);

            col++;
            if (col >= 2) {
                col = 0;
                row++;
            }
        }

        // Seed takes full width on its own line
        const seedY = y + row * (fieldHeight + 8) + fieldHeight + 8;
        this.drawNumberField(ctx, "Seed", this.seedWidget, x, seedY, width, fieldHeight);

        return seedY + fieldHeight + 10;
    }

    drawNumberField(ctx, label, widget, x, y, width, height) {
        const isHover = this.hoverElement === `${label}Field`;
        const isEditing = this.editingField === label;

        ctx.fillStyle = isEditing ? "#333" : (isHover ? "#2a2a2a" : "#252525");
        ctx.strokeStyle = isEditing ? "#667eea" : (isHover ? "#667eea" : "#444");
        ctx.lineWidth = isEditing ? 2 : 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 5);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = "#aaa";
        ctx.font = "10px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(label, x + 6, y + 6);

        // Value
        const value = isEditing ? this.editingValue : String(widget?.value || 0);
        ctx.fillStyle = isEditing ? "#fff" : "#6ee7b7";
        ctx.font = "bold 13px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText(value, x + width / 2, y + height - 6);

        // Cursor
        if (isEditing) {
            const cursorX = x + width / 2 + ctx.measureText(value.substring(0, this.cursorPos)).width / 2;
            ctx.fillStyle = "#667eea";
            ctx.fillRect(cursorX, y + height - 20, 2, 14);
        }

        this.controls[`${label}Field`] = { x, y, w: width, h: height, widget, label, isNumber: true };
    }

    drawDropdownField(ctx, label, widget, x, y) {
        const width = this.node.size[0] - x * 2;
        const height = 32;
        const isHover = this.hoverElement === `${label}Dropdown`;

        ctx.fillStyle = isHover ? "#2a2a2a" : "#252525";
        ctx.strokeStyle = isHover ? "#667eea" : "#444";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 5);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#aaa";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label + ":", x + 8, y + height / 2);

        ctx.fillStyle = "#c4b5fd";
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "right";
        ctx.fillText(String(widget?.value || ""), x + width - 24, y + height / 2);

        ctx.fillStyle = "#667eea";
        ctx.font = "10px Arial";
        ctx.fillText("▼", x + width - 12, y + height / 2);

        this.controls[`${label}Dropdown`] = { x, y, w: width, h: height, widget, label };

        return y + height + 10;
    }

    drawToggleField(ctx, label, widget, x, y) {
        const width = this.node.size[0] - x * 2;
        const height = 32;
        const isHover = this.hoverElement === `${label}Toggle`;
        const isOn = widget?.value || false;

        ctx.fillStyle = isHover ? "#2a2a2a" : "#252525";
        ctx.strokeStyle = isHover ? "#667eea" : "#444";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, height, 5);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#aaa";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(label, x + 8, y + height / 2);

        const switchWidth = 50;
        const switchHeight = 24;
        const switchX = x + width - switchWidth - 8;
        const switchY = y + (height - switchHeight) / 2;

        ctx.fillStyle = isOn ? "#4ade80" : "#555";
        ctx.beginPath();
        ctx.roundRect(switchX, switchY, switchWidth, switchHeight, 12);
        ctx.fill();

        ctx.fillStyle = "#fff";
        const knobX = isOn ? switchX + switchWidth - 22 : switchX + 2;
        ctx.beginPath();
        ctx.arc(knobX + 10, switchY + 12, 10, 0, Math.PI * 2);
        ctx.fill();

        this.controls[`${label}Toggle`] = { x, y, w: width, h: height, widget, label };

        return y + height + 10;
    }

    drawEmotionSection(ctx, x, y) {
        const width = this.node.size[0] - x * 2;
        const headerHeight = 32;
        const isHover = this.hoverElement === "emotionHeader";

        ctx.fillStyle = isHover ? "#2a2a2a" : "#252525";
        ctx.strokeStyle = isHover ? "#667eea" : "#444";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(x, y, width, headerHeight, 5);
        ctx.fill();
        ctx.stroke();

        const arrow = this.emotionsCollapsed ? "▶" : "▼";
        ctx.fillStyle = "#aaa";
        ctx.font = "bold 11px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(`${arrow} Emotion Tags`, x + 8, y + headerHeight / 2);

        this.controls["emotionHeader"] = { x, y, w: width, h: headerHeight };

        y += headerHeight + 8;

        if (!this.emotionsCollapsed) {
            y = this.drawEmotionTags(ctx, x, y);
        }

        return y;
    }

    drawEmotionTags(ctx, x, y) {
        const buttonWidth = (this.node.size[0] - x * 2 - 9) / 4;
        const buttonHeight = 28;
        const padding = 3;

        let row = 0;
        let col = 0;

        for (let i = 0; i < this.emotionTags.length; i++) {
            const emotion = this.emotionTags[i];
            const btnX = x + col * (buttonWidth + padding);
            const btnY = y + row * (buttonHeight + padding);
            const isHover = this.hoverElement === `emotion${i}Btn`;
            const isClicked = this.clickedButtons[`emotion${i}Btn`];

            const gradient = ctx.createLinearGradient(btnX, btnY, btnX, btnY + buttonHeight);
            if (isClicked) {
                gradient.addColorStop(0, emotion.color + "99");
                gradient.addColorStop(0.5, emotion.color + "77");
                gradient.addColorStop(1, emotion.color + "55");
            } else if (isHover) {
                gradient.addColorStop(0, emotion.color + "ff");
                gradient.addColorStop(0.5, emotion.color + "cc");
                gradient.addColorStop(1, emotion.color + "99");
            } else {
                gradient.addColorStop(0, emotion.color + "ee");
                gradient.addColorStop(0.5, emotion.color + "cc");
                gradient.addColorStop(1, emotion.color + "99");
            }
            ctx.fillStyle = gradient;

            ctx.shadowColor = "rgba(0, 0, 0, 0.3)";
            ctx.shadowBlur = 2;
            ctx.shadowOffsetY = 1;

            ctx.beginPath();
            ctx.roundRect(btnX, btnY, buttonWidth, buttonHeight, 5);
            ctx.fill();

            ctx.shadowColor = "transparent";
            ctx.shadowBlur = 0;
            ctx.shadowOffsetY = 0;

            ctx.strokeStyle = isHover ? emotion.color + "ff" : emotion.color + "dd";
            ctx.lineWidth = isHover ? 2 : 1.5;
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 10px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(emotion.display, btnX + buttonWidth / 2, btnY + buttonHeight / 2);

            this.controls[`emotion${i}Btn`] = { x: btnX, y: btnY, w: buttonWidth, h: buttonHeight, emotion };

            col++;
            if (col >= 4) {
                col = 0;
                row++;
            }
        }

        return y + (row + 1) * (buttonHeight + padding);
    }

    handleMouseDown(e, pos, canvas) {
        const node = this.node;
        const relX = e.canvasX - node.pos[0];
        const relY = e.canvasY - node.pos[1];

        // Handle lightbox interactions first
        if (this.lightboxOpen) {
            for (const key in this.controls) {
                const ctrl = this.controls[key];
                if (this.isPointInControl(relX, relY, ctrl)) {
                    if (key === 'lightboxSaveBtn') {
                        // Trigger click animation
                        this.clickedButtons[key] = true;
                        setTimeout(() => {
                            delete this.clickedButtons[key];
                            this.closeLightbox(true);
                        }, 150);
                        node.setDirtyCanvas(true, true);
                        return true;
                    } else if (key === 'lightboxCancelBtn') {
                        // Trigger click animation
                        this.clickedButtons[key] = true;
                        setTimeout(() => {
                            delete this.clickedButtons[key];
                            this.closeLightbox(false);
                        }, 150);
                        node.setDirtyCanvas(true, true);
                        return true;
                    } else if (key.startsWith('lightboxEmotion')) {
                        // Trigger click animation
                        this.clickedButtons[key] = true;
                        setTimeout(() => {
                            delete this.clickedButtons[key];
                            node.setDirtyCanvas(true, true);
                        }, 150);

                        // Insert emotion tag at cursor
                        const tagWithSpace = ctrl.emotion.tag + " ";
                        this.editingValue = this.editingValue.slice(0, this.cursorPos) + tagWithSpace + this.editingValue.slice(this.cursorPos);
                        this.cursorPos += tagWithSpace.length;
                        node.setDirtyCanvas(true, true);
                        return true;
                    } else if (key === 'lightboxTextField') {
                        // Handle text field click for cursor positioning and drag selection
                        const scrollOffset = this.lightboxScrollOffset;
                        const clickX = relX - ctrl.textX;
                        const clickY = relY - ctrl.textY + scrollOffset;

                        const cursorPos = this.calculateCursorPosition(clickX, clickY, this.editingValue, 560);

                        this.isDragging = true;
                        this.dragStartPos = cursorPos;
                        this.cursorPos = cursorPos;
                        this.selectionStart = cursorPos;
                        this.selectionEnd = cursorPos;

                        // Add document-level mouseup listener
                        if (!this.mouseUpHandler) {
                            this.mouseUpHandler = () => {
                                this.isDragging = false;
                                this.dragStartPos = null;
                                if (this.mouseUpHandler) {
                                    document.removeEventListener('mouseup', this.mouseUpHandler, true);
                                    this.mouseUpHandler = null;
                                }
                                node.setDirtyCanvas(true, true);
                            };
                            document.addEventListener('mouseup', this.mouseUpHandler, true);
                        }

                        node.setDirtyCanvas(true, true);
                        return true;
                    }
                }
            }
            // Click outside lightbox modal - do nothing (keep lightbox open)
            return true;
        }

        for (const key in this.controls) {
            const ctrl = this.controls[key];
            if (this.isPointInControl(relX, relY, ctrl)) {
                // Check ExpandBtn BEFORE general Btn to prevent it being caught by Btn handler
                if (key.endsWith('ExpandBtn')) {
                    // Open lightbox for this field
                    const fieldLabel = ctrl.label;
                    const fieldCtrl = this.controls[`${fieldLabel}Field`];
                    if (fieldCtrl && fieldCtrl.widget) {
                        // Stop any current editing
                        if (this.editingField) {
                            this.stopEditing(true);
                        }

                        // Open lightbox
                        this.lightboxOpen = true;
                        this.lightboxField = fieldLabel;
                        this.editingValue = String(fieldCtrl.widget.value || "");
                        this.cursorPos = this.editingValue.length;
                        this.selectionStart = null;
                        this.selectionEnd = null;
                        this.lightboxScrollOffset = 0;

                        // Attach keyboard handlers
                        this.keydownHandler = (e) => this.handleKeyDown(e);
                        document.addEventListener('keydown', this.keydownHandler, true);

                        this.wheelHandler = (e) => this.handleLightboxWheel(e);
                        document.addEventListener('wheel', this.wheelHandler, { passive: false, capture: true });

                        node.setDirtyCanvas(true, true);
                    }
                    return true;
                } else if (key.endsWith('Btn')) {
                    // General button handler (for preset and emotion buttons)
                    // Trigger click animation
                    this.clickedButtons[key] = true;
                    setTimeout(() => {
                        delete this.clickedButtons[key];
                        node.setDirtyCanvas(true, true);
                    }, 150);

                    if (key.startsWith('preset')) {
                        if (this.voiceWidget && ctrl.preset) {
                            this.voiceWidget.value = ctrl.preset.description;
                            node.setDirtyCanvas(true, true);
                        }
                    } else if (key.startsWith('emotion')) {
                        if (this.textWidget && ctrl.emotion) {
                            // Insert at cursor position if text field is being edited
                            if (this.editingField === "Text") {
                                const tagWithSpace = ctrl.emotion.tag + " ";
                                this.editingValue = this.editingValue.slice(0, this.cursorPos) + tagWithSpace + this.editingValue.slice(this.cursorPos);
                                this.cursorPos += tagWithSpace.length;
                            } else {
                                // Append to end if not editing
                                this.textWidget.value += (this.textWidget.value ? " " : "") + ctrl.emotion.tag + " ";
                            }
                            node.setDirtyCanvas(true, true);
                        }
                    }
                    return true;
                } else if (key.endsWith('Field')) {
                    // If already editing this field, start drag selection
                    if (this.editingField === ctrl.label) {
                        const scrollOffset = this.scrollOffsets[ctrl.label] || 0;
                        const clickX = relX - ctrl.textX;
                        const clickY = relY - ctrl.textY + scrollOffset;

                        const cursorPos = this.calculateCursorPosition(clickX, clickY, this.editingValue, ctrl.w - 16);

                        this.isDragging = true;
                        this.dragStartPos = cursorPos;
                        this.cursorPos = cursorPos;
                        this.selectionStart = cursorPos;
                        this.selectionEnd = cursorPos;

                        // Add document-level mouseup listener to catch mouseup anywhere
                        if (!this.mouseUpHandler) {
                            this.mouseUpHandler = () => {
                                this.isDragging = false;
                                this.dragStartPos = null;
                                if (this.mouseUpHandler) {
                                    document.removeEventListener('mouseup', this.mouseUpHandler, true);
                                    this.mouseUpHandler = null;
                                }
                                node.setDirtyCanvas(true, true);
                            };
                            document.addEventListener('mouseup', this.mouseUpHandler, true);
                        }

                        node.setDirtyCanvas(true, true);
                        return true;
                    }

                    // Get initial value
                    const initialValue = String(ctrl.widget?.value || "");

                    // Calculate cursor position from click BEFORE starting editing
                    const cursorPos = this.calculateCursorPosition(
                        relX - ctrl.textX,
                        relY - ctrl.textY,
                        initialValue,
                        ctrl.w - 16
                    );

                    // Start editing with calculated cursor position
                    this.startEditing(ctrl.label, initialValue, ctrl);
                    this.cursorPos = cursorPos;

                    return true;
                } else if (key.endsWith('Dropdown')) {
                    const values = ctrl.widget?.options?.values || [];
                    if (values.length > 0) {
                        const menu = new LiteGraph.ContextMenu(values, {
                            event: e,
                            callback: (v) => {
                                if (ctrl.widget) {
                                    ctrl.widget.value = v;
                                    node.setDirtyCanvas(true, true);
                                }
                            },
                            node: node
                        });
                    }
                    return true;
                } else if (key.endsWith('Toggle')) {
                    if (ctrl.widget) {
                        ctrl.widget.value = !ctrl.widget.value;
                        node.setDirtyCanvas(true, true);
                    }
                    return true;
                } else if (key === "emotionHeader") {
                    this.emotionsCollapsed = !this.emotionsCollapsed;
                    node.setDirtyCanvas(true, true);
                    return true;
                }
            }
        }

        // Click outside - save and stop editing
        if (this.editingField) {
            this.stopEditing(true);
        }

        return false;
    }

    handleKeyDown(e) {
        if (!this.editingField && !this.lightboxOpen) return;

        // CRITICAL: Prevent ALL default behavior and stop propagation
        // This prevents ComfyUI shortcuts from being triggered
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // Handle Ctrl/Cmd shortcuts
        const isCtrl = e.ctrlKey || e.metaKey;

        if (isCtrl && e.key === 'a') {
            // Select all
            this.selectionStart = 0;
            this.selectionEnd = this.editingValue.length;
            this.cursorPos = this.editingValue.length;
            this.node.setDirtyCanvas(true, true);
            return;
        } else if (isCtrl && e.key === 'c') {
            // Copy
            if (this.selectionStart !== null && this.selectionEnd !== null) {
                const selectedText = this.editingValue.substring(
                    Math.min(this.selectionStart, this.selectionEnd),
                    Math.max(this.selectionStart, this.selectionEnd)
                );
                navigator.clipboard.writeText(selectedText);
            }
            return;
        } else if (isCtrl && e.key === 'v') {
            // Paste
            navigator.clipboard.readText().then(text => {
                if (this.selectionStart !== null && this.selectionEnd !== null) {
                    // Replace selection
                    const start = Math.min(this.selectionStart, this.selectionEnd);
                    const end = Math.max(this.selectionStart, this.selectionEnd);
                    this.editingValue = this.editingValue.substring(0, start) + text + this.editingValue.substring(end);
                    this.cursorPos = start + text.length;
                    this.selectionStart = null;
                    this.selectionEnd = null;
                } else {
                    // Insert at cursor
                    this.editingValue = this.editingValue.slice(0, this.cursorPos) + text + this.editingValue.slice(this.cursorPos);
                    this.cursorPos += text.length;
                }
                this.node.setDirtyCanvas(true, true);
            });
            return;
        } else if (isCtrl && e.key === 'x') {
            // Cut
            if (this.selectionStart !== null && this.selectionEnd !== null) {
                const start = Math.min(this.selectionStart, this.selectionEnd);
                const end = Math.max(this.selectionStart, this.selectionEnd);
                const selectedText = this.editingValue.substring(start, end);
                navigator.clipboard.writeText(selectedText);
                this.editingValue = this.editingValue.substring(0, start) + this.editingValue.substring(end);
                this.cursorPos = start;
                this.selectionStart = null;
                this.selectionEnd = null;
                this.node.setDirtyCanvas(true, true);
            }
            return;
        }

        // Handle selection deletion for Backspace and Delete
        if ((e.key === "Backspace" || e.key === "Delete") &&
            this.selectionStart !== null && this.selectionEnd !== null) {
            // Delete selection
            const start = Math.min(this.selectionStart, this.selectionEnd);
            const end = Math.max(this.selectionStart, this.selectionEnd);
            this.editingValue = this.editingValue.substring(0, start) + this.editingValue.substring(end);
            this.cursorPos = start;
            this.selectionStart = null;
            this.selectionEnd = null;
            this.node.setDirtyCanvas(true, true);
            return;
        }

        // Delete selection before inserting new character
        if (this.selectionStart !== null && this.selectionEnd !== null &&
            e.key.length === 1 && !isCtrl) {
            const start = Math.min(this.selectionStart, this.selectionEnd);
            const end = Math.max(this.selectionStart, this.selectionEnd);
            this.editingValue = this.editingValue.substring(0, start) + this.editingValue.substring(end);
            this.cursorPos = start;
            this.selectionStart = null;
            this.selectionEnd = null;
        }

        // Clear selection for arrow keys and other navigation
        if ((e.key === "ArrowLeft" || e.key === "ArrowRight" || e.key === "Home" || e.key === "End") &&
            this.selectionStart !== null && this.selectionEnd !== null) {
            this.selectionStart = null;
            this.selectionEnd = null;
        }

        if (e.key === "Enter") {
            if (this.lightboxOpen) {
                // In lightbox, Enter saves and closes
                this.closeLightbox(true);
            } else if (isCtrl) {
                // Ctrl+Enter saves for all fields
                this.stopEditing(true);
            } else if (this.editingField === "Text" || this.editingField === "Voice Description") {
                // For multiline text fields, Enter inserts newline
                if (this.selectionStart !== null && this.selectionEnd !== null) {
                    // Delete selection first
                    const start = Math.min(this.selectionStart, this.selectionEnd);
                    const end = Math.max(this.selectionStart, this.selectionEnd);
                    this.editingValue = this.editingValue.substring(0, start) + this.editingValue.substring(end);
                    this.cursorPos = start;
                    this.selectionStart = null;
                    this.selectionEnd = null;
                }
                // Insert newline
                this.editingValue = this.editingValue.slice(0, this.cursorPos) + "\n" + this.editingValue.slice(this.cursorPos);
                this.cursorPos++;
                this.node.setDirtyCanvas(true, true);
            } else {
                // For number fields, Enter saves
                this.stopEditing(true);
            }
            return;
        } else if (e.key === "Escape") {
            // Cancel editing (or close lightbox)
            if (this.lightboxOpen) {
                this.closeLightbox(false);
            } else {
                this.stopEditing(false);
            }
            return;
        } else if (e.key === "Backspace") {
            if (this.cursorPos > 0) {
                this.editingValue = this.editingValue.slice(0, this.cursorPos - 1) + this.editingValue.slice(this.cursorPos);
                this.cursorPos--;
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "Delete") {
            if (this.cursorPos < this.editingValue.length) {
                this.editingValue = this.editingValue.slice(0, this.cursorPos) + this.editingValue.slice(this.cursorPos + 1);
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "ArrowLeft") {
            if (this.cursorPos > 0) {
                this.cursorPos--;
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "ArrowRight") {
            if (this.cursorPos < this.editingValue.length) {
                this.cursorPos++;
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "Home") {
            this.cursorPos = 0;
            this.node.setDirtyCanvas(true, true);
        } else if (e.key === "End") {
            this.cursorPos = this.editingValue.length;
            this.node.setDirtyCanvas(true, true);
        } else if (e.key.length === 1 && !isCtrl) {
            // Only handle printable characters (not Ctrl+C, etc.)
            this.editingValue = this.editingValue.slice(0, this.cursorPos) + e.key + this.editingValue.slice(this.cursorPos);
            this.cursorPos++;
            this.node.setDirtyCanvas(true, true);
        }
    }

    handleMouseHover(e, pos, canvas) {
        const node = this.node;
        const relX = e.canvasX - node.pos[0];
        const relY = e.canvasY - node.pos[1];

        let foundHover = null;

        for (const key in this.controls) {
            if (this.isPointInControl(relX, relY, this.controls[key])) {
                foundHover = key;
                break;
            }
        }

        if (this.hoverElement !== foundHover) {
            this.hoverElement = foundHover;
            node.setDirtyCanvas(true, true);
        }
    }

    isPointInControl(x, y, ctrl) {
        return x >= ctrl.x && x <= ctrl.x + ctrl.w && y >= ctrl.y && y <= ctrl.y + ctrl.h;
    }

    wrapText(ctx, text, maxWidth) {
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';

        for (const word of words) {
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            if (ctx.measureText(testLine).width > maxWidth && currentLine) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        }
        if (currentLine) lines.push(currentLine);

        return lines;
    }

    getTooltipKey(controlKey) {
        // Map control keys to tooltip keys
        const mapping = {
            'ModelDropdown': 'model_name',
            'DtypeDropdown': 'dtype',
            'AttentionDropdown': 'attention_mechanism',
            'DeviceDropdown': 'device',
            'Voice DescriptionField': 'voice_description',
            'TextField': 'text',
            'Keep in VRAMToggle': 'keep_model_in_vram',
            'Longform ChunkingToggle': 'chunk_longform',
            'Max TokensField': 'max_tokens',
            'TemperatureField': 'temperature',
            'Top PField': 'top_p',
            'Rep PenaltyField': 'repetition_penalty',
            'SeedField': 'seed',
            'Control After GenerateDropdown': 'control_after_generate'
        };

        return mapping[controlKey] || null;
    }

    drawTooltip(ctx) {
        if (!this.hoverElement) return;

        // Don't show tooltips for headers, emotion tag buttons, or character preset buttons
        if (this.hoverElement === 'emotionHeader' ||
            this.hoverElement.match(/^emotion\d+Btn$/) ||
            this.hoverElement.match(/^preset\d+Btn$/)) {
            return;
        }

        const tooltipKey = this.getTooltipKey(this.hoverElement);
        if (!tooltipKey || !this.tooltips[tooltipKey]) return;

        // Get the control position
        const ctrl = this.controls[this.hoverElement];
        if (!ctrl) return;

        // Save canvas state to prevent affecting other elements
        ctx.save();

        const tooltipText = this.tooltips[tooltipKey];
        const padding = 10;
        const lineHeight = 14;
        const maxWidth = 300;

        // Split into lines
        const lines = tooltipText.split('\n');
        const wrappedLines = [];
        ctx.font = "11px Arial";

        for (const line of lines) {
            if (ctx.measureText(line).width > maxWidth) {
                const words = line.split(' ');
                let currentLine = '';
                for (const word of words) {
                    const testLine = currentLine + (currentLine ? ' ' : '') + word;
                    if (ctx.measureText(testLine).width > maxWidth && currentLine) {
                        wrappedLines.push(currentLine);
                        currentLine = word;
                    } else {
                        currentLine = testLine;
                    }
                }
                if (currentLine) wrappedLines.push(currentLine);
            } else {
                wrappedLines.push(line);
            }
        }

        const tooltipWidth = maxWidth + padding * 2;
        const tooltipHeight = wrappedLines.length * lineHeight + padding * 2;

        // Position tooltip - to the right of node, aligned with the control
        let tooltipX = this.node.size[0] + 10;
        let tooltipY = ctrl.y; // Align with the control's Y position

        // Draw tooltip background
        ctx.fillStyle = "rgba(20, 20, 20, 0.95)";
        ctx.strokeStyle = "#667eea";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 8);
        ctx.fill();
        ctx.stroke();

        // Draw tooltip text
        ctx.fillStyle = "#e0e0e0";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";

        let textY = tooltipY + padding;
        for (const line of wrappedLines) {
            ctx.fillText(line, tooltipX + padding, textY);
            textY += lineHeight;
        }

        // Restore canvas state
        ctx.restore();
    }

    calculateCursorPosition(clickX, clickY, text, maxWidth) {
        // Create a temporary canvas to measure with the correct font
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.font = "12px 'Courier New', monospace";

        // Wrap text into lines
        const lineHeight = 16;
        const lineIndex = Math.floor(clickY / lineHeight);

        // Split text into lines using same logic as rendering
        const lines = this.wrapText(tempCtx, text, maxWidth);

        if (lineIndex >= 0 && lineIndex < lines.length) {
            // Find character position in that line
            let charsBeforeLine = 0;
            for (let i = 0; i < lineIndex; i++) {
                charsBeforeLine += lines[i].length;
                // Add 1 for the space that was used to break the line
                if (i < lines.length - 1 && charsBeforeLine < text.length) {
                    charsBeforeLine += 1;
                }
            }

            const line = lines[lineIndex];
            let charPosInLine = 0;
            let totalWidth = 0;

            for (let i = 0; i <= line.length; i++) {
                if (i === line.length || totalWidth + tempCtx.measureText(line[i]).width / 2 > clickX) {
                    charPosInLine = i;
                    break;
                }
                totalWidth += tempCtx.measureText(line[i]).width;
            }

            return Math.min(charsBeforeLine + charPosInLine, text.length);
        } else {
            // Click below or above all lines
            return lineIndex < 0 ? 0 : text.length;
        }
    }

    handleMouseDrag(e, pos, canvas) {
        if (!this.isDragging) return false;

        const node = this.node;
        const relX = e.canvasX - node.pos[0];
        const relY = e.canvasY - node.pos[1];

        // Handle lightbox drag selection
        if (this.lightboxOpen) {
            const ctrl = this.controls['lightboxTextField'];
            if (ctrl) {
                const scrollOffset = this.lightboxScrollOffset;
                const clickX = relX - ctrl.textX;
                const clickY = relY - ctrl.textY + scrollOffset;

                const cursorPos = this.calculateCursorPosition(clickX, clickY, this.editingValue, 560);

                // Update selection
                this.cursorPos = cursorPos;
                this.selectionStart = this.dragStartPos;
                this.selectionEnd = cursorPos;

                node.setDirtyCanvas(true, true);
            }
            return true;
        }

        // Handle regular field drag selection
        if (!this.editingField) return false;

        // Find the text field control
        const ctrl = this.controls[`${this.editingField}Field`];
        if (!ctrl) return false;

        // Check if mouse is within the field bounds
        if (relX >= ctrl.x && relX <= ctrl.x + ctrl.w &&
            relY >= ctrl.y && relY <= ctrl.y + ctrl.h) {

            const scrollOffset = this.scrollOffsets[this.editingField] || 0;
            const clickX = relX - ctrl.textX;
            const clickY = relY - ctrl.textY + scrollOffset;

            const cursorPos = this.calculateCursorPosition(clickX, clickY, this.editingValue, ctrl.w - 16);

            // Update selection
            this.cursorPos = cursorPos;
            this.selectionStart = this.dragStartPos;
            this.selectionEnd = cursorPos;

            node.setDirtyCanvas(true, true);
        }

        return true;
    }

    handleWheel(e) {
        if (!this.editingField) return;

        const ctrl = this.controls[`${this.editingField}Field`];
        if (!ctrl) return;

        // CRITICAL: Prevent default FIRST, before any checks
        // This prevents ComfyUI zoom from triggering
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // Get canvas and convert coordinates
        const canvas = app.canvas.canvas;
        const rect = canvas.getBoundingClientRect();

        // Convert client to canvas coordinates accounting for zoom
        const canvasX = (e.clientX - rect.left) / (rect.width / canvas.width);
        const canvasY = (e.clientY - rect.top) / (rect.height / canvas.height);

        // Get node position in canvas space
        const node = this.node;
        const nodeRelX = canvasX - node.pos[0];
        const nodeRelY = canvasY - node.pos[1];

        // Check if mouse is within the text field bounds
        if (nodeRelX >= ctrl.x && nodeRelX <= ctrl.x + ctrl.w &&
            nodeRelY >= ctrl.y && nodeRelY <= ctrl.y + ctrl.h) {

            // Calculate total content height
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.font = "12px 'Courier New', monospace";
            const lines = this.wrapText(tempCtx, this.editingValue, ctrl.w - 16);
            const totalHeight = lines.length * 16;
            const visibleHeight = ctrl.h - 16; // subtract padding

            // Only scroll if content is larger than visible area
            if (totalHeight > visibleHeight) {
                const scrollAmount = e.deltaY;
                const currentOffset = this.scrollOffsets[this.editingField] || 0;
                const maxScroll = totalHeight - visibleHeight;

                this.scrollOffsets[this.editingField] = Math.max(0, Math.min(maxScroll, currentOffset + scrollAmount));

                this.node.setDirtyCanvas(true, true);
            }

            return false;
        }
    }

    handleLightboxWheel(e) {
        if (!this.lightboxOpen) return;

        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // Calculate total content height
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.font = "14px 'Courier New', monospace";
        const lines = this.wrapText(tempCtx, this.editingValue, 560);
        const totalHeight = lines.length * 18;
        const visibleHeight = 400;

        // Only scroll if content is larger than visible area
        if (totalHeight > visibleHeight) {
            const scrollAmount = e.deltaY;
            const maxScroll = totalHeight - visibleHeight;

            this.lightboxScrollOffset = Math.max(0, Math.min(maxScroll, this.lightboxScrollOffset + scrollAmount));

            this.node.setDirtyCanvas(true, true);
        }
    }

    closeLightbox(save = true) {
        if (!this.lightboxOpen) return;

        // Save the value if requested
        if (save && this.lightboxField) {
            const fieldCtrl = this.controls[`${this.lightboxField}Field`];
            if (fieldCtrl && fieldCtrl.widget) {
                fieldCtrl.widget.value = this.editingValue;
            }
        }

        // Remove event handlers
        if (this.keydownHandler) {
            document.removeEventListener('keydown', this.keydownHandler, true);
            this.keydownHandler = null;
        }

        if (this.wheelHandler) {
            document.removeEventListener('wheel', this.wheelHandler, { passive: false, capture: true });
            this.wheelHandler = null;
        }

        if (this.mouseUpHandler) {
            document.removeEventListener('mouseup', this.mouseUpHandler, true);
            this.mouseUpHandler = null;
        }

        this.lightboxOpen = false;
        this.lightboxField = null;
        this.editingValue = "";
        this.cursorPos = 0;
        this.selectionStart = null;
        this.selectionEnd = null;
        this.isDragging = false;
        this.dragStartPos = null;
        this.lightboxScrollOffset = 0;

        this.node.setDirtyCanvas(true, true);
    }

    drawLightbox(ctx) {
        // Draw dark overlay
        ctx.fillStyle = "rgba(0, 0, 0, 0.85)";
        ctx.fillRect(0, 0, this.node.size[0], this.node.size[1]);

        // Modal dimensions - increased height to fit all content with proper margin
        const modalWidth = 600;
        const modalHeight = 730;
        const modalX = (this.node.size[0] - modalWidth) / 2;
        const modalY = 40;

        // Modal background
        ctx.fillStyle = "#1a1a1a";
        ctx.strokeStyle = "#667eea";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect(modalX, modalY, modalWidth, modalHeight, 10);
        ctx.fill();
        ctx.stroke();

        // Title
        ctx.fillStyle = "#fff";
        ctx.font = "bold 16px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(`Edit ${this.lightboxField}`, modalX + 20, modalY + 18);

        // Text field
        const textX = modalX + 20;
        const textY = modalY + 60;
        const textWidth = modalWidth - 40;
        const textHeight = 420;

        ctx.fillStyle = "#252525";
        ctx.strokeStyle = "#667eea";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(textX, textY, textWidth, textHeight, 5);
        ctx.fill();
        ctx.stroke();

        // Draw text content with scroll
        ctx.save();
        ctx.beginPath();
        ctx.rect(textX, textY, textWidth, textHeight);
        ctx.clip();

        const padding = 10;
        const maxWidth = textWidth - padding * 2;
        ctx.font = "14px 'Courier New', monospace";
        const lines = this.wrapText(ctx, this.editingValue, maxWidth);

        let textDrawY = textY + padding - this.lightboxScrollOffset;

        // Draw selection highlight
        if (this.selectionStart !== null && this.selectionEnd !== null) {
            const start = Math.min(this.selectionStart, this.selectionEnd);
            const end = Math.max(this.selectionStart, this.selectionEnd);

            let currentPos = 0;
            for (const line of lines) {
                const lineStart = currentPos;
                const lineEnd = currentPos + line.length;

                if (end > lineStart && start < lineEnd) {
                    const selStart = Math.max(0, start - lineStart);
                    const selEnd = Math.min(line.length, end - lineStart);

                    const beforeSel = line.substring(0, selStart);
                    const selected = line.substring(selStart, selEnd);

                    const selX = textX + padding + ctx.measureText(beforeSel).width;
                    const selWidth = ctx.measureText(selected).width;

                    ctx.fillStyle = "#667eea80";
                    ctx.fillRect(selX, textDrawY, selWidth, 18);
                }

                currentPos += line.length + 1;
                textDrawY += 18;
            }

            textDrawY = textY + padding - this.lightboxScrollOffset;
            ctx.fillStyle = "#e0e0e0";
        }

        // Draw text
        ctx.fillStyle = "#e0e0e0";
        for (const line of lines) {
            ctx.fillText(line, textX + padding, textDrawY);
            textDrawY += 18;
        }

        // Draw cursor
        let currentPos = 0;
        let cursorDrawn = false;
        textDrawY = textY + padding - this.lightboxScrollOffset;

        for (const line of lines) {
            const lineEnd = currentPos + line.length;

            if (this.cursorPos <= lineEnd && !cursorDrawn) {
                const posInLine = this.cursorPos - currentPos;
                const beforeCursor = line.substring(0, posInLine);
                const cursorX = textX + padding + ctx.measureText(beforeCursor).width;

                ctx.fillStyle = "#667eea";
                ctx.fillRect(cursorX, textDrawY, 2, 18);
                cursorDrawn = true;
                break;
            }

            currentPos += line.length + 1;
            textDrawY += 18;
        }

        ctx.restore();

        // Draw scrollbar if needed
        if (lines.length * 18 > textHeight - 20) {
            const totalHeight = lines.length * 18;
            const visibleHeight = textHeight - 20;
            const scrollbarHeight = Math.max(30, (visibleHeight / totalHeight) * visibleHeight);
            const scrollbarY = textY + 10 + (this.lightboxScrollOffset / (totalHeight - visibleHeight)) * (visibleHeight - scrollbarHeight);

            ctx.fillStyle = "rgba(100, 100, 100, 0.3)";
            ctx.fillRect(textX + textWidth - 8, textY + 10, 6, visibleHeight);

            ctx.fillStyle = "#667eea";
            ctx.fillRect(textX + textWidth - 8, scrollbarY, 6, scrollbarHeight);
        }

        // Store text field bounds for click detection
        this.controls['lightboxTextField'] = { x: textX, y: textY, w: textWidth, h: textHeight, textX: textX + padding, textY: textY + padding };

        // Emotion tags
        const emotionY = textY + textHeight + 20;
        const buttonWidth = (textWidth - 9) / 4;
        const buttonHeight = 28;
        const btnPadding = 3;

        let row = 0;
        let col = 0;

        for (let i = 0; i < this.emotionTags.length; i++) {
            const emotion = this.emotionTags[i];
            const btnX = textX + col * (buttonWidth + btnPadding);
            const btnY = emotionY + row * (buttonHeight + btnPadding);
            const isHover = this.hoverElement === `lightboxEmotion${i}Btn`;
            const isClicked = this.clickedButtons[`lightboxEmotion${i}Btn`];

            const gradient = ctx.createLinearGradient(btnX, btnY, btnX, btnY + buttonHeight);
            if (isClicked) {
                // Darker when clicked
                gradient.addColorStop(0, emotion.color + "99");
                gradient.addColorStop(0.5, emotion.color + "77");
                gradient.addColorStop(1, emotion.color + "55");
            } else if (isHover) {
                gradient.addColorStop(0, emotion.color + "ff");
                gradient.addColorStop(0.5, emotion.color + "cc");
                gradient.addColorStop(1, emotion.color + "99");
            } else {
                gradient.addColorStop(0, emotion.color + "ee");
                gradient.addColorStop(0.5, emotion.color + "cc");
                gradient.addColorStop(1, emotion.color + "99");
            }
            ctx.fillStyle = gradient;

            ctx.beginPath();
            ctx.roundRect(btnX, btnY, buttonWidth, buttonHeight, 5);
            ctx.fill();

            ctx.strokeStyle = isHover ? emotion.color + "ff" : emotion.color + "dd";
            ctx.lineWidth = isHover ? 2 : 1.5;
            ctx.stroke();

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 10px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(emotion.display, btnX + buttonWidth / 2, btnY + buttonHeight / 2);

            this.controls[`lightboxEmotion${i}Btn`] = { x: btnX, y: btnY, w: buttonWidth, h: buttonHeight, emotion };

            col++;
            if (col >= 4) {
                col = 0;
                row++;
            }
        }

        // Save and Cancel buttons
        const buttonY = emotionY + (row + 1) * (buttonHeight + btnPadding) + 10;
        const btnWidth = 120;
        const btnHeight = 40;
        const btnSpacing = 20;

        // Save button
        const saveBtnX = modalX + (modalWidth - btnWidth * 2 - btnSpacing) / 2;
        const saveBtnHover = this.hoverElement === 'lightboxSaveBtn';
        const saveBtnClicked = this.clickedButtons['lightboxSaveBtn'];

        // Darker when clicked
        if (saveBtnClicked) {
            ctx.fillStyle = "#2a8a5a";
        } else if (saveBtnHover) {
            ctx.fillStyle = "#4ade80";
        } else {
            ctx.fillStyle = "#3aba6a";
        }

        ctx.beginPath();
        ctx.roundRect(saveBtnX, buttonY, btnWidth, btnHeight, 5);
        ctx.fill();

        ctx.fillStyle = "#fff";
        ctx.font = "bold 14px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Save", saveBtnX + btnWidth / 2, buttonY + btnHeight / 2);

        this.controls['lightboxSaveBtn'] = { x: saveBtnX, y: buttonY, w: btnWidth, h: btnHeight };

        // Cancel button
        const cancelBtnX = saveBtnX + btnWidth + btnSpacing;
        const cancelBtnHover = this.hoverElement === 'lightboxCancelBtn';
        const cancelBtnClicked = this.clickedButtons['lightboxCancelBtn'];

        // Darker when clicked
        if (cancelBtnClicked) {
            ctx.fillStyle = "#aa2222";
        } else if (cancelBtnHover) {
            ctx.fillStyle = "#ff5555";
        } else {
            ctx.fillStyle = "#cc4444";
        }

        ctx.beginPath();
        ctx.roundRect(cancelBtnX, buttonY, btnWidth, btnHeight, 5);
        ctx.fill();

        ctx.fillStyle = "#fff";
        ctx.fillText("Cancel", cancelBtnX + btnWidth / 2, buttonY + btnHeight / 2);

        this.controls['lightboxCancelBtn'] = { x: cancelBtnX, y: buttonY, w: btnWidth, h: btnHeight };
    }
}

// Register
app.registerExtension({
    name: "Maya1.TTS",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Maya1TTS_Combined") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                this.maya1Canvas = new Maya1TTSCanvas(this);
            };
        }
    }
});
