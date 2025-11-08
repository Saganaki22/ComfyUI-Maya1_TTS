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
        this.htmlModalOpen = false; // Track if HTML modal is open
        this.htmlModalElement = null; // Reference to HTML modal element
        this.htmlModalFontSize = 15; // Default font size for modal textarea
        this.cursorVisible = true; // Cursor blink state
        this.cursorBlinkInterval = null; // Interval for cursor blinking

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
            // If HTML modal is open, block ComfyUI from processing
            if (self.htmlModalOpen) return true;
            const result = self.handleMouseDown(e, pos, canvas);
            return result;
        };

        node.onMouseMove = function(e, pos, canvas) {
            // CRITICAL: Block ALL mouse events when HTML modal is open
            if (self.htmlModalOpen) {
                return true; // Block ComfyUI from processing
            }

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

        // Start cursor blinking
        this.startCursorBlink();

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

        // Stop cursor blinking
        this.stopCursorBlink();

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

    startCursorBlink() {
        // Stop any existing blink interval
        this.stopCursorBlink();

        // Start cursor visible
        this.cursorVisible = true;

        // Blink cursor every 530ms (standard blink rate)
        this.cursorBlinkInterval = setInterval(() => {
            this.cursorVisible = !this.cursorVisible;
            this.node.setDirtyCanvas(true, true);
        }, 530);
    }

    stopCursorBlink() {
        if (this.cursorBlinkInterval) {
            clearInterval(this.cursorBlinkInterval);
            this.cursorBlinkInterval = null;
        }
        this.cursorVisible = true; // Reset to visible
    }

    resetCursorBlink() {
        // Reset cursor to visible and restart blink cycle
        // Called when user types or moves cursor
        if (this.cursorBlinkInterval) {
            this.cursorVisible = true;
            this.stopCursorBlink();
            this.startCursorBlink();
        }
    }

    openHTMLModal(fieldLabel, initialValue, widget) {
        // Create HTML modal overlay
        const overlay = document.createElement('div');
        overlay.id = 'maya1-modal-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999999;
            animation: fadeIn 0.2s ease;
        `;

        // Create modal container
        const modal = document.createElement('div');
        modal.style.cssText = `
            background: #1a1a1a;
            border: 3px solid #667eea;
            border-radius: 12px;
            width: 90%;
            max-width: 700px;
            max-height: 85vh;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
            animation: slideIn 0.3s ease;
        `;

        // Header container
        const header = document.createElement('div');
        header.style.cssText = `
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 24px;
            border-bottom: 2px solid #667eea40;
        `;

        // Title
        const title = document.createElement('div');
        title.textContent = `Edit ${fieldLabel}`;
        title.style.cssText = `
            color: #fff;
            font-family: Arial, sans-serif;
            font-size: 18px;
            font-weight: bold;
        `;

        // Font size controls container
        const fontControls = document.createElement('div');
        fontControls.style.cssText = `
            display: flex;
            align-items: center;
            gap: 10px;
        `;

        // Small A
        const smallA = document.createElement('span');
        smallA.textContent = 'A';
        smallA.style.cssText = `
            color: #aaa;
            font-family: Arial, sans-serif;
            font-size: 12px;
            font-weight: bold;
            user-select: none;
            -webkit-user-select: none;
            cursor: default;
        `;

        // Font size slider
        const fontSlider = document.createElement('input');
        fontSlider.type = 'range';
        fontSlider.min = '12';
        fontSlider.max = '20';
        fontSlider.value = this.htmlModalFontSize.toString();
        fontSlider.style.cssText = `
            width: 120px;
            cursor: pointer;
        `;

        // Large A
        const largeA = document.createElement('span');
        largeA.textContent = 'A';
        largeA.style.cssText = `
            color: #fff;
            font-family: Arial, sans-serif;
            font-size: 18px;
            font-weight: bold;
            user-select: none;
            -webkit-user-select: none;
            cursor: default;
        `;

        fontControls.appendChild(smallA);
        fontControls.appendChild(fontSlider);
        fontControls.appendChild(largeA);

        header.appendChild(title);
        header.appendChild(fontControls);

        // Textarea
        const textarea = document.createElement('textarea');
        textarea.value = initialValue || '';
        textarea.style.cssText = `
            flex: 1;
            background: #252525;
            color: #e0e0e0;
            border: 2px solid #667eea;
            border-radius: 8px;
            margin: 20px 24px;
            padding: 16px;
            font-family: 'Courier New', monospace;
            font-size: ${this.htmlModalFontSize}px;
            line-height: 1.5;
            resize: none;
            outline: none;
            min-height: max(250px, 30vh);
            max-height: 40vh;
            overflow-y: auto;
            overflow-x: hidden;
            word-wrap: break-word;
            white-space: pre-wrap;
        `;
        textarea.addEventListener('focus', () => {
            textarea.style.borderColor = '#7788ff';
        });
        textarea.addEventListener('blur', () => {
            textarea.style.borderColor = '#667eea';
        });

        // Font slider event listener
        fontSlider.addEventListener('input', (e) => {
            const newSize = parseInt(e.target.value);
            this.htmlModalFontSize = newSize;
            textarea.style.fontSize = `${newSize}px`;
        });

        // Ctrl+Enter to save
        textarea.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                e.stopPropagation();
                if (textarea.value.trim() === '') {
                    this.showErrorNotification('No text to save');
                    this.closeHTMLModal();
                } else {
                    widget.value = textarea.value;
                    this.showSaveNotification();
                    this.closeHTMLModal();
                }
            }
        });

        // Emotion tags container
        const emotionContainer = document.createElement('div');
        emotionContainer.style.cssText = `
            padding: 0 24px 16px 24px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        `;

        // Add emotion tag buttons
        this.emotionTags.forEach(emotion => {
            const btn = document.createElement('button');
            btn.textContent = emotion.display;
            btn.style.cssText = `
                background: linear-gradient(180deg, ${emotion.color}ee, ${emotion.color}99);
                border: 2px solid ${emotion.color}dd;
                border-radius: 6px;
                color: #fff;
                font-family: Arial, sans-serif;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                cursor: pointer;
                transition: all 0.15s ease;
                box-shadow: 0 0 0 0 ${emotion.color}00;
            `;
            btn.addEventListener('mouseenter', () => {
                btn.style.background = `linear-gradient(180deg, ${emotion.color}ff, ${emotion.color}cc)`;
                btn.style.boxShadow = `0 0 0 1px ${emotion.color}ff`;
            });
            btn.addEventListener('mouseleave', () => {
                btn.style.background = `linear-gradient(180deg, ${emotion.color}ee, ${emotion.color}99)`;
                btn.style.boxShadow = `0 0 0 0 ${emotion.color}00`;
            });
            btn.addEventListener('click', () => {
                const cursorPos = textarea.selectionStart;
                const textBefore = textarea.value.substring(0, cursorPos);
                const textAfter = textarea.value.substring(cursorPos);
                textarea.value = textBefore + emotion.tag + ' ' + textAfter;
                textarea.focus();
                const newPos = cursorPos + emotion.tag.length + 1;
                textarea.setSelectionRange(newPos, newPos);
            });
            emotionContainer.appendChild(btn);
        });

        // Buttons container
        const btnContainer = document.createElement('div');
        btnContainer.style.cssText = `
            padding: 0 24px 24px 24px;
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
        `;

        // Save button with hint inside
        const saveBtn = document.createElement('button');
        saveBtn.style.cssText = `
            background: #3aba6a;
            border: none;
            border-radius: 8px;
            color: #fff;
            font-family: Arial, sans-serif;
            padding: 12px 48px;
            cursor: pointer;
            transition: background 0.15s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        `;

        const saveLabel = document.createElement('div');
        saveLabel.textContent = 'Save';
        saveLabel.style.cssText = `
            font-size: 16px;
            font-weight: bold;
        `;

        const saveHint = document.createElement('div');
        saveHint.textContent = 'Ctrl + Enter';
        saveHint.style.cssText = `
            color: rgba(255, 255, 255, 0.5);
            font-size: 10px;
            user-select: none;
            -webkit-user-select: none;
        `;

        saveBtn.appendChild(saveLabel);
        saveBtn.appendChild(saveHint);

        saveBtn.addEventListener('mouseenter', () => saveBtn.style.background = '#4ade80');
        saveBtn.addEventListener('mouseleave', () => saveBtn.style.background = '#3aba6a');
        saveBtn.addEventListener('click', () => {
            if (textarea.value.trim() === '') {
                this.showErrorNotification('No text to save');
                this.closeHTMLModal();
            } else {
                widget.value = textarea.value;
                this.showSaveNotification();
                this.closeHTMLModal();
            }
        });

        // Cancel button with hint inside
        const cancelBtn = document.createElement('button');
        cancelBtn.style.cssText = `
            background: #cc4444;
            border: none;
            border-radius: 8px;
            color: #fff;
            font-family: Arial, sans-serif;
            padding: 12px 48px;
            cursor: pointer;
            transition: background 0.15s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        `;

        const cancelLabel = document.createElement('div');
        cancelLabel.textContent = 'Cancel';
        cancelLabel.style.cssText = `
            font-size: 16px;
            font-weight: bold;
        `;

        const cancelHint = document.createElement('div');
        cancelHint.textContent = 'ESC';
        cancelHint.style.cssText = `
            color: rgba(255, 255, 255, 0.5);
            font-size: 10px;
            user-select: none;
            -webkit-user-select: none;
        `;

        cancelBtn.appendChild(cancelLabel);
        cancelBtn.appendChild(cancelHint);

        cancelBtn.addEventListener('mouseenter', () => cancelBtn.style.background = '#ff5555');
        cancelBtn.addEventListener('mouseleave', () => cancelBtn.style.background = '#cc4444');
        cancelBtn.addEventListener('click', () => this.closeHTMLModal());

        btnContainer.appendChild(saveBtn);
        btnContainer.appendChild(cancelBtn);

        // Assemble modal
        modal.appendChild(header);
        modal.appendChild(textarea);
        modal.appendChild(emotionContainer);
        modal.appendChild(btnContainer);
        overlay.appendChild(modal);

        // Add CSS animations and custom selection color (only once)
        if (!document.getElementById('maya1-modal-animations')) {
            const style = document.createElement('style');
            style.id = 'maya1-modal-animations';
            style.textContent = `
                @keyframes fadeIn {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                @keyframes fadeOut {
                    from { opacity: 1; }
                    to { opacity: 0; }
                }
                @keyframes slideIn {
                    from { transform: translateY(-30px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }

                /* Custom selection color to match theme */
                #maya1-modal-overlay textarea::selection {
                    background: #667eea80;
                    color: #ffffff;
                }
                #maya1-modal-overlay textarea::-moz-selection {
                    background: #667eea80;
                    color: #ffffff;
                }

                /* Custom scrollbar for textarea */
                #maya1-modal-overlay textarea::-webkit-scrollbar {
                    width: 12px;
                }
                #maya1-modal-overlay textarea::-webkit-scrollbar-track {
                    background: rgba(100, 100, 100, 0.2);
                    border-radius: 6px;
                }
                #maya1-modal-overlay textarea::-webkit-scrollbar-thumb {
                    background: #667eea;
                    border-radius: 6px;
                }
                #maya1-modal-overlay textarea::-webkit-scrollbar-thumb:hover {
                    background: #7788ff;
                }
            `;
            document.head.appendChild(style);
        }

        // Handle ESC key
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                this.closeHTMLModal();
            }
        };
        document.addEventListener('keydown', escHandler);

        // Click outside to close
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                this.closeHTMLModal();
            }
        });

        // Prevent clicks inside modal from closing
        modal.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // Store references
        this.htmlModalElement = overlay;
        this.htmlModalEscHandler = escHandler;
        this.htmlModalOpen = true;

        // Add to DOM and focus textarea
        document.body.appendChild(overlay);
        setTimeout(() => textarea.focus(), 100);
    }

    closeHTMLModal() {
        if (this.htmlModalElement) {
            const element = this.htmlModalElement;

            // Prevent any interaction during close
            element.style.pointerEvents = 'none';

            // Listen for animation end to remove element at exact right time
            const handleAnimationEnd = () => {
                if (element.parentNode) {
                    element.parentNode.removeChild(element);
                }
                this.htmlModalElement = null;
            };

            element.addEventListener('animationend', handleAnimationEnd, { once: true });

            // Start fade out animation
            element.style.animation = 'fadeOut 0.2s ease forwards';

            // Fallback timeout in case animationend doesn't fire
            setTimeout(() => {
                if (element.parentNode) {
                    element.parentNode.removeChild(element);
                }
                this.htmlModalElement = null;
            }, 250);
        }

        if (this.htmlModalEscHandler) {
            document.removeEventListener('keydown', this.htmlModalEscHandler);
            this.htmlModalEscHandler = null;
        }

        this.htmlModalOpen = false;
        this.node.setDirtyCanvas(true, true);
    }

    showSaveNotification() {
        // Create toast notification
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(-20px);
            background: #3aba6a;
            color: #fff;
            padding: 12px 24px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 9999999;
            display: flex;
            align-items: center;
            gap: 8px;
            opacity: 0;
            transition: all 0.3s ease;
        `;

        // Checkmark
        const checkmark = document.createElement('span');
        checkmark.textContent = '✓';
        checkmark.style.cssText = `
            font-size: 18px;
            font-weight: bold;
        `;

        const message = document.createElement('span');
        message.textContent = 'Text Saved';

        toast.appendChild(checkmark);
        toast.appendChild(message);
        document.body.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(-50%) translateY(0)';
        }, 10);

        // Animate out and remove
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(-20px)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 2000);
    }

    showErrorNotification(errorMessage) {
        // Create error toast notification
        const toast = document.createElement('div');
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%) translateY(-20px);
            background: #cc4444;
            color: #fff;
            padding: 12px 24px;
            border-radius: 8px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            z-index: 9999999;
            display: flex;
            align-items: center;
            gap: 8px;
            opacity: 0;
            transition: all 0.3s ease;
        `;

        // X mark
        const xmark = document.createElement('span');
        xmark.textContent = '✕';
        xmark.style.cssText = `
            font-size: 18px;
            font-weight: bold;
        `;

        const message = document.createElement('span');
        message.textContent = errorMessage;

        toast.appendChild(xmark);
        toast.appendChild(message);
        document.body.appendChild(toast);

        // Animate in
        setTimeout(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateX(-50%) translateY(0)';
        }, 10);

        // Animate out and remove
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(-50%) translateY(-20px)';
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 2000);
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

            // Reset for text drawing with scroll offset
            textY = y + padding - scrollOffset;
            ctx.fillStyle = "#e0e0e0";
        }

        // Draw text
        for (const line of lines.slice(0, Math.floor((height - padding * 2) / 16))) {
            ctx.fillText(line, x + padding, textY);
            textY += 16;
        }

        // Cursor (only show when visible and no selection)
        if (isEditing && this.cursorVisible && (this.selectionStart === null || this.selectionEnd === null || this.selectionStart === this.selectionEnd)) {
            // Find which line the cursor is on
            let currentPos = 0;
            let cursorLine = 0;
            let cursorX = x + padding;
            let cursorY = y + padding - scrollOffset;

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const lineEnd = currentPos + line.length;

                if (this.cursorPos <= lineEnd) {
                    // Cursor is on this line
                    cursorLine = i;
                    const posInLine = this.cursorPos - currentPos;
                    const beforeCursor = line.substring(0, posInLine);
                    cursorX = x + padding + ctx.measureText(beforeCursor).width;
                    cursorY = y + padding - scrollOffset + i * 16;
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

        // Draw selection highlight for centered text (before drawing text)
        if (isEditing && this.selectionStart !== null && this.selectionEnd !== null && this.selectionStart !== this.selectionEnd) {
            const start = Math.min(this.selectionStart, this.selectionEnd);
            const end = Math.max(this.selectionStart, this.selectionEnd);

            const textWidth = ctx.measureText(value).width;
            const textStartX = x + width / 2 - textWidth / 2;

            const beforeSel = value.substring(0, start);
            const selected = value.substring(start, end);

            const selX = textStartX + ctx.measureText(beforeSel).width;
            const selWidth = ctx.measureText(selected).width;

            ctx.fillStyle = "#667eea80"; // Semi-transparent blue
            ctx.fillRect(selX, y + height - 20, selWidth, 14);

            // Reset text color
            ctx.fillStyle = isEditing ? "#fff" : "#6ee7b7";
        }

        ctx.fillText(value, x + width / 2, y + height - 6);

        // Cursor (only show when visible and no selection)
        if (isEditing && this.cursorVisible && (this.selectionStart === null || this.selectionEnd === null || this.selectionStart === this.selectionEnd)) {
            // Calculate cursor position for centered text
            const textWidth = ctx.measureText(value).width;
            const textStartX = x + width / 2 - textWidth / 2;
            const cursorX = textStartX + ctx.measureText(value.substring(0, this.cursorPos)).width;
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

        for (const key in this.controls) {
            const ctrl = this.controls[key];
            if (this.isPointInControl(relX, relY, ctrl)) {
                // Check ExpandBtn BEFORE general Btn to prevent it being caught by Btn handler
                if (key.endsWith('ExpandBtn')) {
                    // Open HTML modal for this field
                    const fieldLabel = ctrl.label;
                    const fieldCtrl = this.controls[`${fieldLabel}Field`];
                    if (fieldCtrl && fieldCtrl.widget) {
                        // Stop any current editing
                        if (this.editingField) {
                            this.stopEditing(true);
                        }

                        // Open HTML modal
                        const initialValue = String(fieldCtrl.widget.value || "");
                        this.openHTMLModal(fieldLabel, initialValue, fieldCtrl.widget);
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
                        let cursorPos;

                        if (ctrl.isNumber) {
                            // Number fields use centered text
                            cursorPos = this.calculateCursorPositionCentered(
                                relX,
                                this.editingValue,
                                ctrl.x,
                                ctrl.w
                            );
                        } else {
                            // Text fields use left-aligned text
                            const scrollOffset = this.scrollOffsets[ctrl.label] || 0;
                            const clickX = Math.max(0, relX - ctrl.textX); // Clamp to 0 to handle padding area
                            const clickY = Math.max(0, relY - ctrl.textY + scrollOffset); // Clamp to 0

                            cursorPos = this.calculateCursorPosition(clickX, clickY, this.editingValue, ctrl.w - 16);
                        }

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
                    let cursorPos;

                    if (ctrl.isNumber) {
                        // Number fields use centered text - calculate differently
                        cursorPos = this.calculateCursorPositionCentered(
                            relX,
                            initialValue,
                            ctrl.x,
                            ctrl.w
                        );
                    } else {
                        // Text fields use left-aligned text
                        cursorPos = this.calculateCursorPosition(
                            Math.max(0, relX - ctrl.textX), // Clamp to 0 to handle padding area
                            Math.max(0, relY - ctrl.textY), // Clamp to 0
                            initialValue,
                            ctrl.w - 16
                        );
                    }

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
        // Only handle keyboard if we're editing a field (not in HTML modal)
        if (!this.editingField || this.htmlModalOpen) return;

        // Handle Ctrl/Cmd shortcuts
        const isCtrl = e.ctrlKey || e.metaKey;

        if (isCtrl && e.key === 'a') {
            // Select all - prevent default to stop browser selection
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            this.selectionStart = 0;
            this.selectionEnd = this.editingValue.length;
            this.cursorPos = this.editingValue.length;
            this.cursorVisible = true; // Make cursor visible
            this.node.setDirtyCanvas(true, true);
            return;
        } else if (isCtrl && e.key === 'c') {
            // Copy - allow default but also handle manually
            if (this.selectionStart !== null && this.selectionEnd !== null && this.selectionStart !== this.selectionEnd) {
                const selectedText = this.editingValue.substring(
                    Math.min(this.selectionStart, this.selectionEnd),
                    Math.max(this.selectionStart, this.selectionEnd)
                );
                navigator.clipboard.writeText(selectedText).catch(err => console.error('Copy failed:', err));
            }
            return;
        } else if (isCtrl && e.key === 'v') {
            // Paste
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            navigator.clipboard.readText().then(text => {
                if (!this.editingField) return; // Safety check

                if (this.selectionStart !== null && this.selectionEnd !== null && this.selectionStart !== this.selectionEnd) {
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
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            }).catch(err => console.error('Paste failed:', err));
            return;
        } else if (isCtrl && e.key === 'x') {
            // Cut
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();

            if (this.selectionStart !== null && this.selectionEnd !== null && this.selectionStart !== this.selectionEnd) {
                const start = Math.min(this.selectionStart, this.selectionEnd);
                const end = Math.max(this.selectionStart, this.selectionEnd);
                const selectedText = this.editingValue.substring(start, end);
                navigator.clipboard.writeText(selectedText).catch(err => console.error('Cut failed:', err));
                this.editingValue = this.editingValue.substring(0, start) + this.editingValue.substring(end);
                this.cursorPos = start;
                this.selectionStart = null;
                this.selectionEnd = null;
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            }
            return;
        }

        // For all other keys, prevent default to stop ComfyUI shortcuts
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

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
            this.resetCursorBlink();
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
            if (isCtrl) {
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
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            } else {
                // For number fields, Enter saves
                this.stopEditing(true);
            }
            return;
        } else if (e.key === "Escape") {
            // Cancel editing
            this.stopEditing(false);
            return;
        } else if (e.key === "Backspace") {
            if (this.cursorPos > 0) {
                this.editingValue = this.editingValue.slice(0, this.cursorPos - 1) + this.editingValue.slice(this.cursorPos);
                this.cursorPos--;
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "Delete") {
            if (this.cursorPos < this.editingValue.length) {
                this.editingValue = this.editingValue.slice(0, this.cursorPos) + this.editingValue.slice(this.cursorPos + 1);
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "ArrowLeft") {
            if (this.cursorPos > 0) {
                this.cursorPos--;
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "ArrowRight") {
            if (this.cursorPos < this.editingValue.length) {
                this.cursorPos++;
                this.resetCursorBlink();
                this.node.setDirtyCanvas(true, true);
            }
        } else if (e.key === "Home") {
            this.cursorPos = 0;
            this.resetCursorBlink();
            this.node.setDirtyCanvas(true, true);
        } else if (e.key === "End") {
            this.cursorPos = this.editingValue.length;
            this.resetCursorBlink();
            this.node.setDirtyCanvas(true, true);
        } else if (e.key.length === 1 && !isCtrl) {
            // Only handle printable characters (not Ctrl+C, etc.)
            this.editingValue = this.editingValue.slice(0, this.cursorPos) + e.key + this.editingValue.slice(this.cursorPos);
            this.cursorPos++;
            this.resetCursorBlink();
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

        // Don't show tooltips when lightbox is open
        if (this.lightboxOpen) return;

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

    calculateCursorPosition(clickX, clickY, text, maxWidth, lineHeight = 16, font = "12px 'Courier New', monospace") {
        // Create a temporary canvas to measure with the correct font
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.font = font;

        // Wrap text into lines
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

            // Handle clicks before text starts (negative or very small clickX)
            if (clickX < 0) {
                return charsBeforeLine; // Position at start of line
            }

            for (let i = 0; i <= line.length; i++) {
                const charWidth = i < line.length ? tempCtx.measureText(line[i]).width : 0;
                // Click is between this character and the next if clickX is within the character's width
                if (i === line.length || totalWidth + charWidth / 2 > clickX) {
                    charPosInLine = i;
                    break;
                }
                totalWidth += charWidth;
            }

            return Math.min(charsBeforeLine + charPosInLine, text.length);
        } else {
            // Click below or above all lines
            return lineIndex < 0 ? 0 : text.length;
        }
    }

    calculateCursorPositionCentered(clickX, text, fieldX, fieldWidth) {
        // For centered single-line text (used in number fields)
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.font = "bold 13px Arial"; // Same font as number fields

        const textWidth = tempCtx.measureText(text).width;
        const textStartX = fieldX + fieldWidth / 2 - textWidth / 2;

        // Click is relative to text start
        const relativeX = clickX - textStartX;

        // Find which character was clicked
        let accumulatedWidth = 0;
        for (let i = 0; i <= text.length; i++) {
            if (i === text.length || accumulatedWidth + tempCtx.measureText(text[i]).width / 2 > relativeX) {
                return i;
            }
            accumulatedWidth += tempCtx.measureText(text[i]).width;
        }

        return text.length;
    }

    handleMouseDrag(e, pos, canvas) {
        if (!this.isDragging) return false;

        const node = this.node;
        const relX = e.canvasX - node.pos[0];
        const relY = e.canvasY - node.pos[1];

        // Handle regular field drag selection
        if (!this.editingField) return false;

        // Find the text field control
        const ctrl = this.controls[`${this.editingField}Field`];
        if (!ctrl) return false;

        // Check if mouse is within the field bounds
        if (relX >= ctrl.x && relX <= ctrl.x + ctrl.w &&
            relY >= ctrl.y && relY <= ctrl.y + ctrl.h) {

            let cursorPos;

            if (ctrl.isNumber) {
                // Number fields use centered text
                cursorPos = this.calculateCursorPositionCentered(
                    relX,
                    this.editingValue,
                    ctrl.x,
                    ctrl.w
                );
            } else {
                // Text fields use left-aligned text
                const scrollOffset = this.scrollOffsets[this.editingField] || 0;
                const clickX = Math.max(0, relX - ctrl.textX); // Clamp to 0 to handle padding area
                const clickY = Math.max(0, relY - ctrl.textY + scrollOffset); // Clamp to 0

                cursorPos = this.calculateCursorPosition(clickX, clickY, this.editingValue, ctrl.w - 16);
            }

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
