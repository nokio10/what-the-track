(function (global) {
    "use strict";

    function parseJson(value) {
        try {
            return JSON.parse(value);
        } catch (error) {
            return null;
        }
    }

    function clampReconnectDelay(baseDelay, maxDelay, attempt) {
        var delay = baseDelay * Math.pow(1.5, Math.max(0, attempt - 1));
        return Math.min(delay, maxDelay);
    }

    function normalizeBaseUrl(url) {
        if (!url) {
            var protocol = global.location.protocol === "https:" ? "wss://" : "ws://";
            return protocol + global.location.host;
        }
        return url.replace(/^http/i, "ws");
    }

    function normalizeOptions(options) {
        var normalized = options || {};
        return {
            url: normalized.url || null,
            path: normalized.path || "/socket.io",
            timeout: normalized.timeout || 20000,
            reconnection: normalized.reconnection !== false,
            reconnectionDelay: normalized.reconnectionDelay || 1000,
            reconnectionDelayMax: normalized.reconnectionDelayMax || 5000,
            reconnectionAttempts: normalized.reconnectionAttempts == null
                ? Infinity
                : normalized.reconnectionAttempts,
            auth: normalized.auth || null,
        };
    }

    class LiteSocket {
        constructor(options) {
            this.options = normalizeOptions(options);
            this.handlers = {};
            this.ws = null;
            this.connected = false;
            this.id = null;
            this.engineSid = null;
            this.reconnectAttempt = 0;
            this.manualClose = false;
            this.connectTimeoutId = null;
            this.reconnectTimerId = null;
            this.pingTimerId = null;
            this.pingDeadline = 0;
            this.connectErrorSent = false;
            this._open();
        }

        on(eventName, handler) {
            if (!this.handlers[eventName]) {
                this.handlers[eventName] = [];
            }
            this.handlers[eventName].push(handler);
            return this;
        }

        off(eventName, handler) {
            if (!this.handlers[eventName]) {
                return this;
            }
            this.handlers[eventName] = this.handlers[eventName].filter(function (item) {
                return item !== handler;
            });
            return this;
        }

        emit(eventName) {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                return false;
            }

            var args = Array.prototype.slice.call(arguments, 1);
            var payload = [eventName].concat(args);
            this.ws.send("42" + JSON.stringify(payload));
            return true;
        }

        disconnect() {
            this.manualClose = true;
            this._clearTimers();
            if (this.ws) {
                this.ws.close(1000, "client disconnect");
            }
            return this;
        }

        _dispatch(eventName) {
            var args = Array.prototype.slice.call(arguments, 1);
            var listeners = this.handlers[eventName] || [];
            listeners.forEach(function (listener) {
                try {
                    listener.apply(null, args);
                } catch (error) {
                    global.console.error(error);
                }
            });
        }

        _open() {
            var self = this;
            var baseUrl = normalizeBaseUrl(this.options.url);
            var normalizedPath = this.options.path.replace(/\/+$/, "");
            var socketUrl = baseUrl + normalizedPath + "/?EIO=4&transport=websocket";

            this._clearTimers();
            this.connectErrorSent = false;

            this.ws = new WebSocket(socketUrl);

            this.connectTimeoutId = global.setTimeout(function () {
                self._reportConnectError("Connection timeout");
                if (self.ws) {
                    self.ws.close();
                }
            }, this.options.timeout);

            this.ws.onmessage = function (event) {
                self._handlePacket(String(event.data || ""));
            };

            this.ws.onerror = function () {
                self._reportConnectError("WebSocket error");
            };

            this.ws.onclose = function (event) {
                self._handleClose(event && event.reason ? event.reason : "transport close");
            };
        }

        _handlePacket(packet) {
            if (!packet) {
                return;
            }

            if (packet.charAt(0) === "0") {
                var handshake = parseJson(packet.slice(1));
                this.engineSid = handshake && handshake.sid ? handshake.sid : null;
                // Сервер обязан пинговать раз в pingInterval; если молчит дольше
                // pingInterval + pingTimeout, соединение полуоткрыто (телефон
                // потерял сеть без FIN) — как в socket.io-client, рвём и переподключаемся.
                this.pingDeadline = ((handshake && handshake.pingInterval) || 25000)
                    + ((handshake && handshake.pingTimeout) || 20000);
                this._resetPingTimer();
                this._sendConnect();
                return;
            }

            if (packet === "2") {
                this._resetPingTimer();
                this._sendRaw("3");
                return;
            }

            if (packet.charAt(0) === "4" && packet.charAt(1) === "0") {
                var connectPayload = parseJson(packet.slice(2));
                var reconnectAttempt = this.reconnectAttempt;
                this.connected = true;
                this.id = connectPayload && connectPayload.sid ? connectPayload.sid : this.engineSid;
                this.reconnectAttempt = 0;
                this.connectErrorSent = false;
                this._clearConnectTimeout();
                // Как socket.io-client v4: у сокета после переподключения только
                // 'connect' ('reconnect' там событие менеджера). Раньше приходили оба,
                // и страница игрока отправляла вход в игру дважды.
                this._dispatch("connect");
                return;
            }

            if (packet.charAt(0) === "4" && packet.charAt(1) === "1") {
                this._handleClose("namespace disconnect");
                return;
            }

            if (packet.charAt(0) === "4" && packet.charAt(1) === "2") {
                var eventPayload = parseJson(packet.slice(2));
                if (!Array.isArray(eventPayload) || eventPayload.length === 0) {
                    return;
                }
                this._dispatch.apply(this, eventPayload);
                return;
            }

            if (packet.charAt(0) === "4" && packet.charAt(1) === "4") {
                var errorPayload = parseJson(packet.slice(2));
                var message = errorPayload && errorPayload.message
                    ? errorPayload.message
                    : "Socket.IO connect error";
                this._reportConnectError(message);
            }
        }

        _handleClose(reason) {
            var wasConnected = this.connected;

            this._clearConnectTimeout();
            this._clearPingTimer();
            this.connected = false;
            this.id = null;

            if (wasConnected) {
                this._dispatch("disconnect", reason);
            }

            if (!this.manualClose) {
                this._scheduleReconnect();
            }
        }

        _scheduleReconnect() {
            if (!this.options.reconnection) {
                return;
            }

            if (this.reconnectAttempt >= this.options.reconnectionAttempts) {
                return;
            }

            var self = this;
            this.reconnectAttempt += 1;

            this.reconnectTimerId = global.setTimeout(function () {
                self._open();
            }, clampReconnectDelay(
                this.options.reconnectionDelay,
                this.options.reconnectionDelayMax,
                this.reconnectAttempt
            ));
        }

        _reportConnectError(message) {
            if (this.connectErrorSent) {
                return;
            }
            this.connectErrorSent = true;
            this._dispatch("connect_error", new Error(message));
        }

        _resetPingTimer() {
            var self = this;
            this._clearPingTimer();
            if (!this.pingDeadline) {
                return;
            }
            this.pingTimerId = global.setTimeout(function () {
                self._dropSocket();
                self._handleClose("ping timeout");
            }, this.pingDeadline);
        }

        _clearPingTimer() {
            if (this.pingTimerId) {
                global.clearTimeout(this.pingTimerId);
                this.pingTimerId = null;
            }
        }

        _dropSocket() {
            // Отвязываем обработчики до close(): иначе onclose вызвал бы
            // _handleClose второй раз и запланировал бы два переподключения.
            var ws = this.ws;
            this.ws = null;
            if (ws) {
                ws.onopen = ws.onmessage = ws.onerror = ws.onclose = null;
                try {
                    ws.close();
                } catch (error) {
                    global.console.error(error);
                }
            }
        }

        _sendConnect() {
            // Как socket.io-client: auth уходит в пакете CONNECT; функция
            // вызывается при каждом подключении и отдаёт свежее значение.
            var self = this;
            var send = function (data) {
                self._sendRaw("40" + (data ? JSON.stringify(data) : ""));
            };
            var auth = this.options.auth;
            if (typeof auth === "function") {
                auth(send);
            } else {
                send(auth);
            }
        }

        _sendRaw(payload) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(payload);
            }
        }

        _clearConnectTimeout() {
            if (this.connectTimeoutId) {
                global.clearTimeout(this.connectTimeoutId);
                this.connectTimeoutId = null;
            }
        }

        _clearTimers() {
            this._clearConnectTimeout();
            this._clearPingTimer();
            if (this.reconnectTimerId) {
                global.clearTimeout(this.reconnectTimerId);
                this.reconnectTimerId = null;
            }
        }
    }

    global.io = function (options) {
        return new LiteSocket(options);
    };
})(window);
