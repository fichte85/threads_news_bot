#!/usr/bin/env bash
set -euo pipefail

SESSION_PID="$(pgrep -n xfce4-session || true)"
if [[ -z "$SESSION_PID" ]]; then
  echo "xfce4-session not found. 먼저 VNC 세션(:1)에 로그인하세요." >&2
  exit 1
fi

DBUS_ADDR="$(tr '\0' '\n' < /proc/$SESSION_PID/environ | sed -n 's/^DBUS_SESSION_BUS_ADDRESS=//p')"
DISPLAY_VAL="$(tr '\0' '\n' < /proc/$SESSION_PID/environ | sed -n 's/^DISPLAY=//p')"
XDG_RUNTIME="$(tr '\0' '\n' < /proc/$SESSION_PID/environ | sed -n 's/^XDG_RUNTIME_DIR=//p')"

if [[ -z "$DBUS_ADDR" || -z "$DISPLAY_VAL" ]]; then
  echo "세션 환경(DBUS/DISPLAY)을 찾지 못했습니다." >&2
  exit 1
fi

pkill -f "chromium.*remote-debugging-port=9223" || true

exec env \
  DISPLAY="$DISPLAY_VAL" \
  DBUS_SESSION_BUS_ADDRESS="$DBUS_ADDR" \
  XDG_RUNTIME_DIR="${XDG_RUNTIME:-/run/user/$(id -u)}" \
  LANG=ko_KR.UTF-8 \
  LANGUAGE=ko_KR:ko \
  LC_ALL=ko_KR.UTF-8 \
  GTK_IM_MODULE=ibus \
  QT_IM_MODULE=ibus \
  XMODIFIERS=@im=ibus \
  /usr/bin/chromium \
    --remote-debugging-address=127.0.0.1 \
    --remote-debugging-port=9223 \
    --user-data-dir=/home/ubuntu/.config/fichte_news \
    --enable-features=UseOzonePlatform \
    --ozone-platform=x11 \
    --no-first-run \
    --no-default-browser-check
